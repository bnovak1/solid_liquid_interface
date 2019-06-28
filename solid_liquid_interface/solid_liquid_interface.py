# added by pasteurize
#########################################################################################
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import open
from builtins import int
from builtins import range
from future import standard_library
standard_library.install_aliases()
#########################################################################################

import lmfit
import mdtraj
import numpy as np
import pandas as pd
import scipy.constants as constants
import scipy.fftpack as fft
import scipy.interpolate as interp
import scipy.spatial as ss
import scipy.special as spec
import subprocess


def get_ref_vectors(n_neighbors, reference_structure, topfile):

    # Read in reference structure
    reference_structure = mdtraj.load_lammpstrj(reference_structure, top=topfile)
    natoms = reference_structure.n_atoms
    coords = 100.0*reference_structure.xyz*reference_structure.unitcell_lengths
    coords = coords[0, :, :]

    # Find center atom
    center = np.mean(coords, axis=0)
    center_ind = np.argmin(np.sqrt(np.sum((coords - center)**2.0, axis=1)))

    # Construct k-d tree using coordinates
    tree = ss.cKDTree(coords)

    # Find neighbors of center atom
    (neighbor_distances, neighbors) = tree.query(coords[center_ind, :], n_neighbors+1)

    # Vectors from center atom to its neighbors
    vectors_ref = coords[neighbors[1:], :] - coords[neighbors[0], :]

    # Construct k-d tree for those vectors
    tree_ref = ss.cKDTree(vectors_ref)

    return (vectors_ref, tree_ref)


def visualize_bins(box_sizes, latparam, interface_options, height):

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import my_plot_settings_article as mpsa

    layer_width = interface_options['layer_width']
    nbins_per_layer = interface_options['nbins_per_layer']

    ax = plt.axes(projection='3d')
    (X, Y) = np.mgrid[0:box_sizes[0][0]:latparam/2, 0:box_sizes[0][1]:latparam/2]

    colors = ['b', 'g', 'r', 'k', 'm', 'c', 'y']
    for ilayer in range(interface_options['n_layers']-1):
        for ibin in range(nbins_per_layer):
            ax.plot_surface(X, Y,
                            height[:, :, 0] + ilayer*layer_width + ibin*layer_width/nbins_per_layer,
                            color=colors[ibin])
    ax.plot_surface(X, Y, height[:, :, 0]+ (ilayer+1)*layer_width, color=colors[0])
    ax.plot_wireframe(X, Y, height[:, :, 0]+ (ilayer+1)*layer_width)

    ax.set_xlabel('x ($\\mathrm{\\AA}$)', labelpad=mpsa.axeslabelpad)
    ax.set_ylabel('y ($\\mathrm{\\AA}$)', labelpad=mpsa.axeslabelpad)
    ax.set_zlabel('z ($\\mathrm{\\AA}$)', labelpad=mpsa.axeslabelpad)
    ax.dist = 14

    mpsa.save_figure('bins.png', 300)
    plt.close()


def save_pdb(traj_file, coords, snapshot, interface_options, interfaces):

        natoms = coords.shape[0]
        n_layers = interface_options['n_layers']
        bfactors = np.zeros(natoms)

        for iint in range(2):

            beta = 0.0

            for ilayer in range(n_layers):

                beta += 1.0

                ind = interfaces[iint, ilayer]
                bfactors[ind] = beta


        traj_prefix = '.'.join(traj_file.split('.')[:-1])
        traj_prefix = traj_prefix.split('/')[-1]

        pdb = mdtraj.formats.PDBTrajectoryFile( \
            interface_options['traj_dir'] + traj_prefix + '.pdb', mode='w')
        pdb.write(coords, snapshot.top, unitcell_lengths=tuple(10.0*snapshot.unitcell_lengths[0]),
                  unitcell_angles=tuple(snapshot.unitcell_angles[0]), bfactors=bfactors)
        pdb.close()


def interface_concentrations(snapshot, interfaces, interface_options):

    nbins = interfaces.shape[1]
    atom_names = interface_options['atom_names']
    atom_name_list = interface_options['atom_name_list']
    n_names = len(atom_name_list)

    concs = np.zeros((n_names-1)*nbins*2)

    cnt = 0
    for ibin in range(nbins):
        for iint in range(2):

            ind = interfaces[iint, ibin]
            natoms = float(len(ind))

            for iname in range(n_names-1):
                concs[cnt] = np.sum(atom_names[ind] == \
                                    atom_name_list[iname])/natoms
                cnt += 1

    return concs


def itim(snapshot, coords, phase_atoms, interface_options):
    '''
    FIX: STILL USES PHASES
    '''

    import pytim

    r_atoms = interface_options['r_atoms']
    r_probe = interface_options['ITIM']['r_probe']
    grid_spacing = interface_options['ITIM']['grid_spacing']
    n_layers = interface_options['n_layers']

    interfaces = np.empty(2, dtype=object)

    # Remove extra space from box so pytim can find the correct interfaces
    if interface_options['free_boundaries']:
        shift = np.min(snapshot.xyz[0][:, 2])
        snapshot.xyz[0][:, 2] -= shift
        box_shift = snapshot.unitcell_lengths[0][2] - np.max(snapshot.xyz[0][:, 2])
        snapshot.unitcell_lengths[0][2] -= box_shift

    # Find interfacial atoms
    for iphase in range(2):
        interfaces[iphase] = pytim.ITIM(snapshot, group=phase_atoms[iphase], normal='z',
                                        molecular=False, alpha=r_probe,
                                        radii_dict=r_atoms, mesh=grid_spacing,
                                        max_layers=n_layers)

    # Change box back
    if interface_options['free_boundaries']:
        snapshot.unitcell_lengths[0][2] += box_shift
        snapshot.xyz[0][:, 2] += shift

    # Get lists instead of MDAnalysis objects
    interface_atoms = np.empty((2, 2, n_layers), dtype=object)

    cnt = 0
    for ilayer in range(n_layers):
        for iphase in range(2):

            if iphase==0:
    	        interface_atoms[0, iphase, ilayer] = \
                    interfaces[iphase].layers[0, ilayer].indices
    	        interface_atoms[1, iphase, ilayer] = \
                    interfaces[iphase].layers[1, ilayer].indices
            else:
    	        interface_atoms[0, iphase, ilayer] = \
                    interfaces[iphase].layers[1, ilayer].indices
    	        interface_atoms[1, iphase, ilayer] = \
                    interfaces[iphase].layers[0, ilayer].indices

    return interface_atoms


def find_interfacial_atoms_2D_nearest(x, y, height, coords, traj_file, snapshot,
                                      interface_options, latparam):

    n_layers = interface_options['n_layers']
    nbins_per_layer = interface_options['nbins_per_layer']
    nbins = (n_layers - 1)*nbins_per_layer + 1
    bin_width = interface_options['layer_width']

    interfaces = np.empty((2, nbins), dtype=object)


    for iint in range(2):

        z_interp = interp.RectBivariateSpline(x, y, height[:, :, iint])

        atoms = np.intersect1d(np.where(coords[:, 2] > np.min(height[:, :, iint]) - \
                                        bin_width*n_layers/2.0)[0],
                               np.where(coords[:, 2] < np.max(height[:, :, iint]) + \
                                        bin_width*n_layers/2.0)[0])

        interface_projections = z_interp.ev(coords[atoms, 0], coords[atoms, 1])

        for ibin in range(nbins):

            lower_shift = -bin_width*(n_layers/2.0 - ibin/nbins_per_layer)
            upper_shift = lower_shift + bin_width

            ind = np.intersect1d(np.where(coords[atoms, 2] >= interface_projections + lower_shift)[0],
                                 np.where(coords[atoms, 2] < interface_projections + upper_shift)[0])

            interfaces[iint, ibin] = atoms[ind]

    interfaces[1, :] = interfaces[1, ::-1]

    return interfaces


def find_interfacial_atoms_2D_itim(x, y, height, coords, traj_file, snapshot,
                                   interface_options):

    # Split phases
    phase_atoms = np.empty(2, dtype=object)

    z_interp = interp.RectBivariateSpline(x, y, height[:, :, 0])
    interface_positions = z_interp.ev(coords[:, 0], coords[:, 1])
    phase_atoms[0] = np.where(coords[:, 2] < interface_positions)[0]

    z_interp = interp.RectBivariateSpline(x, y, height[:, :, 1])
    interface_positions = z_interp.ev(coords[:, 0], coords[:, 1])
    phase_atoms[0] = np.hstack((phase_atoms[0],
                                np.where(coords[:, 2] > interface_positions)[0]))

    phase_atoms[1] = np.setxor1d(phase_atoms[0], range(coords.shape[0]))


    # Interfacial atoms
    interfaces = itim(snapshot, coords, phase_atoms, interface_options)

    return interfaces


def find_interfacial_atoms_1D(x, height, coords, traj_file, snapshot, interface_options):
    '''
    FIX: STILL USES PHASES
    '''

    # Split phases
    phase_atoms = np.empty(2, dtype=object)

    z_interp_lower = interp.UnivariateSpline(x, height[:, 0], s=0)
    interface_positions = z_interp_lower(coords[:, 0])
    phase_atoms[0] = np.where(coords[:, 2] < interface_positions)[0]

    z_interp_upper = interp.UnivariateSpline(x, height[:, 1], s=0)
    interface_positions = z_interp_upper(coords[:, 0])
    phase_atoms[0] = np.hstack((phase_atoms[0],
                                np.where(coords[:, 2] > interface_positions)[0]))

    phase_atoms[1] = np.setxor1d(phase_atoms[0], range(coords.shape[0]))

    # Interfacial atoms
    interfaces = itim(snapshot, coords, phase_atoms, interface_options)

    return interfaces


def erf_fit_func(order_param_sol, order_param_liq, pos_bound, pos,
                 pos_interface_lower, pos_interface_upper,
                 sigma_lower, sigma_upper, erf_sign):

    order_param_fit = \
        0.5*((order_param_sol + order_param_liq) + \
             erf_sign*(pos < pos_bound)*(order_param_sol - order_param_liq)* \
             spec.erf((pos - pos_interface_lower)/(sigma_lower*np.sqrt(2.0))) - \
             erf_sign*(pos > pos_bound)*(order_param_sol - order_param_liq)* \
             spec.erf((pos - pos_interface_upper)/(sigma_upper*np.sqrt(2.0))))

    return order_param_fit


def residual(params, pos, order_param, wghts, erf_sign):
    """
    Description
    ----
    Compute residuals for fit.

    Inputs
    ----
    :params: Parameters for the model
    :pos: Position in interface normal direction
    :order_param: Order parameter to distinguish liquid from solid
    :wghts: Weights
    :erf_sign: 1 for solid in center of box and -1 for liquid in center of box

    Outputs
    ----
    :residuals: Residuals
    """

    order_param_sol = params['order_param_sol'].value
    order_param_liq = params['order_param_liq'].value
    sigma_lower = params['sigma_lower'].value
    sigma_upper = params['sigma_upper'].value
    pos_interface_lower = params['pos_interface_lower'].value
    pos_interface_upper = params['pos_interface_upper'].value
    pos_bound = params['pos_bound'].value

    model = erf_fit_func(order_param_sol, order_param_liq, pos_bound, pos,
                         pos_interface_lower, pos_interface_upper,
                         sigma_lower, sigma_upper, erf_sign)

    residuals = (order_param - model)*wghts

    return residuals


def fitting_erf(pos, order_param, erf_sign):

    wghts = np.ones(len(pos))  # All weights equal
    params = lmfit.Parameters()

    pos_min = np.min(pos)
    delta_pos = np.max(pos) - pos_min
    order_param_sol_ini = np.max(order_param)
    order_param_liq_ini = np.min(order_param)
    fliq = (np.mean(order_param) - order_param_sol_ini)/ \
           (order_param_liq_ini - order_param_sol_ini)
    fsol = 1 - fliq
    pos_interface_lower_ini = delta_pos*((fliq/2)*(erf_sign==1) + \
                                         (fsol/2)*(erf_sign==-1)) + pos_min
    pos_interface_upper_ini = delta_pos*((1 - fliq/2)*(erf_sign==1) + \
                                         (1 - fsol/2)*(erf_sign==-1)) + pos_min

    params.add('order_param_sol', value=order_param_sol_ini, min=0.0)
    params.add('order_param_liq', value=order_param_liq_ini, min=0.0)
    params.add('pos_interface_lower', value=pos_interface_lower_ini)
    params.add('pos_interface_upper', value=pos_interface_upper_ini)
    params.add('sigma_lower', value=3.0, min=0.0)
    params.add('sigma_upper', value=3.0, min=0.0)
    params.add('pos_bound', value=np.mean(pos))

    fit = lmfit.minimize(residual, params, args=(pos, order_param, wghts, erf_sign))

    crossover = (fit.params['order_param_liq'].value + \
                 fit.params['order_param_sol'].value)/2
    multiplier = 2*np.sqrt(2)*spec.erfinv(0.99)
    interface_widths = multiplier*np.array([fit.params['sigma_upper'].value,
                                            fit.params['sigma_lower'].value])

    # import matplotlib.pyplot as plt
    # plt.plot(pos, order_param)
    # plt.plot(pos, erf_fit_func(fit.params['order_param_sol'].value,
    #                            fit.params['order_param_liq'].value,
    #                            fit.params['pos_bound'].value,
    #                            pos,
    #                            fit.params['pos_interface_lower'].value,
    #                            fit.params['pos_interface_upper'].value,
    #                            fit.params['sigma_lower'].value,
    #                            fit.params['sigma_upper'].value,
    #                            erf_sign))
    # plt.show()
    # import pdb; pdb.set_trace()

    return (crossover, interface_widths)


def interface_positions_2D(frame_num, coords, box_sizes, snapshot, n_neighbors, latparam,
                           vectors_ref, tree_ref, X, Y, Z, smoothing_cutoff,
                           interface_options, outfile_prefix, crossover=None,
                           reduce_flag=True, interface_range=None):

    natoms = coords.shape[0]
    nx_grid = X.shape[0]
    ny_grid = X.shape[1]

    # Apply periodic boundary conditions to insure all coordinates are in [0, box_size].
    # This is required for the k-d tree algorithm with periodic boundary conditions.
    # k-d tree cannot have coordinates exactly at upper boundary, shift to lower boundary
    coords -= box_sizes*np.floor(coords/box_sizes)
    coords -= box_sizes*(coords == box_sizes)

    # Keep only coordinates and grid points near interfaces
    if reduce_flag and frame_num > 0:

        ind = (coords[:, 2] >= interface_range[0, 0])* \
              (coords[:, 2] <= interface_range[0, 1]) + \
              (coords[:, 2] >= interface_range[1, 0])* \
              (coords[:, 2] <= interface_range[1, 1])
        coords = coords[ind, :]
        natoms = coords.shape[0]

        shift = smoothing_cutoff + 2.0*latparam
        ind = ((Z >= interface_range[0, 0] + shift)* \
               (Z <= interface_range[0, 1] - shift) + \
               (Z >= interface_range[1, 0] + shift)* \
               (Z <= interface_range[1, 1] - shift))
        X = X[ind].reshape(nx_grid, ny_grid, -1)
        Y = Y[ind].reshape(nx_grid, ny_grid, -1)
        Z = Z[ind].reshape(nx_grid, ny_grid, -1)

    # List of points to compute psi on
    nz_grid = X.shape[2]
    psi_grid = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # Construct k-d tree using coordinates
    tree = ss.cKDTree(coords, boxsize=box_sizes)

    # Get neighbor distances and neighbors
    (neighbor_distances, neighbors) = tree.query(coords, n_neighbors+1)

    # Vectors from each atom to its neighbors. Use periodic boundary conditions.
    vectors = coords[neighbors[:, 1:], :] - \
                 coords[neighbors[:, 0], :].reshape(natoms, 1, 3)
    vectors -= box_sizes*np.round(vectors/box_sizes, 0)
    del neighbor_distances, neighbors

    # Calculate psi

    # assign vector to closest reference vector
    (vector_distances, vector_neighbors) = tree_ref.query(vectors, 1)
    phi = np.sum(vector_distances**2, axis=1)/n_neighbors
    del vector_distances, vector_neighbors, tree_ref

    try:

        # Neighbor distances (dist) and neighbors (ii)
        (dist, ii) = tree.query(psi_grid, interface_positions_2D.n_neighbors)

    except AttributeError:

        # First frame. Find neighbors within cutoff for each atom
        neighbors = tree.query_ball_point(psi_grid, smoothing_cutoff)

        # Use 1.05 times the maximum number of neighbors for subsequent frames.
        interface_positions_2D.n_neighbors = int(1.05*np.max([len(n) for n in neighbors]))

        # Neighbor distances (dist) and neighbors (ii)
        (dist, ii) = tree.query(psi_grid, interface_positions_2D.n_neighbors)

    # Weights for smoothing. Give points beyond cutoff a zero weight.
    wd = ((1.0 - (dist/smoothing_cutoff)**2)**2)*dist*(dist < smoothing_cutoff)
    del dist, tree

    # psi
    psi = (np.sum(wd*phi[ii], axis=1)/np.sum(wd, axis=1)).reshape(nx_grid, ny_grid, nz_grid)/(latparam**2)
    del wd, ii

    # Save phi and psi from 1st frame and 1 grid point for plotting and testing
    # Calculate and save interface width estimate for 1st frame
    if frame_num == 0:

        ind = np.intersect1d(np.where(psi_grid[:, 0] > 0)[0],
                             np.where(psi_grid[:, 1] > 0)[0])
        grid_point = psi_grid[ind[0], :2]

        ind = (psi_grid[:, 0] == grid_point[0])*(psi_grid[:, 1] == grid_point[1])
        outdata = np.column_stack((psi_grid[ind, 2],
                                   psi.reshape(nx_grid*ny_grid*nz_grid, -1)[ind]))
        np.savetxt(outfile_prefix + '_psi.dat', outdata)
        del outdata

        ind = (np.abs(coords[:, 0] - grid_point[0]) < latparam/4.0)* \
              (np.abs(coords[:, 1] - grid_point[1]) < latparam/4.0)
        outdata = np.column_stack((coords[ind, 2], phi[ind]/latparam**2.0))
        np.savetxt(outfile_prefix + '_phi.dat', outdata)
        del outdata

        zpos = Z[0, 0, :]
        psi_flat = psi.reshape(-1, psi.shape[2])
        erf_sign = 2*(psi_flat[0, int(len(zpos)/2)] > psi_flat[0, 0]) - 1
        nfits = psi_flat.shape[0]

        interface_widths = np.array([fitting_erf(zpos, psi_flat[ifit, :], erf_sign)[1] \
                                     for ifit in range(nfits)]).flatten()
        del zpos, psi_flat
        np.savetxt('interface_width.dat', [np.mean(interface_widths)])

    # Crossover value for psi by fitting to error functions
    if (reduce_flag and frame_num == 0) or not reduce_flag:
        zpos = Z[0, 0, :]
        psi_mean = np.mean(np.mean(psi, axis=0), axis=0)
        erf_sign = 2*(psi_mean[int(len(psi_mean)/2)] > psi_mean[0]) - 1
        (crossover, _) = fitting_erf(zpos, psi_mean, erf_sign)
        if frame_num == 0:
            np.savetxt('crossover.txt', [crossover])

    # import matplotlib.pyplot as plt
    # plt.plot(Z[0, 0, :], np.mean(np.mean(psi, axis=0), axis=0), '.-')
    # plt.savefig(str(frame_num) + '.png')
    # plt.close()

    # Interface heights for each interface
    psi_grid = psi_grid.reshape(nx_grid, ny_grid, nz_grid, 3)
    height = np.zeros((nx_grid, ny_grid, 2))
    for ix in range(nx_grid):
        for iy in range(ny_grid):

            # Indices where psi is greater than the cross over value
            ind = np.where(psi[ix, iy, :] > crossover)[0] # Assume liquid is in center
            if ind[0] == 0: # Solid is actually in center
                ind = np.where(psi[ix, iy, :] < crossover)[0]

            # Height for first crossing (lower interface)
            ind_crossing = range(ind[0]-1, ind[0]+1)
            fit = np.polyfit(psi[ix, iy, ind_crossing], psi_grid[ix, iy, ind_crossing, 2], 1)
            height[ix, iy, 0] = fit[0]*crossover + fit[1]

            # Height for second crossing (upper interface)
            ind_crossing = range(ind[-1], ind[-1]+2)
            fit = np.polyfit(psi[ix, iy, ind_crossing], psi_grid[ix, iy, ind_crossing, 2], 1)
            height[ix, iy, 1] = fit[0]*crossover + fit[1]

    if frame_num == 0:
        interface_range_2D(height, smoothing_cutoff, latparam)

    return height


def interface_range_2D(height, smoothing_cutoff, latparam):

    hmin = np.min(np.min(height, axis=0), axis=0)
    hmax = np.max(np.max(height, axis=0), axis=0)

    hrng_half = 5.0*np.max(hmax - hmin)/2.0 + smoothing_cutoff + 2.0*latparam
    hmean = np.mean(np.mean(height, axis=0), axis=0)
    interface_range = np.column_stack((hmean - hrng_half, hmean + hrng_half))

    np.savetxt('interface_range.txt', interface_range)


def interface_positions_1D(frame_num, coords, box_sizes, snapshot, n_neighbors, latparam,
                           vectors_ref, tree_ref, X, Z, smoothing_cutoff, interface_options,
                           outfile_prefix, crossover=None, reduce_flag=True):

    natoms = snapshot.n_atoms
    nx_grid = X.shape[0]

    # Apply periodic boundary conditions to insure all coordinates are in [0, box_size].
    # This is required for the k-d tree algorithm with periodic boundary conditions.
    # k-d tree cannot have coordinates exactly at upper boundary, shift to lower boundary
    coords -= box_sizes*np.floor(coords/box_sizes)
    coords -= box_sizes*(coords == box_sizes)

    # Keep only coordinates and grid points near interfaces
    try:

        ind = (coords[:, 2] >= interface_positions_1D.interface_range[0, 0])* \
              (coords[:, 2] <= interface_positions_1D.interface_range[0, 1]) + \
              (coords[:, 2] >= interface_positions_1D.interface_range[1, 0])* \
              (coords[:, 2] <= interface_positions_1D.interface_range[1, 1])
        coords = coords[ind, :]
        natoms = coords.shape[0]

        shift = smoothing_cutoff + 2.0*latparam
        ind = ((Z >= interface_positions_1D.interface_range[0, 0] + shift)* \
               (Z <= interface_positions_1D.interface_range[0, 1] - shift) + \
               (Z >= interface_positions_1D.interface_range[1, 0] + shift)* \
               (Z <= interface_positions_1D.interface_range[1, 1] - shift))
        X = X[ind].reshape(nx_grid, -1)
        Z = Z[ind].reshape(nx_grid, -1)

    except AttributeError:
        pass

    # List of points in the x-z plane to compute psi on
    nz_grid = X.shape[1]
    psi_grid = np.vstack((X.flatten(), Z.flatten())).T

    # Construct k-d tree using coordinates
    tree = ss.cKDTree(coords, boxsize=box_sizes)

    # Get neighbor distances and neighbors
    (neighbor_distances, neighbors) = tree.query(coords, n_neighbors+1)

    # Vectors from each atom to its neighbors. Use periodic boundary conditions.
    vectors = coords[neighbors[:, 1:], :] - \
                 coords[neighbors[:, 0], :].reshape(natoms, 1, 3)
    vectors -= box_sizes*np.round(vectors/box_sizes, 0)
    del neighbor_distances, neighbors

    # Calculate phi

    # assign vector to closest reference vector
    (vector_distances, vector_neighbors) = tree_ref.query(vectors, 1)
    phi = np.sum(vector_distances**2, axis=1)/n_neighbors
    del vector_distances, vector_neighbors, tree_ref

    # Create k-d tree for x and z coordinates with periodic boundaries
    tree_xz = ss.cKDTree(coords[:, [0, 2]], boxsize=box_sizes[0, [0, 2]])

    try:

        # Neighbor distances (dist) and neighbors (ii)
        (dist, ii) = tree_xz.query(psi_grid, interface_positions_1D.n_neighbors)

    except AttributeError:

        # First frame. Find neighbors within cutoff for each atom
        neighbors = tree_xz.query_ball_point(psi_grid, smoothing_cutoff)

        # Use 1.05 times the maximum number of neighbors for subsequent frames.
        interface_positions_1D.n_neighbors = int(1.05*np.max([len(n) for n in neighbors]))

        # Neighbor distances (dist) and neighbors (ii)
        (dist, ii) = tree_xz.query(psi_grid, interface_positions_1D.n_neighbors)

    # Weights for smoothing. Give points beyond cutoff a zero weight.
    wd = ((1.0 - (dist/smoothing_cutoff)**2)**2)*dist*(dist < smoothing_cutoff)
    del dist
    if not interface_options['interface_flag']: del tree_xz

    # psi
    psi = (np.sum(wd*phi[ii], axis=1)/np.sum(wd, axis=1)).reshape(nx_grid, nz_grid)/(latparam**2)
    del wd, ii

    # Save phi and psi from 1st frame for plotting and testing
    # Calculate and save interface width estimate for 1st frame
    if frame_num == 0:

        ind = np.where(np.abs(coords[:, 0] - psi_grid[nz_grid, 0]) < latparam/4.0)[0]
        outdata = np.column_stack((coords[ind, 2], phi[ind]/latparam**2.0))
        np.savetxt(outfile_prefix + '_phi.dat', outdata)
        outdata = np.column_stack((psi_grid[nz_grid:2*nz_grid, 1], psi[1, :]))
        np.savetxt(outfile_prefix + '_psi.dat', outdata)

        zpos = Z[0, :]
        erf_sign = 2*(psi[0, int(len(zpos)/2)] > psi[0, 0]) - 1
        nfits = psi.shape[0]
        interface_widths = np.array([fitting_erf(zpos, psi[ifit, :], erf_sign)[1] \
                                     for ifit in range(nfits)]).flatten()
        np.savetxt('interface_width.dat', [np.mean(interface_widths)])

    # Crossover value for psi by fitting to error functions
    if (reduce_flag and frame_num == 0) or not reduce_flag:
        zpos = Z[0, :]
        psi_mean = np.mean(psi, axis=0)
        erf_sign = 2*(psi_mean[int(len(psi_mean)/2)] > psi_mean[0]) - 1
        (crossover, _) = fitting_erf(zpos, psi_mean, erf_sign)
        if frame_num == 0:
            np.savetxt('crossover.txt', [crossover])

    # Interface heights for each interface
    psi_grid = psi_grid.reshape(nx_grid, nz_grid, 2)
    height = np.zeros((nx_grid, 2))
    for ix in range(nx_grid):

        # Indices where psi is greater than the cross over value
        ind = np.where(psi[ix, :] > crossover)[0] # Assume liquid is in center
        if ind[0] == 0: # Solid is actually in center
            ind = np.where(psi[ix, :] < crossover)[0]

        # Height for first crossing (lower interface)
        ind_crossing = range(ind[0]-1, ind[0]+1)
        fit = np.polyfit(psi[ix, ind_crossing], psi_grid[ix, ind_crossing, 1], 1)
        height[ix, 0] = fit[0]*crossover + fit[1]

        # Height for second crossing (upper interface)
        ind_crossing = range(ind[-1], ind[-1]+2)
        fit = np.polyfit(psi[ix, ind_crossing], psi_grid[ix, ind_crossing, 1], 1)
        height[ix, 1] = fit[0]*crossover + fit[1]

    if reduce_flag:

        hmin = np.min(height, axis=0)
        hmax = np.max(height, axis=0)
        try:
            interface_positions_1D.hrng_half = max(interface_positions_1D.hrng_half,
                                                   1.6*np.max(hmax - hmin)/2.0 + smoothing_cutoff + 2.0*latparam)
        except AttributeError:
            interface_positions_1D.hrng_half = 1.6*np.max(hmax - hmin)/2.0 + smoothing_cutoff + 2.0*latparam

        hmean = np.mean(height, axis=0)

        interface_positions_1D.interface_range = \
            np.column_stack((hmean - interface_positions_1D.hrng_half,
                             hmean + interface_positions_1D.hrng_half))

    return height


def a_squared(dimension, height):

    if dimension == 1:

        # Subtract mean from heights
        height -= np.mean(height, axis=0)

        # Compute A^2 for each interface
        Asq = np.abs(fft.fft(height, axis=0)/height.shape[0])**2

    elif dimension == 2:

        # Subtract mean from heights
        height -= np.mean(np.mean(height, axis=0), axis=0)

        # Compute A^2 for each interface
        Asq = np.abs(fft.fft2(height, axes=[0, 1])/np.product(height.shape[:2]))**2.0

    return Asq
