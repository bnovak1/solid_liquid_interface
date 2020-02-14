"""
Compute phi and psi order parameters based on local structure relative to a reference crystal
structure. Find interfacial atoms. Compute concentration profiles across the interface.
Compute interface widths. 2D and quasi-1D inteface geometries. Currently works only with cubic
crystal structures.
"""

import lmfit
import mdtraj
import numpy as np
import scipy.interpolate as interp
import scipy.spatial as ss
import scipy.special as spec


def get_ref_vectors(n_neighbors, reference_structure, topfile):
    """
    Description
    ----
    Compute the vectors from a single atom to its n_neighbors nearest neighbors in the reference
    structure. Construct a k-d tree for those vectors.

    Inputs
    ----
    :n_neighbors: Number of nearest neighbors to consider.
    :reference_structure: Name of file with reference structure.
    :top_file: Name of file with topology information such as a pdb file.

    Outputs
    ----
    :vectors_ref: Vectors from a single atom to its nearest neighbors in the reference structure.
    :tree_ref: k-d tree of vectors_ref.
    """

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


def visualize_bins(box_sizes, latparam, interface_options, height, outfile_prefix):
    """
    Description
    ----
    Create 3D visualization of surfaces defining bin edges used for concentration profiles.
    Alternate colors for overlapping bins. For example, if there are 3 bins which are overlapping,
    then the colors would be blue, green, red, blue, green, red, ...

    Inputs
    ----
    :box_sizes: Simulation box sizes.
    :latparam: Crystal lattice parameter.
    :interface_options: Options dictionary related to the interface read from the input file.
                        Here the layer_width, n_bins_per_layer, and n_layers keys are used.
    :height: 3D array defining the z positions as a function of x and y for the 2 interfaces.
    :outfile_prefix: Prefix for file name of output png file.
    """

    from mpl_toolkits import mplot3d
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    import my_plot_settings_article as mpsa

    mpl.rcParams['font.size'] = 8

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
    ax.plot_wireframe(X, Y, height[:, :, 0]+ (ilayer+1)*layer_width, linewidth=0.3, color='0.5')

    xticks = ax.get_xticks()
    xtick_spacing = 2*(xticks[1] - xticks[0])
    yticks = ax.get_yticks()
    ytick_spacing = 2*(yticks[1] - yticks[0])
    zticks = ax.get_zticks()
    ztick_spacing = 2*(zticks[1] - zticks[0])

    ml = MultipleLocator(xtick_spacing)
    ax.xaxis.set_major_locator(ml)

    ml = MultipleLocator(ytick_spacing)
    ax.yaxis.set_major_locator(ml)

    ml = MultipleLocator(ztick_spacing)
    ax.zaxis.set_major_locator(ml)

    ax.set_xlabel('x ($\\mathrm{\\AA}$)', labelpad=mpsa.axeslabelpad)#, fontsize=7)
    ax.set_ylabel('y ($\\mathrm{\\AA}$)', labelpad=mpsa.axeslabelpad)#, fontsize=7)
    ax.set_zlabel('z ($\\mathrm{\\AA}$)', labelpad=mpsa.axeslabelpad)#, fontsize=7)
    ax.dist = 14

    mpsa.save_figure(outfile_prefix + '_bins.png', 300)
    plt.close()


def save_pdb(traj_file, coords, snapshot, interface_options, interfaces):
    """
    Description
    ----
    Save a pdb file with atoms near interfaces colored based on the bin they are in.
    Only makes sense with non-overlapping bins.

    Inputs
    ----
    :traj_file: Name of trajector file. Used to determine the name of the output file.
    :coords: Atom coordinates array.
    :interface_options: Options dictionary related to the interface read from the input file.
                        Here the traj_dir key is used.
    :interfaces: Indices of atoms in the bins defined around each interface.
    """

    natoms = coords.shape[0]
    nbins = interfaces[0].shape[0]
    bfactors = np.zeros(natoms)

    for iint in range(2):

        beta = 0.0

        for ibin in range(nbins):

            beta += 1.0

            ind = interfaces[iint, ibin]
            bfactors[ind] = beta


    traj_prefix = '.'.join(traj_file.split('.')[:-1])
    traj_prefix = traj_prefix.split('/')[-1]

    pdb = mdtraj.formats.PDBTrajectoryFile( \
        interface_options['traj_dir'] + traj_prefix + '.pdb', mode='w')
    pdb.write(coords, snapshot.top, unitcell_lengths=tuple(10.0*snapshot.unitcell_lengths[0]),
              unitcell_angles=tuple(snapshot.unitcell_angles[0]), bfactors=bfactors)
    pdb.close()


def interface_concentrations(snapshot, interfaces, interface_options):
    """
    Description
    ----
    Concentrations in each bin near the interfaces.

    Inputs
    ----
    :interfaces: Indices of atoms in the bins defined around each interface.
    :interface_options: Options dictionary related to the interface read from the input file.
                        Here the atom_names and atom_name_list keys are used.

    Outputs
    ----
    :concs: Concentrations.
    """

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


def find_interfacial_atoms_2D(x, y, height, coords, traj_file, snapshot, interface_options,
                              latparam):
    """
    Description
    ----
    Find the atoms in the bins near the interfaces for a 2D interface.

    Inputs
    ----
    :x: Positions of grid points in interface plane.
    :y: Positions of grid points in interface plane.
    :height: Interface locations in the interface normal direction.
    :coords: Atom coordinates array.
    :interface_options: Options dictionary related to the interface read from the input file.
                        Here the n_layers, n_bins_per_layer, and layer_width keys are used.

    Outputs
    ----
    :interfaces: Indices of atoms in the bins defined around each interface.
    """

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

            ind = np.intersect1d(np.where(coords[atoms, 2] >= interface_projections + \
                                          lower_shift)[0],
                                 np.where(coords[atoms, 2] < interface_projections + \
                                          upper_shift)[0])

            interfaces[iint, ibin] = atoms[ind]

    interfaces[1, :] = interfaces[1, ::-1]

    return interfaces


# def find_interfacial_atoms_1D(x, height, coords, traj_file, snapshot, interface_options):
#     '''
#     FIX: STILL USES PHASES
#     '''
#
#     # Split phases
#     phase_atoms = np.empty(2, dtype=object)
#
#     z_interp_lower = interp.UnivariateSpline(x, height[:, 0], s=0)
#     interface_positions = z_interp_lower(coords[:, 0])
#     phase_atoms[0] = np.where(coords[:, 2] < interface_positions)[0]
#
#     z_interp_upper = interp.UnivariateSpline(x, height[:, 1], s=0)
#     interface_positions = z_interp_upper(coords[:, 0])
#     phase_atoms[0] = np.hstack((phase_atoms[0],
#                                 np.where(coords[:, 2] > interface_positions)[0]))
#
#     phase_atoms[1] = np.setxor1d(phase_atoms[0], range(coords.shape[0]))
#
#     # Interfacial atoms
#     interfaces = itim(snapshot, coords, phase_atoms, interface_options)
#
#     return interfaces


def erf_one_interface(order_param_sol, order_param_liq, pos, pos_interface, sigma, erf_sign):
    """
    Description
    ----
    Evaluate an error function for the order parameter as a function of interface normal position
    (pos) for a single interface.

    Inputs
    ----
    :order_param_sol: Parameter for average order parameter in solid phase.
    :order_param_sol: Parameter for average order parameter in liquid phase.
    :pos: Array of position in the interface normal direction.
    :pos_interface: Parameter for average interface position in the interface normal direction.
    :sigma: Width parameter.
    :erf_sign: Can be set to 1 or -1 depending on the relative locations of the solid and liquid
               in the simulation box.

    Outputs
    ----
    :order_param_fit: Array of values of the fitted function corresponding to pos.
    """

    order_param_fit = \
        0.5*((order_param_sol + order_param_liq) + erf_sign*(order_param_sol - order_param_liq)* \
             spec.erf((pos - pos_interface)/(sigma*np.sqrt(2.0))))

    return order_param_fit


def residual_erf_one_interface(params, pos, order_param, wghts, erf_sign):
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
    sigma = params['sigma'].value
    pos_interface = params['pos_interface'].value

    model = erf_one_interface(order_param_sol, order_param_liq, pos, pos_interface, sigma, erf_sign)

    residuals = (order_param - model)*wghts

    return residuals


def plot_interface_width(pos, order_param, fit, erf_sign, interface_width):
    """
    Function to make a plot of interface width. Not used normally.
    """

    import matplotlib.pyplot as plt
    import my_plot_settings_article as mpsa

    plt.plot(-pos, order_param, '.', label='Data', markersize=1, alpha=1, mec='none')
    plt.plot(-np.sort(pos),
             erf_one_interface(fit.params['order_param_sol'].value,
                               fit.params['order_param_liq'].value, np.sort(pos),
                               fit.params['pos_interface'].value, fit.params['sigma'].value,
                               erf_sign),
             label='Fit')

    dy = 0.05*np.diff(plt.ylim())
    ydata = [plt.ylim()[0] + dy, plt.ylim()[1] - dy]
    plt.plot([interface_width/2, interface_width/2], ydata, 'k--')
    plt.plot([-interface_width/2, -interface_width/2], ydata, 'k--')

    mpsa.axis_setup('x')
    mpsa.axis_setup('y')

    plt.xlabel('$\\mathrm{Z - Z_{interface} \\left( \\AA \\right)}$',
               labelpad=mpsa.axeslabelpad)
    plt.ylabel('$\\mathrm{\\psi}$', labelpad=mpsa.axeslabelpad)
    plt.legend()

    mpsa.save_figure('garbage.png', res=300)
    plt.close()


def fitting_erf_one_interface(pos, order_param, erf_sign, order_params_ini=[]):
    """
    Description
    ----
    Fit an error function for the order parameter as a function of interface normal position
    (pos) for a single interface with the purpose of defining an interface width.

    Inputs
    ----
    :pos: Array of position in the interface normal direction.
    :order_param: Array of order parameters values corresponding to pos.
    :erf_sign: Can be set to 1 or -1 depending on the relative locations of the solid and liquid
               in the simulation box.
    :order_params_ini: Optional. List of initial guesses for the average order parameter values in
                       the solid and liquid. Can be specified in any order. The order parameter
                       for the solid is assumed to be larger than in the liquid. If not specified,
                       the max and min values of order_param are used as intial guesses in the solid
                       and liquid, respectively.

    Outputs
    ----
    :interface_width: Width of the interface defined as 2*sqrt(2)*erfinv(0.99)*sigma where sigma is
                      the error function width parameter and erfinv is the inverse error function.
    """

    wghts = np.ones(len(pos))  # All weights equal
    params = lmfit.Parameters()

    pos_min = np.min(pos)
    delta_pos = np.max(pos) - pos_min

    if not order_params_ini:
        order_param_sol_ini = np.max(order_param)
        order_param_liq_ini = np.min(order_param)
        params.add('order_param_sol', value=order_param_sol_ini, min=0.0)
        params.add('order_param_liq', value=order_param_liq_ini, min=0.0)
    else:
        order_param_sol_ini = np.max(order_params_ini)
        order_param_liq_ini = np.min(order_params_ini)
        params.add('order_param_sol', value=order_param_sol_ini, vary=False)
        params.add('order_param_liq', value=order_param_liq_ini, vary=False)

    fliq = (np.mean(order_param) - order_param_sol_ini)/ \
           (order_param_liq_ini - order_param_sol_ini)
    fsol = 1 - fliq

    pos_interface = delta_pos*((fliq/2)*(erf_sign == 1) + (fsol/2)*(erf_sign == -1)) + pos_min
    params.add('pos_interface', value=pos_interface)

    params.add('sigma', value=5.0, min=0.0)

    fit = lmfit.minimize(residual_erf_one_interface, params,
                         args=(pos, order_param, wghts, erf_sign))

    # order_param_sol = fit.params['order_param_sol'].value
    # order_param_liq = fit.params['order_param_liq'].value
    # crossover = (order_param_sol + order_param_liq)/2
    multiplier = 2*np.sqrt(2)*spec.erfinv(0.99)
    interface_width = multiplier*fit.params['sigma'].value

    # plot_interface_width(pos, order_param, fit, erf_sign, interface_width)
    # import pdb; pdb.set_trace()

    return interface_width


def erf_two_interface(order_param_sol, order_param_liq, pos, pos_interface_lower,
                      pos_interface_upper, sigma_lower, sigma_upper, erf_sign):
    """
    Description
    ----
    Evaluate a set of error functions for the order parameter as a function of interface
    normal position (pos) for both interfaces. Switches from 1 error function to the other half
    way betweeen the interfaces.

    Inputs
    ----
    :order_param_sol: Parameter for average order parameter in solid phase. Same for upper & lower.
    :order_param_sol: Parameter for average order parameter in liquid phase. Same for upper & lower.
    :pos: Array of position in the interface normal direction.
    :pos_interface_lower: Parameter for average lower interface position in the
                          interface normal direction.
    :pos_interface_upper: Parameter for average upper interface position in the
                          interface normal direction.
    :sigma_lower: Width parameter for lower interface.
    :sigma_upper: Width parameter for upper interface.
    :erf_sign: Can be set to 1 or -1 depending on the relative locations of the solid and liquid
               in the simulation box.

    Outputs
    ----
    :order_param_fit: Array of values of the fitted function corresponding to pos.
    """

    pos_bound = (pos_interface_lower + pos_interface_upper)/2.0

    order_param_fit = \
        0.5*((order_param_sol + order_param_liq) + \
             erf_sign*(pos < pos_bound)*(order_param_sol - order_param_liq)* \
             spec.erf((pos - pos_interface_lower)/(sigma_lower*np.sqrt(2.0))) - \
             erf_sign*(pos > pos_bound)*(order_param_sol - order_param_liq)* \
             spec.erf((pos - pos_interface_upper)/(sigma_upper*np.sqrt(2.0))))

    return order_param_fit


def residual_erf_two_interface(params, pos, order_param, wghts, erf_sign):
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

    model = erf_two_interface(order_param_sol, order_param_liq, pos, pos_interface_lower,
                              pos_interface_upper, sigma_lower, sigma_upper, erf_sign)

    residuals = (order_param - model)*wghts

    return residuals


def fitting_erf_two_interface(pos, order_param, erf_sign, order_params_ini=[]):
    """
    Description
    ----
    Fit a set of error functions for the order parameter as a function of interface normal position
    (pos) for both interfaces.

    Inputs
    ----
    :pos: Array of position in the interface normal direction.
    :order_param: Array of order parameters values corresponding to pos.
    :erf_sign: Can be set to 1 or -1 depending on the relative locations of the solid and liquid
               in the simulation box.
    :order_params_ini: Optional. List of initial guesses for the average order parameter values in
                       the solid and liquid. Can be specified in any order. The order parameter
                       for the solid is assumed to be larger than in the liquid. If not specified,
                       the max and min values of order_param are used as intial guesses in the solid
                       and liquid, respectively.

    Outputs
    ----
    :crossover: Value of the order parameter half way between the average values in the
                solid and liquid phases.
    :interface_widths: Widths of each interface defined as 2*sqrt(2)*erfinv(0.99)*sigma where
                       sigma is the error function width parameter and erfinv is the
                       inverse error function.
    :[order_param_sol, order_param_liq]: A list containing the average values of the
                                         order parameter in the solid and liquid phases.
    """

    wghts = np.ones(len(pos))  # All weights equal
    params = lmfit.Parameters()

    pos_min = np.min(pos)
    delta_pos = np.max(pos) - pos_min

    if not order_params_ini:
        order_param_sol_ini = np.max(order_param)
        order_param_liq_ini = np.min(order_param)
        params.add('order_param_sol', value=order_param_sol_ini, min=0.0)
        params.add('order_param_liq', value=order_param_liq_ini, min=0.0)
    else:
        order_param_sol_ini = np.max(order_params_ini)
        order_param_liq_ini = np.min(order_params_ini)
        params.add('order_param_sol', value=order_param_sol_ini, vary=False)
        params.add('order_param_liq', value=order_param_liq_ini, vary=False)

    fliq = (np.mean(order_param) - order_param_sol_ini)/ \
           (order_param_liq_ini - order_param_sol_ini)
    fsol = 1 - fliq

    pos_interface_lower_ini = delta_pos*((fliq/2)*(erf_sign == 1) + \
                                         (fsol/2)*(erf_sign == -1)) + pos_min
    pos_interface_upper_ini = delta_pos*((1 - fliq/2)*(erf_sign == 1) + \
                                         (1 - fsol/2)*(erf_sign == -1)) + pos_min
    params.add('pos_interface_lower', value=pos_interface_lower_ini)
    params.add('pos_interface_upper', value=pos_interface_upper_ini)

    params.add('sigma_lower', value=5.0, min=0.0)
    params.add('sigma_upper', value=5.0, min=0.0)

    fit = lmfit.minimize(residual_erf_two_interface, params,
                         args=(pos, order_param, wghts, erf_sign))

    order_param_sol = fit.params['order_param_sol'].value
    order_param_liq = fit.params['order_param_liq'].value
    crossover = (order_param_sol + order_param_liq)/2
    multiplier = 2*np.sqrt(2)*spec.erfinv(0.99)
    interface_widths = multiplier*np.array([fit.params['sigma_upper'].value,
                                            fit.params['sigma_lower'].value])

    # import matplotlib.pyplot as plt
    # plt.plot(pos, order_param)
    # plt.plot(pos, erf_two_interface(order_param_sol, order_param_liq, pos,
    #                                 fit.params['pos_interface_lower'].value,
    #                                 fit.params['pos_interface_upper'].value,
    #                                 fit.params['sigma_lower'].value,
    #                                 fit.params['sigma_upper'].value, erf_sign))
    # plt.show()
    # import pdb; pdb.set_trace()

    return (crossover, interface_widths, [order_param_sol, order_param_liq])


def interface_positions_2D(frame_num, coords, box_sizes, snapshot, n_neighbors, latparam,
                           vectors_ref, tree_ref, X, Y, Z, smoothing_cutoff,
                           interface_options, outfile_prefix, crossover=None,
                           reduce_flag=True, interface_range=None):
    """
    Description
    ----
    Interface positions for a 2D interface.

    Inputs
    ----
    :frame_num: Frame number.
    :coords: Array of atom coordinates.
    :box_sizes: Array of simulation box sizes.
    :n_neighbors: Number of nearest neighbors to consider in order parameter calculation.
    :latparam: Lattice parameter.
    :tree_ref: k-d tree of vectors from a single atom to its nearest neighbors in the
               reference structure.
    :X: Grid positions.
    :Y: Grid positions.
    :Z: Grid positions.
    :smoothing_cutoff: Cutoff for smoothing phi order parameter to obtain psi order parameter.
    :outfile_prefix: Prefix for output files with phi data, psi data, and crossover.
    :crossover: Value of the order parameter half way between the average values in the
                solid and liquid phases. Default is None.
                If None, then it is computed using error function fits.
    :reduce_flag: Reduce the domain for calculating order parameters to only near the
                  regions near interfaces to save time? Currently only works with interfaces which
                  are stationary on average (interfacial free energy calculation). Default is True.
    :interface_range: Ranges defining where to compute order parameters.
                      Default is None since it is calculated and output to a file for the
                      first frame even if reduce_flag is True.
                      Then those values can be used in subsequent frames if reduce_flag is True.

    Outputs
    ----
    :height: Interface positions in the interface normal direction.
    :interface_widths: Non-intrinsic interface widths.
    :interface_widths_local: Intrinsic interface widths.
    """

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
    psi = (np.sum(wd*phi[ii], axis=1)/np.sum(wd, axis=1)).reshape(nx_grid, ny_grid,
                                                                  nz_grid)/(latparam**2)
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

    # Crossover value for psi and interface widths by fitting to error functions
    if (reduce_flag and frame_num == 0) or not reduce_flag:

        zpos = Z[0, 0, :]
        psi_mean = np.mean(np.mean(psi, axis=0), axis=0)
        erf_sign = 2*(psi_mean[int(len(psi_mean)/2)] > psi_mean[0]) - 1
        (crossover, interface_widths, order_param_limits) = \
            fitting_erf_two_interface(zpos, psi_mean, erf_sign)

        del zpos, psi_mean

        if frame_num == 0:
            np.savetxt(outfile_prefix + '_crossover.txt', [crossover])


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
            # If problem with crossover, interfaces are too close. Set all heights to -1, break.
            ind_crossing = range(ind[0]-1, ind[0]+1)
            try:
                fit = np.polyfit(psi[ix, iy, ind_crossing], psi_grid[ix, iy, ind_crossing, 2], 1)
                height[ix, iy, 0] = fit[0]*crossover + fit[1]
            except IndexError:
                height[:, :, :] = -1
                break

            # Height for second crossing (upper interface)
            ind_crossing = range(ind[-1], ind[-1]+2)
            try:
                fit = np.polyfit(psi[ix, iy, ind_crossing], psi_grid[ix, iy, ind_crossing, 2], 1)
                height[ix, iy, 1] = fit[0]*crossover + fit[1]
            except IndexError:
                height[:, :, :] = -1
                break

        if np.mean(height) == -1:
            break

    # Local interface widths by fitting to error functions
    if (reduce_flag and frame_num == 0) or not reduce_flag:

        interface_widths_local = np.zeros(2)

        if np.mean(height) != -1:

            mid = np.mean((height[:, 1] + height[:, 0])/2.0)

            # Lower interface
            ind = np.where(Z[0, 0, :] < mid)[0]
            zpos = np.nan*np.ones(Z[:, :, ind].shape)
            cnt = 0
            for i in ind:
                zpos[:, :, cnt] = Z[:, :, i] - height[:, :, 0]
                cnt += 1
            zpos = zpos.flatten()
            psi_flat = psi[:, :, ind].flatten()
            interface_widths_local[1] = fitting_erf_one_interface(zpos, psi_flat, erf_sign,
                                                                  order_param_limits)
            del zpos, psi_flat

            # Upper interface
            ind = np.where(Z[0, 0, :] > mid)[0]
            zpos = np.nan*np.ones(Z[:, :, ind].shape)
            cnt = 0
            for i in ind:
                zpos[:, :, cnt] = Z[:, :, i] - height[:, :, 1]
                cnt += 1
            zpos = -zpos.flatten()
            psi_flat = psi[:, :, ind].flatten()
            interface_widths_local[0] = fitting_erf_one_interface(zpos, psi_flat, erf_sign,
                                                                  order_param_limits)
            del zpos, psi_flat

    if frame_num == 0:
        interface_range_2D(height, smoothing_cutoff, latparam, outfile_prefix)

    try:
        return (height, interface_widths, interface_widths_local)
    except UnboundLocalError:
        return (height, -1, -1)


def interface_range_2D(height, smoothing_cutoff, latparam, outfile_prefix):
    """
    Description
    ----
    Estimated range of interface normal positions around the interface to
    calculate the order parameter in. Does not work for moving interfaces.
    This is used on the first frame if only the parts of the system near interfaces are
    considered for order parameter calculation.
    The values are written to a file and can be used for subsequent frames.

    Inputs
    ----
    """

    hmin = np.min(np.min(height, axis=0), axis=0)
    hmax = np.max(np.max(height, axis=0), axis=0)

    hrng_half = 5.0*np.max(hmax - hmin)/2.0 + smoothing_cutoff + 2.0*latparam
    hmean = np.mean(np.mean(height, axis=0), axis=0)
    interface_range = np.column_stack((hmean - hrng_half, hmean + hrng_half))

    np.savetxt(outfile_prefix + '_interface_range.txt', interface_range)


def interface_positions_1D(frame_num, coords, box_sizes, snapshot, n_neighbors, latparam,
                           vectors_ref, tree_ref, X, Z, smoothing_cutoff, interface_options,
                           outfile_prefix, crossover=None, reduce_flag=True):
    """
    Description
    ----
    Interface positions for a quasi-1D interface.

    Inputs
    ----
    :frame_num: Frame number.
    :coords: Array of atom coordinates.
    :box_sizes: Array of simulation box sizes.
    :snapshot: mdtraj trajectory object.
    :n_neighbors: Number of nearest neighbors to consider in order parameter calculation.
    :latparam: Lattice parameter.
    :tree_ref: k-d tree of vectors from a single atom to its nearest neighbors in the
               reference structure.
    :X: Grid positions.
    :Y: Grid positions.
    :Z: Grid positions.
    :smoothing_cutoff: Cutoff for smoothing phi order parameter to obtain psi order parameter.
    :outfile_prefix: Prefix for output files with phi data, psi data, and crossover.
    :crossover: Value of the order parameter half way between the average values in the
                solid and liquid phases. Default is None.
                If None, then it is computed using error function fits.
    :reduce_flag: Reduce the domain for calculating order parameters to only near the
                  regions near interfaces to save time? Currently only works with interfaces which
                  are stationary on average (interfacial free energy calculation). Default is True.

    Outputs
    ----
    :height: Interface positions in the interface normal direction.
    :interface_widths: Non-intrinsic interface widths.
    :interface_widths_local: Intrinsic interface widths.
    """

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
    if not interface_options['interface_flag']:
        del tree_xz

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

    # Crossover value for psi by fitting to error functions
    if (reduce_flag and frame_num == 0) or not reduce_flag:
        zpos = Z[0, :]
        psi_mean = np.mean(psi, axis=0)
        erf_sign = 2*(psi_mean[int(len(psi_mean)/2)] > psi_mean[0]) - 1
        (crossover, interface_widths, order_param_limits) = \
            fitting_erf_two_interface(zpos, psi_mean, erf_sign)

        nfits = psi.shape[0]
        interface_widths_local = np.array([fitting_erf_two_interface(zpos, psi[ifit, :], erf_sign,
                                                                     order_param_limits)[1] \
                                           for ifit in range(nfits)])
        interface_widths_local = np.mean(interface_widths_local, axis=0)

        del zpos, psi_mean, erf_sign

        if frame_num == 0:
            np.savetxt(outfile_prefix + '_crossover.txt', [crossover])

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
            interface_positions_1D.hrng_half = \
                max(interface_positions_1D.hrng_half,
                    1.6*np.max(hmax - hmin)/2.0 + smoothing_cutoff + 2.0*latparam)
        except AttributeError:
            interface_positions_1D.hrng_half = \
                1.6*np.max(hmax - hmin)/2.0 + smoothing_cutoff + 2.0*latparam

        hmean = np.mean(height, axis=0)

        interface_positions_1D.interface_range = \
            np.column_stack((hmean - interface_positions_1D.hrng_half,
                             hmean + interface_positions_1D.hrng_half))

    try:
        return (height, interface_widths, interface_widths_local)
    except UnboundLocalError:
        return (height, -1, -1)


def a_squared(dimension, height):
    """
    Description
    ----
    Fourier transform (FFT) of interface position - mean interface position in the
    interface normal direction.

    Inputs
    ----
    :dimension: Dimension of interfaces. 1 for quasi-1D and 2 for 2D.
    :height: Array of interface positions in the interface normal direction.

    Outputs
    ----
    :asq: Absolute value squared of FFT values, normalized.
    """

    if dimension == 1:

        # Subtract mean from heights
        height -= np.mean(height, axis=0)

        # Compute A^2 for each interface
        asq = np.abs(np.fft.rfft(height, axis=0)/height.shape[0])**2


    elif dimension == 2:

        # Subtract mean from heights
        height -= np.mean(np.mean(height, axis=0), axis=0)

        # Compute A^2 for each interface
        asq = np.abs(np.fft.fft2(height, axes=[0, 1])/np.product(height.shape[:2]))**2.0

    return asq
