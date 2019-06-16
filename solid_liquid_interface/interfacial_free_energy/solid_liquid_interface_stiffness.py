'''
# quasi-1D interface, no concentration, 4 threads
>>> import numpy as np
>>> import solid_liquid_interface_stiffness as slis
>>> slis.main('tests/traj_analysis_1D.json')
>>> data = np.loadtxt('test_data.dat')
>>> data_test = np.loadtxt('tests/Ti-results/1D/test_data.dat')
>>> np.allclose(data, data_test)
True
>>> phi = np.loadtxt('test_phi.dat')
>>> phi_test = np.loadtxt('tests/Ti-results/1D/test_phi.dat')
>>> np.allclose(phi, phi_test)
True
>>> psi = np.loadtxt('test_psi.dat')
>>> psi_test = np.loadtxt('tests/Ti-results/1D/test_psi.dat')
>>> np.allclose(psi, psi_test)
True

# 2D interface

'''

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

import mdtraj
from joblib import Parallel, delayed, cpu_count
import numpy as np
import scipy.constants as constants
import json, subprocess, sys
import solid_liquid_interface as sli


def analyze_frame(dimension, traj_file, topfile, n_neighbors, latparam, vectors_ref, tree_ref,
                 smoothing_cutoff, crossover, interface_options, outfile_prefix, psi_avg_flag,
                 save_flag=False):

    # Read trajectory frame
    snapshot = mdtraj.load_lammpstrj(traj_file, top=topfile)
    snapshot.xyz *= snapshot.unitcell_lengths
    box_sizes = 10.0*snapshot.unitcell_lengths
    coords = 100.0*snapshot.xyz
    coords = coords[0, :, :]
    snapshot.xyz *= 10.0

    if dimension == 1:

        # List of points in the x-z plane to compute psi on
        (X, Z) = np.mgrid[0:box_sizes[0, 0]:latparam/2, 0:box_sizes[0, 2]:latparam/2]

        # Calculate interface positions
        height = sli.interface_positions_1D(coords, box_sizes, snapshot, n_neighbors,
                                        latparam, vectors_ref, tree_ref, X, Z,
                                        smoothing_cutoff, crossover, interface_options,
                                        outfile_prefix, psi_avg_flag)

        # Find interface atoms
        if interface_options['interface_flag']:

            x = np.hstack((np.unique(X), box_sizes[0, 0]))
            interfaces = sli.find_interfacial_atoms_1D(x, np.row_stack((height, height[0, :])),
                                                   coords, traj_file, snapshot,
                                                   interface_options)

    elif dimension == 2:

        (X, Y, Z) = np.mgrid[0:box_sizes[0][0]:latparam/2,
                             0:box_sizes[0][1]:latparam/2,
                             0:box_sizes[0][2]:latparam/2]

        # Calculate interface positions
        height = sli.interface_positions_2D(coords, box_sizes, snapshot, n_neighbors,
                                        latparam, vectors_ref, tree_ref, X, Y, Z,
                                        smoothing_cutoff, crossover, interface_options,
                                        outfile_prefix, psi_avg_flag)

        # Find interface atoms
        if interface_options['interface_flag']:

            x = np.hstack((np.unique(X), box_sizes[0][0]))
            y = np.hstack((np.unique(Y), box_sizes[0][1]))

            height_ext = np.empty((len(x), len(y), 2))
            for iint in range(2):
                h = np.column_stack((height[:, :, iint], height[:, 0, iint]))
                h = np.row_stack((h, h[0, :]))
                height_ext[:, :, iint] = h

            interfaces = sli.find_interfacial_atoms_2D(x, y, height_ext, coords, traj_file,
                                                   snapshot, interface_options)

    # Save pdb file
    if interface_options['traj_flag']:
        sli.save_pdb(traj_file, coords, snapshot, interface_options, interfaces)

    # Interface concentrations
    if interface_options['conc_flag']:
        concs = sli.interface_concentrations(snapshot, interfaces, interface_options)

    # FFT to get A^2
    A_sq = sli.a_squared(dimension, height)

    if interface_options['conc_flag']:
        return [A_sq, concs]
    else:
        return A_sq

def main(infile):

    with open(infile) as f:
        inputs = json.load(f)

    dimension = inputs['dimension']
    interface_options = inputs['interface_options']
    psi_avg_flag = inputs['psi_avg_flag']

    interface_options['interface_flag'] = \
        interface_options['traj_flag'] or interface_options['conc_flag']

    # Make directories for interface trajectories
    if interface_options['traj_flag']:
        interface_options['traj_dir'] = inputs['traj_dir'] + '_interfaces/'
        subprocess.call('mkdir -p ' + interface_options['traj_dir'], shell=True)

    # File names
    reference_lammps = inputs['traj_dir'] + '/' + inputs['reference_prefix'] + '.lammpstrj'
    reference_top = inputs['traj_dir'] + '/' + inputs['reference_prefix'] + '.pdb'
    traj_top_file = inputs['traj_dir'] + '/' + inputs['traj_top_file']

    # Compute reference vectors from reference crystal structure
    (vectors_ref, tree_ref) = sli.get_ref_vectors(inputs['n_neighbors'], reference_lammps,
                                              reference_top)

    # Get list of files with trajectory frames
    shell_str = "ls -lv " + inputs['traj_dir'] + "/" + inputs['traj_pattern'] + " | awk '{print $9}'"
    traj_files = subprocess.check_output(shell_str, shell=True)
    traj_files = traj_files.decode().split('\n')[:-1]
    nframes = len(traj_files)

    # Read 1 frame to get box sizes in x and y, atom names
    snapshot = mdtraj.load_lammpstrj(traj_files[0], top=traj_top_file)
    box_sizes = 10.0*snapshot.unitcell_lengths

    if interface_options['conc_flag']:
        (table, _) = snapshot.top.to_dataframe()
        interface_options['atom_names'] = table['name']
        interface_options['atom_name_list'] = np.unique(interface_options['atom_names'])
        n_names = len(interface_options['atom_name_list'])

    del snapshot

    # Computation of A^2 or average psi for first frame is not done in parallel,
    # since this frame is used to determine an upper bound on the maximum number of
    # neighbors in the smoothing cylinders perpendicular to the y direction.
    # This bound is taken to be 1.05 times the maximum number of neighbors in those
    # cylinders for the first frame. Computation of A^2 or average psi for the remaining
    # frames is done in parallel.
    output1 = analyze_frame(dimension, traj_files[0], traj_top_file,
                           inputs['n_neighbors'],  inputs['latparam'], vectors_ref,
                           tree_ref, inputs['smoothing_cutoff'], inputs['crossover'],
                           interface_options, inputs['outfile_prefix'], psi_avg_flag,
                           save_flag=True)

    output2 = Parallel(n_jobs=inputs['nthreads']) \
              (delayed(analyze_frame) \
               (dimension, traj_files[frame], traj_top_file,
                inputs['n_neighbors'], inputs['latparam'], vectors_ref, tree_ref,
                inputs['smoothing_cutoff'], inputs['crossover'],
                interface_options, inputs['outfile_prefix'], psi_avg_flag) \
               for frame in range(1, nframes))

    if psi_avg_flag:

        # Combine first and subsequent frame data, average psi over frames, save
        psi_avg = np.mean(np.hstack((output1, output2)))
        np.savetxt(inputs['outfile_prefix'] + '.dat', [psi_avg],
                   header='Average value of psi')

    else:

        # Combine first and subsequent frame data
        # Mean value of A^2 for each interface
        # k^2. k = 2*pi*n/L, units of A^2 or m^-2/10^20
        if inputs['dimension'] == 1:

            if interface_options['conc_flag']:

                nx_grid = output1[0].shape[0]
                Asq = np.vstack((output1[0].reshape(1, nx_grid, 2),
                                 np.array([output2[i][0] for i in range(nframes-1)])))
                conc = np.vstack((output1[1],
                                  np.array([output2[i][1] for i in range(nframes-1)])))

            else:

                nx_grid = output1.shape[0]
                Asq = np.vstack((output1.reshape(1, nx_grid, 2), output2))

            del output1, output2

            Asq_mean = np.mean(Asq, axis=0)

            ksq = (2.0*np.pi*np.arange(nx_grid)/box_sizes[0][0])**2

        elif inputs['dimension'] == 2:

            if interface_options['conc_flag']:

                nx_grid = output1[0].shape[0]
                ny_grid = output1[0].shape[1]
                sz = tuple(np.hstack((nframes, output1[0].shape)))
                Asq = np.empty(sz)
                Asq[0, :, :, :] = output1[0]
                Asq[1:, :, :, :] = np.array([output2[i][0] for i in range(nframes-1)])
                conc = np.vstack((output1[1],
                                  np.array([output2[i][1] for i in range(nframes-1)])))

            else:

                nx_grid = output1.shape[0]
                ny_grid = output1.shape[1]
                sz = tuple(np.hstack((nframes, output1.shape)))
                Asq = np.empty(sz)
                Asq[0, :, :, :] = output1
                Asq[1:, :, :, :] = np.array(output2)

            del output1, output2

            Asq_mean = np.mean(Asq, axis=0)

            kxsq = (2.0*np.pi*np.arange(nx_grid)/box_sizes[0][0])**2
            kysq = (2.0*np.pi*np.arange(ny_grid)/box_sizes[0][1])**2

        # Interface area
        area = np.product(box_sizes[0][:2])

        kT = constants.Boltzmann*inputs['temp'] # J

        const = 1000.0*kT/area
        Asq_inv = 1.0e20*const/Asq_mean # mJ/m^4/10^20 or mJ/m^2/A^2 (so that slope is in mJ/m^2)

        # Save data to be used for stiffness calculation, ksq & Asq_inv
        if inputs['dimension'] == 1:

            ind = np.where(ksq > 0)[0]
            outdata = np.column_stack((ksq[ind], Asq_inv[ind, :]))
            np.savetxt(inputs['outfile_prefix'] + '_data.dat', outdata,
                       header='ksq (A^-2) | kT/(area*Asq) (mJ/m^2/A^2)')

        elif inputs['dimension'] == 2:

            (kxsq_grid, kysq_grid) = np.meshgrid(kxsq, kysq)
            outdata = np.column_stack((kxsq_grid.flatten(), kysq_grid.flatten(),
                                       Asq_inv[:, :, 0].flatten(),
                                       Asq_inv[:, :, 1].flatten()))
            np.savetxt(inputs['outfile_prefix'] + '_data.dat', outdata[1:, :],
                       header='kxsq (A^-2) | kysq (A^-2) | kT/(area*Asq) (mJ/m^2/A^2)')

        # Save concentrations to file
        if interface_options['conc_flag']:

            cols = []
            for iint in range(2):
                for ilayer in range(interface_options['n_layers']):
                    for iphase in range(2):
                        for iname in range(n_names):
                            code = str(iint) + str(ilayer) + str(iphase) + \
                                   interface_options['atom_name_list'][iname]
                            cols.append(code)

            np.savetxt(inputs['outfile_prefix'] + '_concs.dat', conc,
                       header='Codes: 1st num is for interface (0=lower, 1=upper), ' + \
                              '2nd num is for layer (0=at interface, 1=one layer from interface, ...), ' + \
                              '3rd num is for phase (0=phase at edge of box, 1=phase in center of box), ' + \
                              'Last is the atom type.\n' + ' | '.join(cols))
            # df = pd.DataFrame(conc, columns=cols)
            # df.to_csv(inputs['outfile_prefix'] + '_concs.dat', index=False)

if __name__ == "__main__":

    # Name of input file from command line
    infile = sys.argv[1]

    main(infile)
