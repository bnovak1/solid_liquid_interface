# added by pasteurize
########################################################################################
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

from joblib import Parallel, delayed, cpu_count
import mdtraj
import numpy as np
import pandas as pd
import scipy.constants as constants
import json, subprocess, sys
import solid_liquid_interface as sli


def get_boundary_atoms(coordsz, latparam):

    zmin = np.min(coordsz)
    boundary_atoms_lower = np.where(coordsz < zmin + latparam/4)[0]

    zmax = np.max(coordsz)
    boundary_atoms_upper = np.where(coordsz > zmax - latparam/4)[0]

    return (boundary_atoms_lower, boundary_atoms_upper)


def analyze_frame(nframes, frame, traj_file, topfile, n_neighbors, latparam, vectors_ref,
                  tree_ref, smoothing_cutoff, crossover, interface_options, outfile_prefix,
                  psi_avg_flag, reduce_flag=True):

    print(traj_file)

    # Read trajectory frame
    snapshot = mdtraj.load_lammpstrj(traj_file, top=topfile)
    snapshot.xyz *= snapshot.unitcell_lengths
    snapshot.xyz *= 10.0

    # Apply periodic boundary conditions to insure all coordinates are in [0, box_size].
    # This is required for the k-d tree algorithm with periodic boundary conditions.
    # k-d tree cannot have coordinates exactly at upper boundary, shift to lower boundary
    snapshot.xyz -= snapshot.unitcell_lengths*np.floor(snapshot.xyz/snapshot.unitcell_lengths)
    snapshot.xyz -= snapshot.unitcell_lengths*(snapshot.xyz == snapshot.unitcell_lengths)

    box_sizes = 10.0*snapshot.unitcell_lengths
    coords = 10.0*snapshot.xyz
    coords = coords[0, :, :]

    # Calculate system depth if there are free boundaries
    if interface_options['free_boundaries']:
        system_depth = np.mean(coords[interface_options['boundary_atoms_upper'], 2]) - \
                       np.mean(coords[interface_options['boundary_atoms_lower'], 2])

    # Calculate interface positions
    # Must stay away from free boundaries. This limits how little solid can be used to start.
    if interface_options['free_boundaries']:
        zmin = np.min(coords[:, 2]) + 2.0*latparam + smoothing_cutoff
        zmax = np.max(coords[:, 2]) - 2.0*latparam - smoothing_cutoff
        (X, Y, Z) = np.mgrid[0:box_sizes[0][0]:latparam/2,
                             0:box_sizes[0][1]:latparam/2,
                             zmin:zmax:latparam/2]
    else:
        (X, Y, Z) = np.mgrid[0:box_sizes[0][0]:latparam/2,
                             0:box_sizes[0][1]:latparam/2,
                             0:box_sizes[0][2]:latparam/2]


    if psi_avg_flag:
        psi_avg = sli.interface_positions_2D(coords, box_sizes, snapshot, n_neighbors,
                                    latparam, vectors_ref, tree_ref, X, Y, Z,
                                    smoothing_cutoff, crossover, interface_options,
                                    outfile_prefix, psi_avg_flag, reduce_flag)
        return psi_avg
    else:
        height = sli.interface_positions_2D(coords, box_sizes, snapshot, n_neighbors,
                                    latparam, vectors_ref, tree_ref, X, Y, Z,
                                    smoothing_cutoff, crossover, interface_options,
                                    outfile_prefix, psi_avg_flag, reduce_flag)

    height_avg = np.mean(np.mean(height, axis=0), axis=0)

    # Find interface atoms
    if interface_options['interface_flag']:

        x = np.hstack((np.unique(X), box_sizes[0][0]))
        y = np.hstack((np.unique(Y), box_sizes[0][1]))

        height_ext = np.empty((len(x), len(y), 2))
        for iint in range(2):
            h = np.column_stack((height[:, :, iint], height[:, 0, iint]))
            h = np.row_stack((h, h[0, :]))
            height_ext[:, :, iint] = h

        if interface_options['algorithm'] == 'nearest':

            interfaces = \
                sli.find_interfacial_atoms_2D_nearest(x, y, height_ext, coords, traj_file,
                                                      snapshot, interface_options, latparam)
        elif interface_options['algorithm'] == 'ITIM':
            interfaces = \
                sli.find_interfacial_atoms_2D_itim(x, y, height_ext, coords, traj_file,
                                                   snapshot, interface_options)

    # Save pdb file
    if interface_options['traj_flag']:
            sli.save_pdb(traj_file, coords, snapshot, interface_options, interfaces)

    # Interface concentrations
    if interface_options['conc_flag']:
        concs = sli.interface_concentrations(snapshot, interfaces, interface_options)
        if interface_options['free_boundaries']:
            return [height_avg, concs, system_depth]
        else:
            return [height_avg, concs]
    else:
        if interface_options['free_boundaries']:
            return height_avg
        else:
            return [height_avg, system_depth]


if __name__ == "__main__":

    # Name of input file from command line
    infile = sys.argv[1]

    with open(infile) as f:
        inputs = json.load(f)

    interface_options = inputs['interface_options']
    psi_avg_flag = inputs['psi_avg_flag']
    latparam = inputs['latparam']

    interface_options['interface_flag'] = \
        interface_options['traj_flag'] or interface_options['conc_flag']

    if interface_options['interface_flag']:
        assert interface_options['algorithm'] == 'ITIM' or \
               interface_options['algorithm'] == 'nearest', \
               'Algorithm for finding interfacial atoms must be "ITIM" or "nearest".'

    # Make directories for interface trajectories
    if interface_options['traj_flag']:
        interface_options['traj_dir'] = inputs['traj_dir'] + 'interfaces/'
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

    # Read 1st frame to get atom names, free boundary atoms
    snapshot = mdtraj.load_lammpstrj(traj_files[0], top=traj_top_file)

    if interface_options['conc_flag']:
        (table, _) = snapshot.top.to_dataframe()
        interface_options['atom_names'] = table['name']
        interface_options['atom_name_list'] = np.unique(interface_options['atom_names'])
        n_names = len(interface_options['atom_name_list'])

    if interface_options['free_boundaries']:

        coords = 100.0*snapshot.xyz
        coordsz = coords[0, :, 2]
        del coords

        (interface_options['boundary_atoms_lower'],
         interface_options['boundary_atoms_upper']) = get_boundary_atoms(coordsz, latparam)
        del coordsz

    del snapshot

    # First frame
    output1 = analyze_frame(nframes, 0, traj_files[0], traj_top_file,
                           inputs['n_neighbors'],  latparam, vectors_ref,
                           tree_ref, inputs['smoothing_cutoff'], inputs['crossover'],
                           interface_options, inputs['outfile_prefix'], psi_avg_flag, False)

    # Rest of frames
    output2 = Parallel(n_jobs=inputs['nthreads']) \
              (delayed(analyze_frame) \
               (nframes, frame, traj_files[frame], traj_top_file,
                inputs['n_neighbors'], latparam, vectors_ref, tree_ref,
                inputs['smoothing_cutoff'], inputs['crossover'],
                interface_options, inputs['outfile_prefix'], psi_avg_flag, False) \
               for frame in range(1, nframes))

    if psi_avg_flag:

        # Combine first and subsequent frame data, average psi over frames, save
        psi_avg = np.mean(np.hstack((output1, output2)))
        np.savetxt(inputs['outfile_prefix'] + '.dat', [psi_avg],
                   header='Average value of psi')

    else:

        n_layers = interface_options['n_layers']

        # Combine first and subsequent frame data
        if interface_options['conc_flag']:

            height = np.vstack((output1[0],
                                np.array([output2[i][0] for i in range(nframes-1)])))

            conc = np.vstack((output1[1],
                                  np.array([output2[i][1] for i in range(nframes-1)])))

            # Save concentrations to file
            cols = ['frame']
            nbins = (n_layers-1)*interface_options['nbins_per_layer'] + 1
            for ibin in range(nbins):
                #for phase in ['E', 'C']:
                for interface in ['L', 'U']:
                    for iname in range(n_names-1):
                        code = interface + str(ibin) + \
                                   interface_options['atom_name_list'][iname]
                        cols.append(code)

            with open(inputs['outfile_prefix'] + '_concs.dat', 'w') as f:
                f.write('Codes: 1st is interface (L=lower, U=upper), '+ \
                        #'2nd is layer (0=at interface, 1=one layer from interface, ...), ' + \
                        '2nd is bin (edge to center, middle is around interface boundary), ' + \
                        #'3rd is phase (E=phase at edge of box, C=phase in center of box), ' + \
                        'Last is the atom name.\n')
            outdata = pd.DataFrame(columns=cols)
            outdata[cols[0]] = range(nframes)
            outdata[cols[1:]] = conc

            outdata.to_csv(inputs['outfile_prefix'] + '_concs.dat', mode='a', index=False, sep=str(' '))

            if interface_options['free_boundaries']:
                system_depth = np.hstack((output1[2], np.array([output2[i][2] for i in range(nframes-1)])))

        else:

            if interface_options['free_boundaries']:

                height = np.vstack((output1[0],
                                    np.array([output2[i][0] for i in range(nframes-1)])))

                system_depth = np.vstack((output1[1],
                                          np.array([output2[i][1] for i in range(nframes-1)])))

            else:

                height = np.vstack((output1, np.array(output2)))

        del output1, output2


        # Save interface positions to file
        outdata = np.column_stack((range(nframes), height))
        np.savetxt(inputs['outfile_prefix'] + '_pos.dat', outdata,
                   header='Frame | Interface postions (angstroms) for lower and upper interfaces')

        # Save system depth to file
        outdata = np.column_stack((range(nframes), system_depth))
        np.savetxt(inputs['outfile_prefix'] + '_sys_depth.dat', outdata,
                   header='Frame | System depth (angstroms)')
