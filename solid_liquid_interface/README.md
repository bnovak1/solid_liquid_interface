# Purpose

* Computation of solid-liquid interfacial properties from molecular dynamics simulation data.   
    * Interface velocities needed for kinetic coefficient calculation
    * Interface fluctuations needed for interfacial free energy calculation (quasi 1D or 2D interfaces)
    * Interface widths
    * Concentration profiles for multicomponent systems

# Limitations

* Input is a LAMMPS format trajectory split into 1 frame per file
* Orthogonal simulation box
* Only tested for fcc and bcc structures.

# Background

## Kinetic coefficient from free solidification simulations

FILL

## Interfacial free energy using capillary fluctuation method

FILL

# Usage

## JSON input file variables

Due to the large number of options, a JSON input file is used. Below is a description of the keys. Some are required, while if others are not included there is a default behavior.

* traj_dir: Name of directory containing trajectory frames.
* reference_prefix: Prefix for reference structure files. Currently, these are assumed to have extensions of .lammpstrj and .pdb and be in the traj_dir directory. Reference structures should be a perfect lattice with the same lattice parameter and in the same crystallographic orientation as the solid in your coexistence simulation. The .pdb file is needed by mdtraj and can be created from the .lammpstrj file.
* traj_pattern: Regular expression for trajectory file names not including the directory name which is assumed to be traj_dir.
* traj_top_file: Name of topology file for trajectory files not including the directory name which is assumed to be traj_dir. This is needed for mdtraj and can be created from one of the trajectory frames. Could use any type of topology file accepted by mdtraj such as pdb.
* temp: Temperature in Kelvin.
* latparam: Mean lattice parameter in angstroms, a, at melting temperature.
* n_neighbors: Number of nearest neighbors to consider for calculation of order parameter. 14 for bcc, 12 for fcc.
* smoothing_cutoff: Cutoff for smoothing when calculating psi (2.5a and )
* outfile_prefix: Prefix for output files
* dimension: 1 for quasi-1D interface, 2 for 2D interface
* interface_options: Options for interface determination and concentrations for alloys.
    * traj_flag: If true, write pdb files for each frame with different B factors for different interfacial layers. Should be true or false.
    * conc_flag: For alloys, set to true to compute concentrations in each interfacial layer. Should be true or false.
    * free_boundaries: Specify as true if doing solidification simulations with free boundaries in the interface normal direction, otherwise false.
    * r_atoms: Dictionary of radii for each atom type in the system with atom names as the keys. A reasonable choice for this is the radius calculated from a perfect crystal which is about 0.35355a for fcc and 0.433a for bcc. For dilute alloys, using half the distances to the first peaks in g(r<sub>solvent-solvent</sub>) and g(r<sub>solute-solvent</sub>). Only needs to be included if traj_flag or conc_flag is true.
    * layer_width: Bin width for concentration profiles. A good choice would be the diameter of the largest atom in the system. Only needed if conc_flag is true.
    * n_layers: Number of layers to consider for concentration profiles. Since the central layer is centered on the interface, this must be an odd number. Each layer has a width of layer_width. Only needed if conc_flag is true.
    * nbins_per_layer: Number of overlapping bins per layer. For example if you have 3 bins per layer, then each layer contributes to 3 overlapping bins.
* nthreads: Number of threads to use. This number of trajectory frames will be analyzed in parallel, so you may need to be careful not to use too much RAM.
