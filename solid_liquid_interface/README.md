# JSON input file variables
* traj_dir: Name of directory containing trajectory frames.
* reference_prefix: Prefix for reference structure files. Currently, these are assumed to have extensions of .lammpstrj and .pdb and be in the traj_dir directory. Reference structures should be a perfect lattice with the same lattice parameter and in the same crystallographic orientation as the solid in your coexistence simulation. The .pdb file is needed by mdtraj and can be created from the .lammpstrj file.
* traj_pattern: Regular expression for trajectory file names not including the directory name which is assumed to be traj_dir.
* traj_top_file: Name of topology file for trajectory files not including the directory name which is assumed to be traj_dir. This is needed for mdtraj and can be created from one of the trajectory frames. Could use any type of topology file accepted by mdtraj such as pdb.
* temp: Temperature in Kelvin.
* latparam: Mean lattice parameter in angstroms, a, at melting temperature.
* n_neighbors: Number of nearest neighbors to consider for calculation of order parameter. 14 for bcc, 12 for fcc.
* smoothing_cutoff: Cutoff for smoothing when calculating psi (2.5a and )
* crossover: Value of psi halfway between pure liquid and pure solid used to determine interface position
* outfile_prefix: Prefix for output files
* psi_avg_flag: If true, average the value of the order parameter in pure liquid/solid instead of calculating interface stiffness for a coexistence system. Use this to figure out the crossover value. If not included, defaults to false.
* dimension: 1 for quasi-1D interface, 2 for 2D interface
* interface_options: Options for interface determination and concentrations for alloys.
    * traj_flag: If true, write pdb files for each frame with different B factors for different interfacial layers. Should be true or false.
    * conc_flag: For alloys, set to true to compute concentrations in each interfacial layer. Should be true or false.
    * free_boundaries: Specify as true if doing solidification simulations with free boundaries in the interface normal direction, otherwise false.
    * r_atoms: Dictionary of radii for each atom type in the system with atom names as the keys. A reasonable choice for this is the radius calculated from a perfect crystal which is about 0.35355a for fcc and 0.433a for bcc. In many cases, it is probably alright to set all atom types in an alloy to have the same radius. Only needs to be included if traj_flag or conc_flag is true.
    * r_probe: Probe radius in angstroms. A reasonable choice for this is probably the size of the smallest atom in the system. Only needs to be included if traj_flag or conc_flag is true.
    * n_layers: Number of layers of solid and number of layers of liquid to identify at each interface. Only needs to be included if traj_flag or conc_flag is true.
    * grid_spacing: Grid spacing for ITIM in angstroms. ITIM will not find all interfacial atoms. It will find more if grid_spacing is smaller, but at some point it will cost a lot more to find a few more atoms. Around 1/10 the size of the atoms seems to be a good compromise. Note that ITIM is probably not a good algorithm to use if trying to find concentrations when there is a large difference in atom sizes since proportionally more of the small atoms will be missed than the large atoms. Only needs to be included if traj_flag or conc_flag is true.
* nthreads: Number of threads to use. This number of trajectory frames will be analyzed in parallel, so you may need to be careful not to use too much RAM.
