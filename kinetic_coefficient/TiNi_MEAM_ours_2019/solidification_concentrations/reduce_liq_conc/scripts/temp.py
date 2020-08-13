def find_interfacial_atoms_2D_nearest(x, y, height, coords, traj_file, snapshot,
                                      interface_options, latparam):

    n_layers = interface_options['n_layers']

    interfaces = np.empty((2, n_layers), dtype=object)

    bin_width = np.max(interface_options['r_max'].values())

    for iint in range(2):

        z_interp = interp.RectBivariateSpline(x, y, height[:, :, iint])

        atoms = np.intersect1d(np.where(coords[:, 2] > np.min(height[:, :, iint]) - \
                                        bin_width*n_layers/2.0)[0],
                               np.where(coords[:, 2] < np.max(height[:, :, iint]) + \
                                        bin_width*n_layers/2.0)[0])

        interface_projections = z_interp.ev(coords[atoms, 0], coords[atoms, 1])

        for ilayer in range(interface_options['n_layers']):

            lower_shift = -bin_width*(n_layers/2.0 - ilayer)
            upper_shift = -bin_width*(n_layers/2.0 - ilayer - 1)

            ind = np.intersect1d(np.where(coords[atoms, 2] > interface_projections + lower_shift)[0],
                                 np.where(coords[atoms, 2] < interface_projections + upper_shift)[0])

            interfaces[iint, ilayer] = atoms[ind]

    return interfaces
