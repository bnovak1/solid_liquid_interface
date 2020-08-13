import numpy as np
import sys

liquid_file = sys.argv[1]
solid_file = sys.argv[2]
lattice_param_file = sys.argv[3]
natoms_per_cell = int(sys.argv[4])
outfile = sys.argv[5]

liquid_data = np.loadtxt(liquid_file)[:2]
solid_data = np.loadtxt(solid_file)[:2]
lattice_param = np.loadtxt(lattice_param_file)[0]

latent_heat_data = np.array([liquid_data[0] - solid_data[0],
                             np.sqrt(liquid_data[1]**2.0 + solid_data[1]**2.0)])

latent_heat_data = np.hstack((latent_heat_data*natoms_per_cell/(lattice_param**3.0),
                              latent_heat_data*96.4855364))

np.savetxt(outfile, latent_heat_data.reshape(1, 4),
           header='Latent heat (eV/(angstrom^3 solid) | uncertainty for 95% CI | ' + \
                  'Latent heat (kJ/mol) | uncertainty for 95% CI')
