import numpy as np
import sys

def calc_lattice_param(temperature):

    volume_per_atom = 10.903456 + 5.2815496e-4*temperature - \
                      6.9286058e-8*temperature**2.0 + 7.6641229e-11*temperature**3.0
    # volume_per_atom = 10.371678 + 1.8632748e-3*temperature - \
    #                   4.4098649e-7*temperature**2.0 + 1.1025970e-10*temperature**3.0

    lattice_param = (4.0*volume_per_atom)**(1.0/3.0)

    return lattice_param


if __name__ == "__main__":

    temperature = float(sys.argv[1])
    outfile = sys.argv[2]

    lattice_param = calc_lattice_param(temperature)

    np.savetxt(outfile, [lattice_param], header='Lattice parameter (angstrom)')
