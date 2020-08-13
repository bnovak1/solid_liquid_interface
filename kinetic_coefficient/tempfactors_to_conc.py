'''
First write a pdb file from the solid liquid interface code which will write the bin number to the
temperature factors field in pdb file. Also need the average concentrations in each bin.

This codes uses MDAnalysis changes the temperature factors to the average concentrations in the
pdb file for visualization in VMD.
'''
import sys
import numpy as np
import MDAnalysis as mda


def convert_tempfactors(pdb_infile, conc_file, pdb_outfile):
    '''
    Read pdb_infile input pdb file and conc_file which contains average concentrations in each bin
    in order from first bin to last bin, change tempfactors to 100*concentrations,
    then write pdb_outfile with interfacial region atoms.
    '''

    # Multiply by 100 to get more significant digits,
    # since only 2 numbers are written after the decimal in the pdb file
    concs = 100.0*np.loadtxt(conc_file)

    # Number of bins
    nbins = len(concs)

    # Read the pdb file
    snapshot = mda.Universe(pdb_infile, pdb_infile)

    # Change temperature factors
    tempfactors = np.copy(snapshot.atoms.tempfactors)
    for bin_num in range(nbins):

        ind = np.where(tempfactors == bin_num + 1)[0]
        tempfactors[ind] = concs[bin_num]

    snapshot.atoms.tempfactors = tempfactors

    # Write new pdb file with interface atoms
    ind = np.where(snapshot.atoms.tempfactors > 0)[0]

    snapshot.atoms[ind].write(pdb_outfile)


if __name__ == '__main__':

    PDB_INFILE = sys.argv[1]
    CONC_FILE = sys.argv[2]
    PDB_OUTFILE = sys.argv[3]

    convert_tempfactors(PDB_INFILE, CONC_FILE, PDB_OUTFILE)
