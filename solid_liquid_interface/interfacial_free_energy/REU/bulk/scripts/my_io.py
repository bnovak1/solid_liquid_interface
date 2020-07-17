from itertools import cycle
import numpy as np
import os

def write_xyz(fout, coords, title="", atomtypes=("A",)):
    """
    https://github.com/pele-python/pele/blob/master/pele/utils/xyz.py
    write a xyz file from file handle
    Writes coordinates in xyz format. It uses atomtypes as names. The list is
    cycled if it contains less entries than there are coordinates.
    ----------

    Parameters
    fout : an open file
    coords : np.array
        array of coordinates
    title : title section, optional
        title for xyz file
    atomtypes : iteratable
        list of atomtypes.
    """

    fout.write("%d\n%s\n" % (coords.size / 3, title))
    for x, atomtype in zip(coords.reshape(-1, 3), cycle(atomtypes)):
        fout.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))

def read_xvg(infile): # Does not work if imported, works if copied current file.
    """
    Read .xvg files by discarding header lines starting with # or @
    """

    with open(infile, 'r') as f:
        cnt = 0
        nskiprows = 0
        while nskiprows == 0:
            data = f.readline()
            if data[0] != '#' and data[0] != '@':
                nskiprows = cnt
            cnt += 1

    data = np.loadtxt(infile, skiprows=nskiprows)

    return data


def copy_files(infiles, outfiles):
    """
    Copy files using names from one list (infiles) to files using names from another list
    (outfiles), elmentwise.
    """

    import shutil

    nfiles = len(infiles)
    for ifile in range(nfiles):
        shutil.copy(infiles[ifile], outfiles[ifile])


def replace_in_template(data, toreplace, replacements, outfile):

    assert toreplace.shape[0] == replacements.shape[0], \
        'Number of things to replace must match number of things to replace them with.'

    for ireplace in range(toreplace.shape[0]):
        data = np.core.defchararray.replace(data, toreplace[ireplace],
                                            replacements[ireplace])

    with open(outfile, 'w') as f:
        for line in data:
            f.write(line)


def edit_ffs_template(template, left_dir, right_dir, outfile):

        to_replace = np.array(['[LEFT_DIR]', '[RIGHT_DIR]'])
        replacements = np.array([left_dir, right_dir])

    	# read template file & make replacements
        with open(template, 'r') as f:
            template_data = np.array(f.readlines())

        template_data_new = replace_in_template(template_data, to_replace, replacements,
                                                outfile)
