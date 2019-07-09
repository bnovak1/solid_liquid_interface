import numpy as np
import sys
import my_io

if __name__ == '__main__':

    # command line arguments
    template = sys.argv[1]
    lattice_type = sys.argv[2]
    latparam = sys.argv[3]
    orientation = sys.argv[4]
    traj_name = sys.argv[5]
    outfile = sys.argv[6]

    to_replace = np.array(['[LATTICE_TYPE]', '[LATPARAM]', '[ORIENTATION]', '[TRAJ]'])
    replacements = np.array([lattice_type, latparam, orientation, traj_name])

	# read template file & make replacements
    with open(template, 'r') as f:
        template_data = np.array(f.readlines())

    template_data_new = my_io.replace_in_template(template_data, to_replace,
		replacements, outfile)
