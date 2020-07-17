import sys
import numpy as np
import my_io

if __name__ == '__main__':

	# command line arguments
    template = sys.argv[1]
    potential = sys.argv[2]
    temperature = sys.argv[3]
    outfile = sys.argv[4]

    to_replace = np.array(['[POTENTIAL]', '[TEMPERATURE]'])
    replacements = np.array([potential, temperature])

	# read template file & make replacements
    with open(template, 'r') as f:
        template_data = np.array(f.readlines())

    template_data_new = my_io.replace_in_template(template_data, to_replace,
		replacements, outfile)
