import sys
import numpy as np
import my_stats

outfile = sys.argv[1]
velocity_files = sys.argv[2:]
nsims = len(velocity_files)

velocity = np.empty((nsims, 2))

for isim, velocity_file in enumerate(velocity_files):
    velocity[isim, :] = np.loadtxt(velocity_file, usecols=[0, 2])

velocity_mean_pe = np.hstack((np.mean(velocity[:, 0]), my_stats.uncertainty(velocity[:, 0])))
velocity_mean_int = np.hstack((np.mean(velocity[:, 1]), my_stats.uncertainty(velocity[:, 1])))
velocity_mean = np.hstack((velocity_mean_pe, velocity_mean_int))

np.savetxt(outfile, velocity_mean,
           header='Mean velocity (m/s) based on potential energy | ' + \
                  'Uncertainty in velocity (m/s) based on potential energy | ' + \
                  'Mean velocity (m/s) based on interface positions | ' + \
                  'Uncertainty in velocity (m/s) based on interface positions')
