import numpy as np
import sys

thermo_file = sys.argv[1]
pe_col = int(sys.argv[2])
depth_col = int(sys.argv[3])
latent_heat_file = sys.argv[4]
volume_change_file = sys.argv[5]
area_file = sys.argv[6]
interface_fit_file = sys.argv[7]
outfile = sys.argv[8]

thermo_data = np.loadtxt(thermo_file)
time = thermo_data[:, 0]
pe = thermo_data[:, pe_col]
depth = thermo_data[:, depth_col]

latent_heat = np.loadtxt(latent_heat_file)
volume_change = np.loadtxt(volume_change_file)
area = np.loadtxt(area_file)

interface_fit_data = np.loadtxt(interface_fit_file)

interface_dist_start = interface_fit_data[1, 4] - 12.0
ind_start = np.where(interface_fit_data[1:, 4] >= interface_dist_start)[0][-1] + 1
tstart = interface_fit_data[ind_start, 0]

interface_dist_end = 50.0
ind_end = np.where(interface_fit_data[1:, 4] <= interface_dist_end)[0][0] - 1
tend = interface_fit_data[ind_end, 0]

ind_thermo = np.intersect1d(np.where(time >= tstart)[0], np.where(time <= tend)[0])
pe_deriv = np.polyfit(time[ind_thermo], pe[ind_thermo], 1)[0]
depth_deriv = np.polyfit(time[ind_thermo], depth[ind_thermo], 1)[0]

velocity_pe = (1.0/(2.0*area*latent_heat[0]))*pe_deriv
velocity_pe *= 100.0

velocity_depth = (1.0/(2.0*volume_change[0]))*depth_deriv
velocity_depth *= 100.0

ind_direct = range(ind_start, ind_end+1)
interface_dist_deriv = np.polyfit(interface_fit_data[ind_direct, 0],
                                  interface_fit_data[ind_direct, 4], 1)[0]
velocity_direct = 0.5*(interface_dist_deriv - depth_deriv)
velocity_direct *= 100.0

outdata = np.array([velocity_pe, velocity_depth, velocity_direct]).reshape(1, 3)
np.savetxt(outfile, outdata,
           header='Interface velocity (m/s): From potential energy slope | ' + \
                  'From system depth slope | ' + \
                  'From slope of distance between interfaces')
