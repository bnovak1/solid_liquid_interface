import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import my_plot_settings_article as mpsa
import json

melting_temp = np.loadtxt('melting_point.dat')

temps = np.loadtxt('temperatures.dat', dtype=str)
temps_num = temps.astype(float)
ntemps = len(temps)

with open('simulate.json', 'r') as f:
    json_data = json.loads(f.read())

orientations = json_data['ORIENTATIONS']
norientations = len(orientations)

orientation_labels = json_data['ORIENTATION_LABELS']

nruns = 10

velocity_pe = np.zeros((norientations, ntemps, nruns))
velocity_ip = np.zeros((norientations, ntemps, nruns))

for iorientation in range(norientations):

    orientation = orientations[iorientation]

    for irun in range(nruns):

        run = str(irun + 1)

        for itemp in range(ntemps):

            temp = temps[itemp]

            vel_data = np.loadtxt('../results/velocity_' + temp + 'K_' + orientation + \
                                  '_' + run + '.dat')

            velocity_pe[iorientation, itemp, irun] = -vel_data[0]
            velocity_ip[iorientation, itemp, irun] = -vel_data[2]

velocity_mean_pe = np.mean(velocity_pe, axis=2)
velocity_mean_ip = np.mean(velocity_ip, axis=2)

Z = stats.t.isf(0.025, df=nruns)
velocity_unc_pe = Z*np.std(velocity_pe, axis=2)/np.sqrt(nruns)
velocity_unc_ip = Z*np.std(velocity_ip, axis=2)/np.sqrt(nruns)

temps_num = np.hstack((temps_num, melting_temp))
velocity_mean_pe = np.column_stack((velocity_mean_pe, np.zeros(norientations)))
velocity_mean_ip = np.column_stack((velocity_mean_ip, np.zeros(norientations)))
velocity_unc_pe = np.column_stack((velocity_unc_pe, np.zeros(norientations)))
velocity_unc_ip = np.column_stack((velocity_unc_ip, np.zeros(norientations)))

colors = ['b', 'g', 'r', 'k']

for iorientation in range(norientations):

    orientation = orientations[iorientation]
    color = colors[iorientation]

    fit_pe = np.polyfit(temps_num[:-1], velocity_mean_pe[iorientation, :-1], 1)
    fit_ip = np.polyfit(temps_num[:-1], velocity_mean_ip[iorientation, :-1], 1)

    kinetic_coefficient_pe = fit_pe[0]
    kinetic_coefficient_ip = fit_ip[0]

    # plt.errorbar(melting_temp - temps_num, velocity_mean_pe[iorientation, :],
    #              yerr=velocity_unc_pe[iorientation, :], fmt=color + 'o-', mec=color,
    #              mfc='none', label='Potential energy, ' + orientation)
    # plt.errorbar(melting_temp - temps_num, velocity_mean_ip[iorientation, :],
    #              yerr=velocity_unc_ip[iorientation, :], fmt=color + 's-', mec=color,
    #              mfc='none', label='Direct, ' + orientation)
    plt.plot(melting_temp - temps_num, velocity_mean_pe[iorientation, :],
             color + 'o-', mec=color, mfc='none',
             label='E: ' + json_data['ORIENTATION_LABELS'][orientation])
    plt.plot(melting_temp - temps_num, velocity_mean_ip[iorientation, :],
             color + 's-', mec=color, mfc='none',
             label='Direct: ' + json_data['ORIENTATION_LABELS'][orientation])

mpsa.axis_setup('x')
mpsa.axis_setup('y')

plt.xlabel('Undercooling (K)', labelpad=mpsa.axeslabelpad)
plt.ylabel('Interface velocity (m/s)', labelpad=mpsa.axeslabelpad)

# plt.title('Ti using Ni-Ti potential (MEAM with bcc reference)')

plt.legend(fontsize=7)

mpsa.save_figure('../results/velocities_all.png')
plt.close()


ntemps = len(temps_num)
orientx = np.hstack((np.repeat('1 0 0', ntemps), np.repeat('1 -1 0', ntemps),
                     np.repeat('0 0 1', ntemps), np.repeat('1 -1 0', ntemps)))
orienty = np.hstack((np.repeat('0 1 0', ntemps), np.repeat('0 0 -1', ntemps),
                     np.repeat('1 -1 0', ntemps), np.repeat('1 1 -2', ntemps)))
orientz = np.hstack((np.repeat('0 0 1', ntemps), np.repeat('1 1 0', ntemps),
                     np.repeat('1 1 0', ntemps), np.repeat('1 1 1', ntemps)))
T = np.hstack((temps_num, temps_num, temps_num, temps_num))

for iorientation in range(norientations):
    try:
        v_pe = np.hstack((v_pe, velocity_mean_pe[iorientation, :]))
        v_u_pe = np.hstack((v_u_pe, velocity_unc_pe[iorientation, :]))
        v_ip = np.hstack((v_ip, velocity_mean_ip[iorientation, :]))
        v_u_ip = np.hstack((v_u_ip, velocity_unc_ip[iorientation, :]))
    except NameError:
        v_pe = velocity_mean_pe[iorientation, :]
        v_u_pe = velocity_unc_pe[iorientation, :]
        v_ip = velocity_mean_ip[iorientation, :]
        v_u_ip = velocity_unc_ip[iorientation, :]

d = np.column_stack((orientx, orienty, orientz, T, v_pe, v_u_pe, v_ip, v_u_ip))
df = pd.DataFrame(d, columns=['orient x', 'orient y', 'orient z', 'temperature (K)',
                  'interface velocity using PE',
                  'interface velocity uncertainty for 95% CI using PE',
                  'interface velocity using interface positions',
                  'interface velocity uncertainty for 95% CI using interface positions'])
df.to_csv('../results/velocities_all.dat', sep=' ', index=False)
