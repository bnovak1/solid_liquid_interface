import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import my_plot_settings_article as mpsa


nruns = int(sys.argv[1])
temperature_file = sys.argv[2]
melting_temp = float(sys.argv[3])
outfile_vel = sys.argv[4]
outfile_mu = sys.argv[5]
velocity_files = sys.argv[6:]
ntemps = len(velocity_files)

# Temperatures
with open(temperature_file) as f:
    temps = f.readlines()
temps = [t.rstrip('\n') for t in temps]
temps_num = np.array(temps, dtype=float)
ntemps = len(temps)

# Velocities
velocity = np.empty((ntemps, 4))

for itemp, file in enumerate(velocity_files):
    velocity[itemp, :] = np.loadtxt(file)

# Save velocity as a function of temperature
outdata = np.column_stack((temps_num, velocity))
np.savetxt(outfile_vel, outdata,
           header='Temperature (K) | Interface velocity from potential energy (m/s) | ' + \
                  'Interface velocity from interface positions (m/s)')

# Weighted linear fit to get kinetic coefficient. Force through 0, 0.
X = melting_temp - temps_num

Y = -velocity[:, 0]
W = 1/velocity[:, 1]**2
wls_model = sm.WLS(Y, X, weights=W)
results = wls_model.fit()
mu_pe = results.params[0]
mu_pe_unc = np.diff(results.conf_int()[0, :])
plt.plot(np.hstack((0, X)), np.hstack((0, Y)), 'bo', mfc='none', label='data PE')
plt.plot(np.hstack((0, X)), np.hstack((0, results.predict(X))), 'b--', label='fit PE')

Y = -velocity[:, 2]
W = 1/velocity[:, 3]**2
wls_model = sm.WLS(Y, X, weights=W)
results = wls_model.fit()
mu_int = results.params[0]
mu_int_unc = np.diff(results.conf_int()[0, :])
plt.plot(np.hstack((0, X)), np.hstack((0, Y)), 'ro', mfc='none', label='data direct')
plt.plot(np.hstack((0, X)), np.hstack((0, results.predict(X))), 'r--', label='fit direct')

plt.xlabel('$T_m - T$ (K)', labelpad=mpsa.axeslabelpad)
plt.ylabel('Interface velocity (m/s)', labelpad=mpsa.axeslabelpad)
plt.legend()
mpsa.save_figure('.'.join(outfile_vel.split('.')[:-1]) + '_fit.png', 300)

# Save kinetic coefficient
np.savetxt(outfile_mu, np.hstack((mu_pe, mu_pe_unc, mu_int, mu_int_unc)),
           header='Kinetic coefficient (m/s-K) based on potential energy | ' + \
                  'Uncertainty in kinetic coefficient (m/s-K) based on potential energy | ' +
                  'Kinetic coefficient (m/s-K) based on interface position | ' + \
                  'Uncertainty in kinetic coefficient (m/s-K) based on interface position')
