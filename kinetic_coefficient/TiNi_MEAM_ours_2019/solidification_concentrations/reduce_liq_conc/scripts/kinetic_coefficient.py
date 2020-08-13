import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

Xliq0 = 0.11

X = ['0.05', '0.06', '0.07', '0.08', '0.09', '0.10']
nconcs = len(X)
Xliq = np.empty(nconcs)
Xsol = np.empty(nconcs)
interface_vel = np.empty(nconcs)

conc_solid_file = 'conc_solid_avg.dat'
conc_liquid_file = 'conc_liquid_avg.dat'
interface_vel_file = 'interface_velocity.dat'

for iconc in range(nconcs):

    conc = X[iconc]

    Xsol[iconc] = np.loadtxt('../analysis/' + conc + '/' + conc_solid_file)
    Xliq[iconc] = np.loadtxt('../analysis/' + conc + '/' + conc_liquid_file)
    interface_vel[iconc] = np.loadtxt('../analysis/' + conc + '/' + interface_vel_file)[0]

y = Xliq/Xliq0 - 1
x = (1 - 0.9*Xsol/Xliq)*interface_vel
model = sm.OLS(y, x)
results = model.fit()

plt.plot(x, y, label='Data')
plt.plot(x, x*results.params, '--', label='Fit')
plt.legend()
plt.xlabel('(1-k)V')
plt.ylabel('$\mathrm{c_l/c_{l,0} - 1}$')
plt.title('$\mathrm{\\beta}$ = ' + str(round(float(results.params), 4)) + '$\mathrm{\pm}$' + \
          str(round(float(np.diff(results.conf_int())/2), 4)))
plt.savefig('../analysis/beta_all.png')
plt.close()
