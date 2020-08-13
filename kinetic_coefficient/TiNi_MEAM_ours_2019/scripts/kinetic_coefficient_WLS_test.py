import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys

vel_file = sys.argv[1]
melting_point = float(sys.argv[2])
outfile = sys.argv[3]

data = pd.read_csv(vel_file, sep='\s+')

orientations = np.unique(data['orient y'])

mu = pd.DataFrame(columns=['Method', 'orient x', 'orient y', 'orient z', 'deltaT min', 'mu', 'mu unc (a=0.05)'])

cnt = 0
for orientation in orientations:

    ind = np.intersect1d(np.where(data['orient y'] == orientation)[0],
                         np.where(data['interface velocity using PE'] != 0)[0])
    deltaT = np.array(data['temperature (K)'][ind] - melting_point)
    vel_pe = -np.array(data['interface velocity using PE'][ind])
    vel_pe_unc = np.array(data['interface velocity uncertainty for 95% CI using PE'][ind])
    vel_dir = -np.array(data['interface velocity using interface positions'][ind])
    vel_dir_unc = np.array(data['interface velocity uncertainty for 95% CI using interface positions'][ind])
    npts = len(deltaT)

    ind_sort = np.argsort(deltaT)[::-1]
    deltaT = deltaT[ind_sort]
    vel_pe = vel_pe[ind_sort]
    vel_pe_unc = vel_pe_unc[ind_sort]
    vel_dir = vel_dir[ind_sort]
    vel_dir_unc = vel_dir_unc[ind_sort]

    orientationx = data['orient x'][ind].iloc[0]
    orientationz = data['orient z'][ind].iloc[0]

    for imax in range(2, npts+1):

        wls_model = sm.WLS(vel_pe[:imax], deltaT[:imax])#, weights=1/vel_pe_unc[:imax])
        results = wls_model.fit()
        mu.loc[cnt] = ['PE', orientationx, orientation, orientationz, deltaT[imax-1],
                     results.params[0], float(np.diff(results.conf_int())/2)]
        cnt += 1

        wls_model = sm.WLS(vel_dir[:imax], deltaT[:imax])#, weights=1/vel_dir_unc[:imax])
        results = wls_model.fit()
        mu.loc[cnt] = ['direct', orientationx, orientation, orientationz, deltaT[imax-1],
                     results.params[0], float(np.diff(results.conf_int())/2)]
        cnt += 1

mu.to_csv(outfile, index=False)
