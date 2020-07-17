import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import statsmodels.api as sm
import pandas as pd
import my_plot_settings_article as mpsa


class InterfacialFreeEnergyFit:

    def __init__(self, infile):

        # Read spreadsheet into pandas dataframe
        self._data = pd.read_excel(infile, usecols='A:L')

        # Remove rows with missing molar volume or missing enthalpy of fusion
        ind = ~self._data['Molar Volume (m^3/mol)'].isna()
        self._data = self._data[ind]
        ind = ~self._data['ΔH (kJ/mol)'].isna()
        self._data = self._data[ind]

        self._R = constants.R*1000 # Gas constant in mJ/(mol-K)
        self._N_A = constants.N_A # Avogadro's constant

    def fit_all(self):
        '''
        Fit to all data
        '''

        self._fit_data = self._fit_Kaptay()

        with open('../results/fit_all.json', 'w') as jf:
            json.dump(self._fit_data, jf, indent=4)

        self._plot_fit('../results/fit_all_type.png', self._data['Type'])
        self._plot_fit('../results/fit_all_structure.png', self._data['Structure'])

    # def fit_bcc(self):
    #
    # def fit_fcc(self):
    #
    def fit_exp(self):
        '''
        Fit only to experimental data
        '''

        ind = self._data['Type'] == 'exp'
        self._fit_data = self._fit_Kaptay(ind)

        with open('../results/fit_exp.json', 'w') as jf:
            json.dump(self._fit_data, jf, indent=4)

        self._plot_fit('../results/fit_exp.png', self._data['Structure'][ind])

    # def fit_sim(self):

    def _fit_Kaptay(self, ind=[]):
        '''
        Fit to the Kaptay model which include terms proportional to the latent heat of fusion
        and to the temperature. Can specify a boolean array, ind, to choose a subset of the data.
        By default all data is used.
        '''

        if len(ind) == 0:
            ind = np.ones(self._data.shape[0], dtype=bool)

        enth = self._data['ΔH (kJ/mol)'][ind]*1e6 # Convert to mJ
        vmol = self._data['Molar Volume (m^3/mol)'][ind]
        temp = self._data['Temperature (K)'][ind]
        denom = vmol**(2/3)*self._N_A**(1/3)
        x1 = enth/denom # divide by denominator in relation
        x2 = self._R*temp/denom # Multiply by R, divide by denominator in relation
        x = np.column_stack((x1, x2))
        y = self._data['Interfacial Energy (mJ/m^2)'][ind]

        model = sm.OLS(y, x)
        results = model.fit()

        fit_data = self._compile_fit_data(x, y, results)

        return fit_data

    def _compile_fit_data(self, x, y, results):
        return {'x': x.tolist(), 'y': y.tolist(), 'y_predict': results.predict(x).tolist(),
                'parameters': {'mean': results.params.to_list(),
                               'uncertainty': np.diff(results.conf_int()).tolist(),
                               'covariance': np.array(results.cov_params()).tolist()},
                'rsq': results.rsquared,
                'AIC': results.aic}

    def _plot_fit(self, outfile, color_col=[]):

        if len(color_col) == 0:
            plt.plot(self._fit_data['y'], self._fit_data['y_predict'], 'o', mfc='none')
        else:
            for type in np.unique(color_col):
                ind = color_col == type
                plt.plot(np.array(self._fit_data['y'])[ind], np.array(self._fit_data['y_predict'])[ind],
                         'o', mfc='none', label=type)

        plt.plot(self._fit_data['y'], self._fit_data['y'])
        plt.xlabel('$\mathrm{\sigma}$ (mJ/m$^\mathrm{2}$)', labelpad=mpsa.axeslabelpad)
        plt.ylabel('$\mathrm{\sigma_{pred}}$ (mJ/m$^\mathrm{2}$)', labelpad=mpsa.axeslabelpad)
        plt.legend()

        mpsa.axis_setup('x')
        mpsa.axis_setup('y')

        mpsa.save_figure(outfile, 300)
        plt.close()
