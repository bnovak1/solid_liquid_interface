import numpy as np
import scipy.constants as constants

def calc_free_energy(infile_, X_):
    params = np.loadtxt(infile_)
    free_energy = (constants.R/1000.0)*(X_*np.log(X_) + (1-X_)*np.log(1-X_)) + np.polyval(params)
    return free_energy

def calc_delta_mu(infile_, X_):
    params = np.loadtxt(infile_)
    delta_mu = (constants.R/1000.0)*np.log(X_/(1-X_)) + np.polyval(params)
    return delta_mu
