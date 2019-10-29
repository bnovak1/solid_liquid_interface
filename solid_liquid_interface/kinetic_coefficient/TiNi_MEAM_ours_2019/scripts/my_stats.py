"""
Statistics
Uncertainty in uncorrelated 1D numpy array based on t-distribution: uncertainty
Empirical continuous distribution function: empirical_CDF
Slope and uncertainty with uncertainties in each y point: get_slope_with_uncy
Weighted mean: wghtdavg
Standard error of the weighted mean
    Based on number of samples: wghtdunc_nsamples
    Based on variance, base = sum(w**2)/sum(w)**2: wghtdunc_variance
    Based on variance, base = len(x): wghtunc_variance_nblocks
    SPSS: wghtunc_spss
    WinCross: wghtunc_wincross
"""


import numpy as np
import scipy.interpolate as interp
import scipy.stats as stats


def uncertainty(data, alpha=0.05):
    '''
    Compute uncertainty in uncorrelated data based on t-distribution.
    alpha: 1 - fractional confidence desired. For 95% confidence interval, alpha = 1 - 0.95 = 0.05.
    '''
    
    nsamples=data.shape[0] # Number of samples. Assumes data is 1D.
    
    sd = np.std(data, ddof=1) # Standard deviation
    se = sd/np.sqrt(nsamples) # Standard error
    tfactor = stats.t.isf(alpha/2, nsamples-1) # Degrees of freedom is number of samples - 1
    unc = se*tfactor # Uncertainty. Half width of confidence interval.
    
    return unc


def empirical_CDF(data):
    '''
    Empirical CDF. Return interpolating function.
    '''

    x = np.sort(np.unique(data))
    n = x.shape[0]
    y = np.arange(1, n+1) / n

    x = np.hstack((x[0] - (x[1] - x[0]), x))
    y = np.hstack((0, y))

    interp_func = interp.UnivariateSpline(x, y, s=0, k=1, ext='const')

    return interp_func


def get_slope_with_uncy(x,mu,sig,nsamples):
    """
    Calculate mean of nsamples slopes of data x,y where
    y values are drawn from Gaussian distributions with means mu and standard deviations sig.
    """

    import random

    n = len(x)
    y = np.zeros(n)
    A = np.vstack([x, np.ones(n)]).T

    slope = np.zeros(nsamples)
    for irnd in range(nsamples):

        for i in range(n):
            y[i] = random.gauss(mu[i],sig[i])

        slope[irnd],c = np.linalg.lstsq(A, y)[0]

    slope_mean = np.mean(slope)
    slope_unc = np.std(slope)

    return (slope_mean,slope_unc)

def wghtdavg(x,w):
    """
    Weighted average of x with weights w
    """

    m = sum(x*w)/sum(w);

    return m

def wghtdunc_nsamples(x,w):
    """
    Weighted mean and standard error of the weighted mean
    Wikipedia formula (http://en.wikipedia.org/wiki/Weighted_mean, Weighted sample variance section, w = num samples) times sum(w^2)/sum(w)^2
    Result depends slightly on magnitude of w (sampling frequency), normalize by min(w) so the smallest weight is 1 (1 sample).
    """

    m = wghtdavg(x,w)
    w = w/min(w)
    se = np.sqrt((1.0/(sum(w) - 1.0))*sum(w*(x-m)**2)*sum(w**2)/sum(w)**2)

    return (m,se)

def wghtdunc_variance(x,w):
    """
    Weighted mean and standard error of the weighted mean
    Wikipedia formula (http://en.wikipedia.org/wiki/Weighted_mean, Weighted sample variance section, variance = 1/w) times sum(w^2)/sum(w)^2
    """

    m = wghtdavg(x,w)
    w = w/sum(w)
    se = np.sqrt((1.0/(1.0-sum(w**2)))*sum(w*(x-m)**2)*sum(w**2)/sum(w)**2)

    return (m,se)

def wghtdunc_variance_nblocks(x,w):
    """
    Weighted mean and standard error of the weighted mean
    Wikipedia formula (http://en.wikipedia.org/wiki/Weighted_mean, Weighted sample variance section, variance = 1/w) over number of blocks (length(x))
    """

    m = wghtdavg(x,w)
    w = w/sum(w)
    se = np.sqrt((1.0/(1.0-sum(w**2)))*sum(w*(x-m)**2)/len(x))

    return (m,se)

def wghtdunc_spss(x,w):
    """
    Weighted mean and standard error of the weighted mean
    SPSS (http://www.analyticalgroup.com/download/WEIGHTED_MEAN.pdf)
    """

    m = wghtdavg(x,w)
    se = np.sqrt((1.0/(sum(w)-1.0))*sum(w*(x-m)**2)/sum(w))

    return (m,se)

def wghtdunc_wincross(x,w):
    """
    Weighted mean and standard error of the weighted mean
    WinCross (http://www.analyticalgroup.com/download/WEIGHTED_MEAN.pdf)
    """

    m = wghtdavg(x,w)
    se = np.sqrt(np.var(x)*sum(w**2)/sum(w)**2)

    return (m,se)
