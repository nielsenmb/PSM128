""" Module of estimating the large frequency separation.

"""

import numpy as np
from scipy.signal import correlate
import warnings

def to_log10(x, xerr):
    """ Transform to value to log10
    
    Takes a value and related uncertainty and converts them to logscale.
    Approximate.

    Parameters
    ----------
    x : float
        Value to transform to logscale
    xerr : float
        Value uncertainty

    Returns
    -------
    logval : list
        logscaled value and uncertainty

    """
    
    if xerr > 0:
        return [np.log10(x), xerr/x/np.log(10.0)]
    return [x, xerr]

def select_prior_data(pdata, numax, teff, KDEsize):
    """ Expand numax interval to reach sufficient KDE sample size

    If necessary, increases the range around starting numax, until the 
    sample contains at least KDEsize targets.

    Otherwise if the number of targets in the range around the input numax 
    is greater than KDEsize, KDEsize samples will be drawn from the 
    distribution within that range.

    Notes
    -----
    If downsampling is necessary it is done uniformly in numax. Multiplying 
    idx by a Gaussian can be done to change this to a normal distribution. 
    This hasn't been tested yet though.

    Parameters
    ----------
    numax : length 2 list [numax, numax_err]
        The estimate of numax and uncertainty in log-scale.
    KDEsize : int
        Number of targets to include in the KDE estimation.             

    Returns
    -------
    prior_data : panda
        Array of length equal to the total prior data sample that will be 
        used to compute the KDE. 1 for targets that are included in the 
        KDE estimation, and 0 otherwise.

    """

    nsigma = 1
    
    idx_numax = (np.abs(pdata.numax.values - numax[0]) < nsigma * numax[1]) 
    idx_teff = (np.abs(pdata.teff.values - teff[0]) < nsigma * teff[1])
    idx = idx_numax & idx_teff
    
    while len(pdata[idx_numax & idx_teff]) < KDEsize:
        
        idx_numax = (np.abs(pdata.numax.values - numax[0]) < nsigma * numax[1]) 
        idx_teff = (np.abs(pdata.teff.values - teff[0]) < nsigma * teff[1])
        idx = idx_numax & idx_teff
    
        if nsigma >= 20:
            break

        nsigma += 0.2

    ntgts = len(idx[idx==1])

    if ntgts == 0:
        raise ValueError('No prior targets found within range of target. This might mean no prior samples exist for stars like this, consider increasing the uncertainty on your numax input.')

    elif ntgts < KDEsize:
        warnings.warn(f'Prior sample is less than the requested {KDEsize}.')
        KDEsize = ntgts

    return pdata.sample(KDEsize, weights=idx, replace=False)


def select_freq_range(f, s, numax, W, N = 2):
    """ Select +/- N env half widths around numax
    
    Parameters
    ----------
    f : ndarray
        Array of frequency bins for the power spectrum.
    s : ndarray
        Array of power for each frequency bin.
    numax : float
        Input value of numax.
    W : float
        Envelope width.
    N : int, optional
        Number of widths to include either side of numax.
        
    Returns
    -------
    fenv : ndarray
        Slice of input array f corresponding to N times the envelope width W.
    senv : ndarray
        Slice of input array S corresponding to N times the envelope width W.
        
    """
    
    idx = abs(f-numax) < N * W
    
    return f[idx], s[idx]

def autocorrelate(f, s):
    """ Compute the autocorrelation function
    
    Takes a input array s and computes the autocorrelation function. Maximum
    correlation is as len(s)/2.
    
    Parameters
    ----------
    f : ndarray
        Array of frequency bins for the power spectrum.
    s : ndarray
        Array of power for each frequency bin.
        
    Returns
    -------
    lags : ndarray
        Lags in units of $\delta f$ where the autocorrelation is computed.
    acf : ndarray
        Autocorrelation of the spectrum s.
    
    """
    
    a = s-np.mean(s)
    acf = correlate(a,a, mode='same', method='fft')
    lags = f-min(f)
    n = int(np.floor(len(f)/2))
    return lags[:n][::-1], acf[:n]

def get_prior_dnu(prior):
    """ Get an estimate of the 5 sigma confidence interval of dnu"""
    # 10**np.median(prior.dnu.values)
    return np.percentile(prior.dnu.values, [50-99.9999998027/2, 50, 50 + 99.9999998027/2])

def get_env_width(prior):
    """ Get the median of the envelope width distribution"""
    return np.median(prior.env_width.values)

def get_mode_width(prior):
    """ Get the median of the mode width distribution. """
    return np.median(prior.mode_width.values)

def step_filter(dnu, w, df, norders=2):
    """ Setup the step filter
    
    Uses a series of N step functions of width dnu. The top of the step is 
    1 with a width equivalent to the estimated mode wdiths w, and 0 otherwise.
    
    Convolving the spectrum with a set of N step functions will convolve the 
    parts of the spectrum that are separated by dnu, reducing the relative
    power in frequency bins that are not.
        
    Parameters
    ----------
    dnu : float
        Rough estimate of dnu.
    w : float
        Rough estimate of the mode widths
    df : float
        Frequency resolution of the spectrum
    norders : int, optional
        Number of step functions to include in the filter.
    
    Returns
    -------
    filter : ndarray
        Step function filter to convole withe the power spectrum.
        
    """
    
    nw = int(np.floor(w/df))
    ndnu = int(np.floor(dnu/df))
    filt = np.zeros(ndnu)
    filt[:nw] = 1
    return filt.repeat(norders)


def bin_arrays(f, s, binfac):
    """ Bin array by an integer factor
    
    Parameters
    ----------
    f : ndarray
        Array of frequency bins for the power spectrum.
    s : ndarray
        Array of power for each frequency bin.
    binfac : int
        Factor to bin the array by.    
    
    """
    
    n = int(np.floor(len(f)/binfac))*binfac
    fbin = f[:n].reshape((-1,binfac)).mean(axis = 1)
    sbin = s[:n].reshape((-1,binfac)).mean(axis = 1)
    return fbin, sbin


def get_dnu(numax, teff, f, s, pdata, norders = 2, binfac = 2, KDEsize = 100):
    """ Estimates dnu based on power spectrum
    
    Takes the power spectrum of a solar-like oscillator
    and estimates dnu based on input numax and effective
    temperature.
    
    Parameters
    ----------
    numax : list
        Estimated value of numax and associated error
    teff : list
        Estimated value of teff and associated error
    f : ndarray
        Array of frequency bins for the power spectrum
    s : ndarray
        Array of power for each frequency bin
    norders : int
        Number of orders to use to construct the filter kernel.
    binfac : int
        Number of frequency bins to bin.
    KDEsize : int
        Number of prior targets to include in estimating expected values of
        dnu, mode width and envelope width.
    
    Returns
    -------
    dnu :
        Estimated value of dnu
    
    """
    
    log_numax = to_log10(*numax)
    log_teff = to_log10(*teff)

    prior = select_prior_data(pdata, log_numax, log_teff, KDEsize)
     
    prior_W = 10**get_env_width(prior)
    
    prior_gamma = 10**get_mode_width(prior)
    
    prior_dnu = 10**get_prior_dnu(prior)
    envelope = select_freq_range(f, s, numax[0], prior_W)
    
    f,s = bin_arrays(*envelope, binfac)
    
    df = np.median(np.diff(f))

    filt = step_filter(prior_dnu[1], prior_gamma, df, norders) 
    
    c = np.convolve(s, filt, 'same')
    
    lags, acfc = autocorrelate(f,c)
    
    fac = 0.25
    print((1-fac)*prior_dnu[0], (1+fac)*prior_dnu[2])
    idx = ((1-fac)*prior_dnu[0] < lags) & (lags < (1+fac)*prior_dnu[2])
                               
    acf_dnu = lags[idx][np.argmax(acfc[idx])]
    
    return acf_dnu 