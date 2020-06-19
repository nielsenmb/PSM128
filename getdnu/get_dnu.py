""" Module for estimating the large frequency separation.

"""

import numpy as np
from scipy.signal import correlate

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
    
    while len(pdata[idx]) < KDEsize:
        
        idx_numax = (np.abs(pdata.numax.values - numax[0]) < nsigma * numax[1]) 
        idx_teff = (np.abs(pdata.teff.values - teff[0]) < nsigma * teff[1])
        idx = idx_numax & idx_teff
    
        if nsigma >= 20:
            break

        nsigma += 0.2

    ntgts = len(idx[idx==1])

    if ntgts < 10:
        raise ValueError('No prior targets found within range of target. This might mean no prior samples exist for stars like this, consider increasing the uncertainty on your numax input.')

    elif ntgts < KDEsize:
        KDEsize = ntgts

    return pdata.sample(KDEsize, weights=idx, replace=False)


def select_freq_range(f, s, numax, W, nwidths = 2):
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
    nwidths : int, optional
        Number of widths to include either side of numax.
        
    Returns
    -------
    fenv : ndarray
        Slice of input array f corresponding to N times the envelope width W.
    senv : ndarray
        Slice of input array S corresponding to N times the envelope width W.
        
    """
    
    idx = abs(f-numax) < nwidths * W
    
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
    return lags[:n-1][::-1], acf[:n-1]

def get_prior_dnu(prior):
    """ Get an estimate of the 5 sigma confidence interval of dnu"""
    # 10**np.median(prior.dnu.values)
    return np.percentile(prior.dnu.values, [50-99.9999998027/2, 50, 50 + 99.9999998027/2])

def get_env_width(prior):
    """ Get the median of the envelope width distribution"""
    return np.median(prior.env_width.values)

def make_filter(w, df):
    """ Setup the step filter
    
    At the moment this is just a boxcar. 
            
    Parameters
    ----------
    w : float
        Rough estimate of the mode widths
    df : float
        Frequency resolution of the spectrum

    Returns
    -------
    filter : ndarray
        Filter to convole withe the power spectrum.
        
    """
    
    nw = int(np.floor(w/df))
    
    if nw < 2:
        nw = 2

    return np.ones(nw)

def find_peaks(x, y, dnu):
    
     
    idx = (y[:-3] < y[1:-2]) & (y[1:-2] > y[2:-1])
    pidx = np.append(np.append(False,idx),[False,False])
    
    fac = 1
    ridx = (dnu[0] < x) & (x < dnu[2])
   
    # Expand the range around dnu by up to 10%
    while len(y[pidx & ridx]) == 0:
        fac += 0.01
        ridx = (fac*dnu[0] < x) & (x < fac*dnu[2])
        if fac >= 1.1:
            break
    
    # If no peak in raneg of dnu is found, pick maximum
    if not any(y[pidx & ridx]):
        k = np.argmax(y[ridx])
        pidx[ridx][k] = True
    
    idx = pidx & ridx
    
    return pidx & ridx

def get_dnu(numax, teff, f, s, pdata, KDEsize = 100, nwidths = 3):
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
        
    prior_dnu = 10**get_prior_dnu(prior)
    
    f, s = select_freq_range(f, s, numax[0], prior_W, nwidths)
    
    df = np.median(np.diff(f))
    
    filter_width = prior_dnu[1]/5
    
    filt = make_filter(filter_width, df) 
    
    fudge = 2*int(np.floor(filter_width/df))
    
    c = np.convolve(s, filt, 'full')[fudge:len(s)]
    
    lags, acfc = autocorrelate(f[fudge:],c)
    
    idx = find_peaks(lags, acfc, prior_dnu)
    
    lags_slice = lags[idx]
    
    acfc_slice = acfc[idx]
    
    acf_dnu = lags_slice[np.argmax(acfc_slice)]
    
    return acf_dnu, lags, acfc