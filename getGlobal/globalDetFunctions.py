import LightkurveCacheAccess as lkAcc
import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse as sp
from scipy.stats import median_absolute_deviation
import scipy.sparse.linalg as spln
import emcee, warnings, os
import scipy.special as sc

def normlogpdf(x, mu, sigma=None, normed=True):
    """ Gaussian log probability for x ~ N(mu,sigma)
    
    Computes the logarithmic probability density of x, 
    given a mean mu and a covariance matrix S.
        
    Parameters
    ----------
    x : array
        Points at which to evaluate the log-pdf
    mu : array
        Mean values of the pdf. 
    S : array
        Covariance matrix of shape NxN where N is the
        length x. Can be a numpy array or a scipy.sparse
        matrix.
    """
    
    if isinstance(sigma, (np.float, np.int)):
        if normed:
            return np.sum(-np.log(np.sqrt(2*np.pi)*sigma) - 0.5*(x-mu)**2/sigma**2)
        else:
            return -0.5*(x-mu)**2/sigma**2
        
    nx = sigma.shape[0]
    norm_coeff = nx*np.log(2*np.pi)+np.linalg.slogdet(sigma.toarray())[1]

    err = x-mu
    if sp.issparse(sigma):
        numerator = spln.spsolve(sigma, err).T.dot(err)
    else:
        numerator = np.linalg.solve(sigma, err).T.dot(err)

    return -0.5*(norm_coeff+numerator)

def chi2isf(p, df, scale):
    """ Inverse survival function for chi2 distribution 
        
    Parameters
    ----------
    p : float
        Probability to evaluate
    df : int
        Degrees of freedom for the chi2 distribution
    scale : float
        Scale parameter. Should be 1/df.
        
    Returns
    -------
    x : float
        Value that yields p in the chi2-isf.
    """
    
    return sc.chdtri(df, p)*scale

def chi2logsf(x, df, scale):
    """ Log-Survival function for chi2 distribution 
    
    The survival function is 1-CDF    
    
    Parameters
    ----------
    x : float
        Probability to evaluate the survival function at
    df : int
        Degrees of freedom for the chi2 distribution
    scale : float
        Scale parameter. Should be 1/df.
        
    Returns
    -------
    p : float
        log-probability of the chi2-isf at x.
    """
    return np.log(sc.chdtrc(df, x/scale))

def chi2logcdf(x, df, scale):
    """ log-cumulative distribution function for chi2 distribution 
    
    Parameters
    ----------
    x : float
        Probability to evaluate the CDF at
    df : int
        Degrees of freedom for the chi2 distribution
    scale : float
        Scale parameter. Should be 1/df.
        
    Returns
    -------
    p : float
        log-probability of the chi2-cdf at x.
    """
    return np.log(sc.chdtr(df, x/scale))

def chi2logpdf(x, df, scale, normed=True):
    """ Compute log prob of chi2 dist.
    
    If normed=True this is equivalent to using the scipy.stats chi2 as
    chi2.logpdf(x, df=df, loc=0, scale=scale)
    
    If normed=False the normalization to unit area is discarded to speed 
    up the computation. This is a fudge for using this with MCMC etc.
    
    Parameters
    ----------
    x : array
        Points at which to evaluate the log-pdf
    df : int
        Degrees of freedom of the chi2 distribution
    scale : float
        Scale factor for the pdf. Same as the scale parameter in the
        scipy.stats.chi2 implementation.
    
    Returns
    -------
    logpdf : array
        Log of the pdf of a chi^2 distribution with df degrees of freedom. 
    
    """
    x /=scale
    
    if normed:
        return -(df/2)*np.log(2) - sc.gammaln(df/2) - x/2 - np.log(scale) + np.log(x)*(df/2-1)
    else:
        return np.log(x)*(df/2-1) - x/2

def numaxExceedsLimits(numax, Teff, f):
    """ Test numax is within limits
    
    - Based on the input Teff we can put some simple limits on what numax
    values should be considered. Should be generous since Teff isn't 
    always that accurate/precise.
    
    - Numax should probably also be within some reasonable bounds based on 
    the observations. Not 0 or greater than Nyquist for example.
    
    Parameters
    ----------
    numax : float
        Test value of numax
    f : array
        Frequency axis for the spectrum
    
    Returns
    -------
    x : bool
        True if inside the limits, False if outside the limits.
    """
    
    TeffRed0 = 9200 # This should be 8907, but I seems to cut off some red-clump targets otherwise
    Teff0 = 5777
    numax0 = 3050
    
    beta0 = numax < numax0 * ((Teff/TeffRed0) * (Teff/Teff0)**0.47)**(1/0.11)
    
    NumaxTooLo = numax <= max([0, f[1]])
    
    NumaxTooHi = numax > f[-1]
    
    if any([beta0, NumaxTooLo, NumaxTooHi]):
        return True
    else:
        return False

def dnuExceedsLimits(dnu, f):  
    """ Test dnu is within limits
    
    These are probably not necessary since the numax constraint is much 
    tighter. It may still be useful for the sampling process.
    
    Parameters
    ----------
    dnu : float
        Test dnu 
    f : array
        Frequency axis for the spectrum
    
    Returns
    -------
    x : bool
        True if inside the limits, False if outside the limits.   
    """
    
    dnuTooLo = dnu <= 0 # This is probably the most likely limit to be required
    dnuTooHi = dnu >= f[-1] # This probably very unlikely
    
    if any([dnuTooLo, dnuTooHi]):
        return True
    else:
        return False

def logProbEmcee(theta, fb, pb, bb, Nbin, Nyquist, dfb, lags, acf, C, Teff, 
                   numaxGuess):
    """ Wrapper for all the likelihood functions
    
    This is a wrapper for all the likelihood functions.
    
    Simply add more priors to the return line if needed.
    
    If some of the parameters exceed the hard limits, returns -inf before
    anything else is done. (Not the neatest way to write it up, but it's faster)
    
    Parameters
    ----------
    theta : array
        Model parameters to calculate the likelihood at. numax, dnu, lorentzian
        height, lorentizan width, background.
    fb : array
        Frequency array for the spectrum (binned).
    pb : array
        Power array for the spectum (binned)
    bb : array
        Background level for the spectrum (binned)
    Nbin : int
        Binning factor of the spectrum
    Nyquist : float
    
    lags : array
        Lag frequencies (test Dnu values) for the autocorrelation of the spectrum
    acf : array
        Autocorrelation of the spectrum
    C : sparse array or float
        Variance of the ACF bins. Can either be a float, in which case no 
        co-variance is assumed (this is a fudge), or a MxM sparse Scipy matrix 
        which includes the co-variances of the ACF bins, where M is the length
        of the ACF array. 
    Teff : float
        Effective temperature estimate of the star in question
    numaxGuess : float
        Initial guess for numax, doesn't have to be very accurate. This is used
        to apply a Gaussian prior with a std of 50% of the guess value. 
        
    Returns
    -------
    lnprob : float
        Sum of the log-likelihood functions and log-priors.
    
    """
    
    # Some simple limit checks first, that should break the probability early
    # if not satisfied. 
    if numaxExceedsLimits(theta[0], Teff, fb):
        return -np.inf
    
    if dnuExceedsLimits(theta[1], fb):
        return -np.inf
    
    # ACF lorentzian height limits
    if (theta[2] < 0):
        return -np.inf
    
    # ACF lorentzian width limits
    if (theta[3] < -3) or (theta[3] > 2):
        return -np.inf
    
    logp_numax = numaxLogProbability(theta[0], fb, pb, bb, Teff, dfb, Nbin, 
                                     Nyquist, numaxGuess) 
    
    logp_dnu = dnuLogProbability(theta[1:], lags, acf, C)
    
    logp_dnuNumaxPrior = dnuNumaxPrior(theta[0], theta[1])
    
    return np.sum(logp_numax) + logp_dnu + logp_dnuNumaxPrior

def dnuLogProbability(phi, lags, acf, C):
    """ Evaluate ACF model probability
    
    Evaulates the probability of observing a model consisting of a series of 
    lorentzian peaks, given an ACF.
    
    The likelihood is assumed to follow a normal distribution with correlation
    matrix C (C can also be a float, in which case ACF bins are assumed
    independent, which they aren't!)
    
    Parameters
    ----------
    phi : array
        Array of model parameters
    lags : array
        Frequency lags for the ACF
    acf : array
        Auto-correlation values of the power spectrum
    C : float, scipy.sparse matrix
        Correlation between the acf bins. If float the bins are assumed to be 
        independent. Can also be a NxN matrix (must be scipy.sparse), where N
        is the length of the ACF, and the entries are estimates of the 
        correlation coefficients between each bin.
    
    Returns
    -------
    logp : float
        Log-probability of the observed ACF model.
    """
    
    Dnu, H, w, B = phi
        
    w = 10**w
    
    model = lorentz(lags, acf, Dnu, H, w, B)

    logp = normlogpdf(model, acf, C)
    
    return logp

def dnuNumaxPrior(numax, Dnu, lims = [-0.13, 0.21]):
    """ Add the dnu/numax relation as a prior
    
    The relative difference between Dnu and the scaling relation estimate of Dnu
    based on the input numax falls between more or less fixed range, based on 
    previous observations from Kepler. So
    
    a <= Dnu / Dnu_scaling(numax)-1 < b
    
    We can then add a uniform prior, that limits the values of dnu and numax. 
    
    Can also be replaced by a normal distribution, but since the ACF is 
    multi-modal, the fit might get stuck on the wrong peak, so a uniform prior
    is perhaps safer.
    
    Parameters
    ----------
    numax : float
        Value of numax to test.
    Dnu : float
        Value of Dnu to test.
    lims : list, optional
        Limits of the relative difference between Dnu and numax.
    
    Returns
    -------
    logprior : float
        0 if the combination of numax and Dnu are inside the range, -inf otherwise.
    
    """
    
    r = Dnu / dnuScale(numax)-1
    
    if (r <= lims[0]) or (lims[1]< r):
        return -np.inf
    else:
        return 0
#        logp = norm.logpdf(r, 0, 0.12)
#
#        if np.isnan(logp):
#            return -np.inf
#        else:
#            return logp


def runSampler(fb, pb, bb, Nbin, Nyquist, lags, acf, C, Teff, numaxGuess, func,
                ndim, nwalkers=50, bsteps=500, nsteps=200, method='emcee', 
                progress=False):
    """ Wrapper for running MCMC sampler
    
    Samples the likelihood space of the model func
    
    Parameters
    ----------
    fb : array
        Frequency array for the spectrum (binned).
    pb : array
        Power array for the spectum (binned)
    bb : array
        Background level for the spectrum (binned)
    Nbin : int
        Binning factor of the spectrum.
    Nyquist : float
        Nyquist frequency of the spectrum.
    lags : array
        Lag frequencies (test Dnu values) for the autocorrelation of the 
        spectrum
    acf : array
        Autocorrelation of the spectrum
    C : sparse array or float
        Variance of the ACF bins. Can either be a float, in which case no 
        co-variance is assumed (this is a fudge), or a MxM sparse Scipy matrix 
        which includes the co-variances of the ACF bins, where M is the length
        of the ACF array. 
    Teff : float
        Effective temperature estimate of the star in question
    numaxGuess : float
        Initial guess for numax, doesn't have to be very accurate. This is used
        to apply a Gaussian prior with a std of 50% of the guess value. 
    func : callable function
        Likelihood function to sample.
    ndim : int
        Number of dimensions of the model
    nwalkers : int
        Number of walkers to use for the sampling. Default is 50.
    bsteps : int, optional
        Number of steps to burn. Default is 500.
    nsteps : int, optional
        Number of steps to use for the result. Default is 200.
    method : str, optional
        Sampling method, options are 'emcee' and 'cpnest'?
    
    Returns
    -------
    samples : array
        Array of samples drawn from the likelihood function of shape (nsteps, 
        nwalkers, ndim)
    
    """
    
    dfb = np.median(np.diff(fb))
    
    # Initial guess position
    numax_i, dnu_i, H_i, w_i, B_i = numaxGuess, dnuScale(numaxGuess), max(acf[5:]), np.log10(0.2), np.mean(acf[-5:])

    if method == 'emcee':
        # Samples are started in a tight ball around the initial positions
        pos = np.array([numax_i, dnu_i, H_i, w_i, B_i]) + 1e-2*np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=(fb, pb, bb, Nbin, Nyquist, dfb, lags, acf, C, Teff, numaxGuess))
           
        pos, prob, state = sampler.run_mcmc(initial_state=pos, nsteps=bsteps, progress=progress);
        
        # Fold in low acceptance value walkers. This is a fudge.
        pos = fold(sampler, pos, spread=1e-4)
        
        sampler.reset()
        
        sampler.run_mcmc(initial_state=pos, nsteps=nsteps, progress=progress);
        
        samples = sampler.get_chain()
        
        sampler.reset()
        
    return samples

def tsToPsd(ID, downloadDir=None, mission='Kepler', cadence='long', skipfac=1800):
    """ Get TS and turn it into PSD
    
    Gets the lightcurve for a target either from cache or online and computes
    the periodogram of it along with the background estimate.
        
    Parameters
    ----------
    ID : str
        Target ID. Can be anything that Simbad or MAST can resolve. 
    download_dir : str
        Cache directory to store the lightcurves. Default is the Lightkurve 
        cache directory. 
    mission : str
        Select the mission to download data from. Default is Kepler. Cannot 
        use both Kepler and TESS for overlapping targets (yet).
    cadence : str
        Observation cadence to use. Should be selected based on the target in 
        question (ideally not). Default is long.
    skipfac : int
        The background estimation interpolates between points spaced by skipfac
        in the spectrum. The default value of 3600 frequency bins works for 
        Kepler targets, but found not necessarily for TESS. TODO: This should 
        probably scaled to the frequency resolution somehow.
    
    Returns
    -------
    f : array
        Array of frequency bins in the spectrum.
    p : array
        Array of psd values in the spectrum.
    b : array
        Array of psd values representing the background
    
    Notes
    -----
    In a few cases Lightkurve may complain that there are multiple targets for 
    a specific ID. It's lying, it happens when there are two targets within the
    target imagette (foregound/background targets). TODO: come up with a way to
    separate them.
    """
    if downloadDir is None:
        downloadDir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = lkAcc.query_lightkurve(ID, downloadDir, True, {'mission': mission, 'cadence': cadence})
        
    lc.flux = (lc.flux-1)*1e6

    f, p, pg = makePsd(lc, NyquistScale=1.05)

    b = getBkg(f, p, skips = len(f)//skipfac)  # 3600 is an arbitrarily chosen number
    idx = ~np.isnan(b)
    return f[idx], p[idx], b[idx]


def fold(sampler, pos, accept_lim = 0.2, spread=0.1):
    """ Fold low acceptance walkers into main distribution

    At the end of the burn-in, some walkers appear stuck with low
    acceptance fraction. These can be selected using a threshold, and
    folded back into the main distribution, estimated based on the median
    of the walkers with an acceptance fraction above the threshold.

    The stuck walkers are redistributed with multivariate Gaussian, with
    mean equal to the median of the high acceptance walkers, and a standard
    deviation equal to the median absolute deviation of these.

    Parameters
    ----------
    pos : ndarray, optional
        The positions of the walkers after the burn-in phase.
    accept_lim: float, optional
        The value below which walkers will be labelled as bad and/or hence
        stuck.
    spread : float, optional
        Factor by which to scatter the folded walkers.

    Returns
    -------
    pos : ndarray
        The positions of the walkers after the low accepatance walkers have
        been folded into high acceptance distribution.

    """
    
    idx = sampler.acceptance_fraction < accept_lim

    nbad, ndim = np.shape(pos[idx, :])

    if nbad > 0:
        flatchains = sampler.chain[~idx, :, :].reshape((-1, ndim))
        
        good_med = np.median(flatchains, axis = 0)
        
        good_mad = median_absolute_deviation(flatchains, axis=0) * spread
        
        pos[idx, :] = np.array([np.random.randn(ndim) * good_mad + good_med for n in range(nbad)])
        
    return pos


def getSummedAcfMuCov(ACF, width, sparse=False):
    """ Magic
    
    Parameters
    ----------
    ACF : array
        Autocorrelation of a power spectrum
    
    Returns
    -------
    ?? : int
    cov : array
        An NxN array covariance matrix of the ACF.
        Where N is the length of the ACF.
    """
    
    K = len(ACF)#ACF.shape[1]
    dN2 = (K + K*(K-1)*0.8)*(3.*K/2.)

    ones = np.ones(ACF.shape[0])
    frac = 1./width
    if sparse:
        C = sp.spdiags(ones, 0, K, K)
        for i in range(1, width):
            C += sp.spdiags(ones[:-i]*(1.-i*frac), i, K, K)
            C += sp.spdiags(ones[:-i]*(1.-i*frac),-i, K, K)
    else:
        C = np.diag(ones, 0)
        for i in range(1, width):
            C += np.diag(ones[:-i]*(1.-i*frac), i)
            C += np.diag(ones[:-i]*(1.-i*frac),-i)
        
    return 2*K**2, C*dN2

def lorentz(f, acf, Dnu, H, w, B, f0s=np.array([0.5, 1, 1.5, 2, 2.5]), 
            H0s=np.array([1.25, 1, 0.75, 0.5, 0.25])):
    """ Model of lorentzian peaks
    
    Computes a model consisting of three lorenztian profiles to match the 
    fundamental of the large separation and its harmonics observed in the ACF of
    the power spectrum of a solar-like oscillator. 
       
    As an approximation we set the height of the harmonics as some fraction of 
    the fundamental.
    
    A constant background level is also added. 
    
    Parameters
    ----------
    f :  array
        Frequencies (or lags) at which to compute the 
        model.
    Dnu : float
        Test frequency (or lag) of Dnu
    H : float
        Height of the peak at Dnu. 
    w : float
        Width of the lorentzian profiles.
    B : float
        Background offset of the model
    f0s : array, int, or float. Optional
        Scaling values of Dnu. If float should probably
        be 1, if array should probably contain at least
        1 to fit the main Dnu peak. 
    H0s : array, int, or float. Optional
        Scaling values of the lorentizan peaks. If float
        should probably be 1, if array should probably 
        contain at least 1 to fit the main Dnu peak. 
    """

    lors = np.sum(H0s*H/(1.0 + (f.reshape((-1,1)) - f0s*Dnu)**2/w**2), axis=1)
    lors += acf[0]/(1.0 + f**2/w**2)
    return B + lors

def numaxLogProbability(nu, fb, pb, bb, Teff, dfb, Nbin, Nyquist, numaxGuess):
    
    SNR, SNR_pred = getSNR(nu, fb, pb, bb, Teff, Nyquist)

    logp_numax = obsNumaxPrior(nu, fb, numaxGuess)
     
    logp_H0 = chi2logpdf(1+SNR, df=2*Nbin, scale=1/(2*Nbin), normed=False) 
    #log_like_H0 = chi2.logpdf(1+SNR, df=2*Nbin, scale=1/(2*Nbin))
    
    logp_H1 = chi2logsf((1+SNR_pred)/(1+SNR), df=2*Nbin, scale=1/(2*Nbin))  
    #log_like_H1 = chi2.logsf((1+SNR_pred)/(1+SNR), df=2*Nbin, scale=1/(2*Nbin))  
    
    logp_threshold = thresholdPrior(SNR_pred, Nbin)
    
    logp_bkgRatio = bkgRatioPrior(nu, fb, bb, Teff, Nbin)
    
    if any(np.isnan([logp_H0, logp_H1, logp_threshold, logp_numax, logp_bkgRatio])):
        return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf
    
    return logp_numax, logp_threshold, logp_H1, -logp_H0, logp_bkgRatio

def getTrimmedACF(f, p, b, freqBinSize=1):
    fb, pb, bb, dfb, Nbin = binTheSpectrum(f, p, b, freqBinSize)
    snr = pb/bb
    lags, acf = fftAcf(fb, snr-np.nanmean(snr))
    
    # Set the limit to consider in the ACF. This is 3 times the Dnu corresponding
    # to the Nyquist frequency times a bit. Should be enough to include 
    # everything of interest.
    lmax = 3*dnuScale(fb[-1]*1.2)  
    idx = (0 < lags) & (lags < lmax)
    return lags[idx], acf[idx]

def dnuScale(nu, p = [0.75423185, -0.55338623]):
    return 10**np.polyval(p, np.log10(nu))

def getSNR(nu, fb, pb, bb, Teff, Nyquist):
    
    dfb = fb[1]-fb[0]
    
    height = envHeight(nu, Teff)*0.4 # Note this is multiplied by 0.4! 
    
    width = envWidth(nu, Teff) * np.sqrt(2.*np.log(2.)) / 2 # Note this is divided by 2! 
    
    env = pmodeEnv(fb, nu, height, width)*np.sinc(0.5*fb/Nyquist)**2

    envRange = abs(fb-nu) < (width + dfb)/2  
        
    Ptot = np.sum(pb[envRange]- bb[envRange])
    
    PtotPred = np.sum(env[envRange])           
    
    Btot = np.sum(bb[envRange])      
   
    SNR = Ptot / Btot
    
    SNRPred = PtotPred / Btot
    
    return SNR, SNRPred

def bkgRatioPrior(nu, f, b, Teff, Nbin, falseAlarm=0.1):
    bkgRatio = bkgSlopeRatio(nu, f, b, Teff)
    x = chi2isf(falseAlarm, df=2*Nbin, scale=1/(2*Nbin))
    #x = chi2.isf(falseAlarm, df=2*Nbin, scale=1/(2*Nbin))
    logpBkgRatio = chi2logsf(x*bkgRatio, df=2*Nbin, scale=1/(2*Nbin))
    #logpBkgRatio = chi2.logsf(x*bkgRatio, df=2*Nbin, scale=1/(2*Nbin))
    return logpBkgRatio

def bkgSlopeRatio(nuBack, fb, bb, Teff, nugran0 = 1/250*1e6/(2*np.pi), numax0 = 3050):
    """ Estimate the slope background
    
    Estimates the slope of the background level between at test freqeuency
    nuBack and a higher frequency estimated by the granulation timescale 
    relations. 
    
    """
    scale = numax0/nugran0
    
    idxBack = np.argmin(abs(fb-nuBack))
    nuForward = fb[idxBack]*scale-envWidth(fb[idxBack]*scale, Teff)
    idxForward = np.argmin(abs(fb-nuForward))     
    return bb[idxForward]/bb[idxBack]

def thresholdPrior(SNRPred, Nbin, falseAlarm = 0.01):
    """ Computes the false alarm probability prior
    
    Estimates where we can be reasonably sure that the predicted p-modes will
    be visible, given the observed background. This is then used as a prior.
    
    Parameters
    ----------
    SNRPred : array
        Array of predicted SNR values in the spectrum (Ptot_pred/Btot).
    Nbin : int
        Binning factor used when binning the spectrum.
    
    Returns
    -------
    prior_thresh : array
        The threshold prior at each frequency.
    """
    x = chi2isf(falseAlarm, df=2*Nbin, scale=1/(2*Nbin))
    #x = chi2.isf(false_alarm, df=2*Nbin, scale=1/(2*Nbin))
    SNRThresh = x-1
    logp_thresh = chi2logsf((1+SNRThresh)/(1+SNRPred), df=2*Nbin, scale=1/(2*Nbin)) 
    #logp_thresh = chi2.logsf((1+SNR_thresh)/(1+SNR_pred), df=2*Nbin, scale=1/(2*Nbin)) 
    return logp_thresh

def obsNumaxPrior(nu, fb, numaxGuess):
    """ Prior on numax based on non-seismic observations
    
    To be replaced by scaling relations from Gaia obs?
    
    """
    mu = numaxGuess
    sigma = 0.5*numaxGuess
    #return norm.logpdf(nu, mu, sigma)
    return normlogpdf(nu, mu, sigma, normed=False)


def fftAcf(f, s, skip = 1, axis=-1, **kwargs):
    """ Compute ACF using the FFT method
       
    Compute the autocorrelation function (ACF) of 
    an array using NumPy's fast Fourier transform. 

    Parameters
    ----------
    f : array
        Frequency array.
    s : array
        Spectrum to compute the ACF of.
    skip : int
        Factor to use for downsampling the ACF if 
        the full resolution is not required. If you
        want to oversample the ACF skip should be 
        < 1
    axis : int
        Axis of x to use for computing the FFT and
        iFFT, in case x.ndim > 1.
    kwargs : dict
        Additional keywords to be passed to FFT and 
        iFFT.
    """

    F = np.fft.fft(s, axis=axis)
    
    acf = np.real(np.fft.ifft(F*np.conj(F), n = int(len(F)//skip), axis=axis))
    
    df = np.median(np.diff(f))*skip
    lags = np.arange(len(acf))*df
    
    #acf /= acf[0]
    
    return lags[:int(len(F)//(skip*2))], acf[:int(len(F)//(skip*2))]



def binTheSpectrum(f, p, b, freqBinSize = 10):
    """ Bin the spectrum and the background
    
    The binsize should be set based on which cadence you are using. For short 
    cadence targets you'll probably need 20-30 muHz binning, and 2-5muHz binning
    for long cadence targets. 
    
    Parameters
    ----------
    f : array
        Array of frequency bins in the spectrum.
    p : array
        Array of psd values in the spectrum.
    b : array
        Array of psd values representing the background
    dpb: array
        Array of p-b values representing the background
    freq_bin_size : float
        Bin size in units of frequency to bin by
        
    Returns
    -------
    fb : array
        Binned frequency
    pb : array
        Binned psd
    bb : array
        Binned background
    dpbb : array
        Binned p-b
    dfb : array
        Frequency resolution after binning
    Nbin : array
        Binning factor that was used
    """
    
    df = f[1]-f[0] 
    Nbin = int(freqBinSize / df) 
    if Nbin==0:
        Nbin=1
    fb = binThis(f, Nbin)
    pb = binThis(p, Nbin)
    bb = binThis(b, Nbin)

    dfb = fb[1] - fb[0] 
    return fb, pb, bb, dfb, Nbin

def getBkg(f, p, a=0.66, b=0.88, factor=0.1, skips=10):
    """ Estimate the background
    
    Takes an average of the power at linearly spaced points along the frequency
    axis, where the width of the averaging window increases as a power law. 
    
    A power law with a=0.66 and b=0.88 is approxiately the scaling of Dnu with
    nu_max. This is scaled by a factor which is typically less than 1, i.e.,
    the averaging windows are some fraction of Dnu if the test frequency would
    be nu_max. 
    
    If skips is small (~1) the averaging windows will overlap. This may be
    desirable? It'll be pretty slow for short-cadence targets though. 
    
    Finally the mean power values are interpolated back onto the full frequency
    axis.
    
    Parameters
    ----------
    f : array
        Array of frequency bins in the spectrum.
    p : array
        Array of psd values in the spectrum.
    a : float
        Power law scale
    b : float
        Power law exponent
    factor : float
        Factor by which 
    skips : int
        Number of array indices to skip between each point where the average
        power is evaluated.
        
    Returns
    -------
    b : array
        Array of psd values approximating the background
    """
    
    skips = int(skips)
    
    if skips > len(f) or skips ==0:
        raise ValueError(f'A skip value of {skips} is unacceptable.')
    
    m = [np.median(p[np.abs(f-fi) < a*fi**b*factor]) for fi in f[::skips]]
    
    m = interp1d(f[::skips], m, bounds_error=False)
    
    return m(f)/np.log(2.)

def binThis(x, n):
    """ Bin x by a factor n
    
    This only works for evenly spaced data, since it works by reshaping the
    array.
    
    If len(x) is not equal to an integer number of n, 
    the remaining frequency bins are discarded. 
    Half at low frequency and half at high frequency. 
    
    Parameters
    ----------
    x : array
        Array of values to bin.
    n : int
        Binning factor
    
    Returns
    -------
    xbin : array
        The binned version of the input array
    """
    
    trim = (len(x)//n)*n # The input array isn't always an integer number of the binning factor
    halfRest = (len(x)-trim)//2 
    x = x[halfRest:halfRest+trim] # Trim the input array
    xbin = x.reshape((-1, n)).mean(axis = 1) # reshape and average 
    return xbin


def makePsd(lc, NyquistScale=1):
    """ Compute psd from lightkurve periodogram
    
    Computes the power spectral density from the lightkurve periodogram, since
    Lightkurve doesn't normalize things correctly.
    
    Parameters
    ----------
    lc : lightkurve object
        lightkurve.lightcurve object containing the time series
    NyquistScale : float
        Scaling for the Nyquist frequency. Sometimes it might be useful to 
        compute the spectrum below or beyond the Nyquist frequency. 
        Default is 1.
    
    Returns
    -------
    f : array
        Array of frequency bins in the spectrum.
    p : array
        Array of psd values in the spectrum.
    pg : lightkurve.periodogram
        lightkurve.periodogram object  
    """
    
    
    
    dt = np.median(np.diff(lc.time))*60*60*24/1e6
    
    nyq = 0.5/dt
    
    pg = lc.to_periodogram(normalization = 'psd', maximum_frequency = nyq*NyquistScale)
    
    pg.power = pg.power / np.mean(pg.power) * (2*np.std(lc.flux)**2 * dt) 
    
    f, p = pg.frequency.value, pg.power.value
    
    return f, p, pg

def pmodeEnv(f, numax, H, W):
    """ p-mode envelope as a function of frequency.
    
    The p-mode envelope is assumed to be a Guassian.
    
    Parameters
    ----------
    f : array
        Frequency bins in the spectrum
    numax : float
        Frequency to place the p-mode envelope at. 
    teff : float
        Effective temperature of the star
    
    Returns
    -------
    envelope : array
        Predicted Guassian p-mode envelope.
    """
    envelope = H*np.exp(-(f-numax)**2/(2*W**2))
    
    return envelope

def envHeight(numax, teff, teff0=5777, numax0 = 3090, HenvSun=0.1, 
               teffRed0=8907):
    """ Scaling relation for the envelope height
    
    Parameters
    ----------
    numax : float
        Frequency of maximum power of the p-mode envelope. 
    teff : float
        Effective temperature of the star
    Henv_sun: float, optional
        Envelope height for the Sun. Default is 0.1 ppm^2/muHz.

    Returns
    -------
    Henv : float
        Height of the p-mode envelope
    """
    DT = 1250
    
    Tred = teffRed0*(numax/numax0)**0.11*(teff/teff0)**-0.47
    
    beta = 1 - np.exp(-(Tred-teff)/DT)
    
    Henv = HenvSun*beta**2*(numax/numax0)**-2.79*(teff/teff0)**3 
    
    return Henv

def envWidth(numax, teff, teff0=5777):
    """ Scaling relation for the envelope width
    
    Currently just a crude estimate. This can probably 
    be improved.
    
    Parameters
    ----------
    numax : float
        Frequency of maximum power of the p-mode envelope.
    teff : float
        Effective temperature of the star
    
    Returns
    -------    
    width : float
        Envelope width in muHz
    
    """
    if teff <= 5600:
        width = 0.66*numax**0.88
    else:
        width = 0.66*numax**0.88*(1+(teff-teff0)*6e-4)

    return width

#def set_init_numax(f, numax, N):
#    """ set initial starting points for numax
#    
#    Based on a uniform distribution between +/-50% of input numax.
#    
#    Defaults to frequency limits if numax is <1/T or >Nyquist. 
#    
#    If for some reason the upper limit is less than the lower limit, the default
#    is the entire frequency range. 
#    """
#    
#    if numax <= 0:
#        raise ValueError('numax must be >= 0.')
#    
#    l = max([numax*0.5, f[0]])
#    u = min([numax*1.5, f[-1]])
#    if u <= l:
#        l = f[0]
#        u = f[-1]
#    return np.random.uniform(l, u, size = N)


#def get_lc(ID, cadence, mission, download_dir, flatten=2001, cache_expire = 300):
#    """ Collect lightcurve from MAST
#    
#    Parameters
#    ----------
#    ID : string
#        ID of the target star (KIC, EPIC, TIC)
#    cadence : str
#        Observation cadence to download
#    mission : string
#        Mission source to use (Kepler, K2, TESS)
#    flatten : int
#        Number of cadences to use when flattening the lightcurve.
#        
#    Returns
#    -------
#    lc : lightkurve.lightcurve
#        lightkurve.lightcurve object containing the time series
#    """
#    
#    if flatten%2 == 0:
#        flatten += 1
#    try:
#        cacheQuery = lkAcc.perform_search(ID, cadence=cadence, mission=mission, cache_expire = cache_expire, download_dir = download_dir)
#        filesInCache = lkAcc.check_lc_cache(cacheQuery)
#        lc = lkAcc.load_fits(filesInCache)
#        lc  = lc.remove_nans().flatten(window_length=flatten).remove_outliers(5)
#    except:
#        return None
#        
#    lc.flux = (lc.flux-1)*1e6 # scale to ppm and zero mean.
#    return lc