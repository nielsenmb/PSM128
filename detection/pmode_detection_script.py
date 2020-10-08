from scipy.stats import chi2
from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import LightkurveCacheAccess as lkAcc
import warnings, os
from scipy.interpolate import interp1d


def makePsd(lc):
    """ Compute psd from lightkurve periodogram
    
    Computes the power spectral density from the lightkurve periodogram, since
    Lightkurve doesn't normalize things correctly.
    
    Parameters
    ----------
    lc : lightkurve object
        lightkurve.lightcurve object containing the time series
    
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
    
    nyq = 1/(2*dt)
    
    pg = lc.to_periodogram(normalization = 'psd', maximum_frequency = nyq)
    
    pg.power = pg.power / np.mean(pg.power) * (2*np.std(lc.flux)**2 * dt) 
    
    f, p = pg.frequency.value, pg.power.value
    
    return f, p, pg

def tsToPsd(ID, download_dir=None, mission='Kepler', cadence='long'):
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
    if download_dir is None:
        download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        lc = lkAcc.query_lightkurve(ID, download_dir, True, {'mission': mission, 'cadence': cadence})
        
    lc.flux = (lc.flux-1)*1e6

    f, p, pg = makePsd(lc)

    b = getBkg(pg=pg, filter_width=0.1)  # 3600 is an arbitrarily chosen number
    
    idx = ~np.isnan(b)
    
    return f[idx], p[idx], b[idx]

def binThis(x, n):
    """ Bin x by a factor n
    
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
    if n == 1:
        return x
    
    trim = (len(x)//n)*n # The input array isn't always an integer number of the binning factor
    
    half_rest = (len(x)-trim)//2 
    
    x = x[half_rest:half_rest+trim] # Trim the input array
    
    xbin = x.reshape((-1, n)).mean(axis = 1) # reshape and average 
    
    return xbin

def getBkg(f=None, p=None, pg=None, filter_width=0.075, method = 'LK'):
    """ Get the background estimate
    
    Estimates the background level at each frequency bin. Different methods can
    be used. 
    
    Parameters
    ----------
    f : array
        Array of frequency bins in the spectrum.
    p : array
        Array of psd values in the spectrum.
    pg : lightkurve.periodogram
        lightkurve.periodogram object
    filter_width: float
        Frequency width to use when estimating the background. Default is 0.1
        dex. Units will vary depending on the chosen method.
    method : str
        Method to use for estimating the background. Default is using the 
        LightKurve periodogram, which uses a running median in log(frequency).
        
    Returns
    -------
    b : array
        Array of psd values representing the background
    """
    
    if method=='LK':
        snr, bkg = pg.flatten(filter_width = filter_width, return_trend=True)
        b = bkg.power.value
        
    if method=='dnuSum':    
        skips = 1000
        
        factor = len(f)//1000
        
        a=0.66
        
        b=0.88 
        
        if skips > len(f) or skips ==0:
            raise ValueError(f'A skip value of {skips} is unacceptable.')
        
        m = [np.median(p[np.abs(f-fi) < a*fi**b*factor]) for fi in f[::skips]]
        
        m = interp1d(f[::skips], m, bounds_error=False)
        
        b = m(f)/np.log(2.)
        
    return b

def binEverything(f, p, b, freq_bin_size = 10):
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
    dfb : array
        Frequency resolution after binning
    Nbin : array
        Binning factor that was used
    """
    
    df = f[1]-f[0] 
    
    Nbin = int(freq_bin_size / df) 
    
    fb = binThis(f, Nbin)
    
    pb = binThis(p, Nbin)
    
    bb = binThis(b, Nbin)
    
    dfb = fb[1] - fb[0] 
    
    return fb, pb, bb, dfb, Nbin

def pmode_env(f, numax, teff):
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
    
    Henv = env_height(numax, teff)
    
    stdenv = env_width(numax, teff)/2. * np.sqrt(2.*np.log(2.))
    
    envelope = Henv*np.exp(-(f-numax)**2/(2*stdenv**2))
    
    return envelope

def env_height(numax, teff, Henv_sun=0.1):
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
    
    Tred = 8907.*(numax/3090.)**0.11*(teff/5777.)**-0.47
    
    beta = 1.0 - np.exp(-(Tred-teff)/1250.)
    
    Henv = Henv_sun*beta**2*(numax/3090.)**-2.79*(teff/5777.)**3  
    
    return Henv

def env_width(numax, teff):
    """ Scaling relation for the envelope height
    
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
        width = 0.66*numax**0.88*(1+(teff-5777)*6e-4)

    #width = numax/4
    return width

def compute_summed_powers(fb, pb, bb, dfb, teff):
    """ Compute summed psd, predicted psd and background
    
    Loops through frequency array fb, and computes a predicted Gaussian
    envelope at each frequency as if that frequency was numax. 
    
    Then a range around each frequency is selected corresponding to the 
    estimated envelope width, and the psd, predicted psd and background are
    summed (individually) in this range. 
    
    Parameters
    ----------
    fb : array
        Binned frequency
    pb : array
        Binned psd
    bb : array
        Binned background
    dfb : array
        Frequency resolution after binning
    teff : float
        Effective temperature of the star
    
    Returns
    -------
    Ptot : array
        Sum of the psd at each frequency
    Ptot_pred : array
        Sum of the predicted psd at each frequency
    Btot : array
        Sum of the background at each frequency
    """
    
    Ptot = np.zeros_like(fb)
    Ptot_pred = np.zeros_like(fb)
    Btot = np.zeros_like(fb)
    
    for i in range(len(fb)):     
        Hgau = pmode_env(fb, fb[i], teff)*0.90   
        l = fb[i] - (env_width(fb[i], teff).astype(int) + dfb)/2  # offset equal to 1 frequency bin to avoid issues at array start and end
        u = fb[i] + (env_width(fb[i], teff).astype(int) + dfb)/2
        env_range = (l < fb) & (fb < u)
        
        Ptot[i]      = dfb * np.sum(pb[env_range]- bb[env_range])
        Ptot_pred[i] = dfb * np.sum(Hgau[env_range])           
        Btot[i]      = dfb * np.sum(bb[env_range])                
    
    return Ptot, Ptot_pred, Btot

def compute_likelihoods(SNR, SNR_pred, Nbin):
    """ Compute the detection likelihoods
    
    Based on a chi^2 distribution with 2Nbin d.o.f. 
    
    Also computes the resulting posterior without any priors.
    
    Parameters
    ----------
    SNR : array
        Array of observed SNR values in the spectrum (Ptot/Btot)
    SNR_pred : array
        Array of predicted SNR values in the spectrum (Ptot_pred/Btot)
    Nbin : int
        Binning factor used when binning the spectrum
        
    Returns
    -------
    like_H0 : array
        H0 likelihood at each frequency
    like_H1 : array
        H1 likelihood at each frequency
    post_H1 : array
        H1 posterior at each frequency, with no prior. Normalized by the sum of
        the likelihoods.
    """    
    
    like_H0 = chi2.pdf(1+SNR, df=2*Nbin, scale=1/(2*Nbin))
    
    like_H1 = chi2.pdf((1+SNR_pred)/(1+SNR), df=2*Nbin, scale=1/(2*Nbin)) 
    
    post_H1 = like_H1/(like_H0+like_H1) 
    
    return like_H0, like_H1, post_H1

def compute_threshold_prior(SNR_pred, Nbin, false_alarm = 0.01):
    """ Computes the false alarm probability prior
    
    Estimates where we can be reasonably sure that the predicted p-modes will
    be visible, given the observed background. This is then used as a prior on 
    H1.
    
    Parameters
    ----------
    SNR_pred : array
        Array of predicted SNR values in the spectrum (Ptot_pred/Btot).
    Nbin : int
        Binning factor used when binning the spectrum.
    
    Returns
    -------
    prior_thresh : array
        The threshold prior at each frequency.
    """
    
    x = chi2.isf(0.01, 2*Nbin, scale=1/(2*Nbin))
    SNR_thresh = x-1
    prior_thresh = chi2.sf((1+SNR_thresh)/(1+SNR_pred), 2*Nbin, scale=1/(2*Nbin)) 
    return prior_thresh

def compute_numax_prior(fb, mu, sigma):
    """ Computes the numax prior
    
    Computes a prior on numax. Currently this a somewhat uninformative
    log-normal prior. 
    
    Parameters
    ----------
    fb : array
        Binned frequency
    mu : float
        Best guess for where numax should be. In natural log.
    sigma : float
        Uncertainty on numax. Don't be too optimistic. In natural log.
    
    Returns
    -------
    prior_numax : array
        The numax prior at each frequency    
    """
    
    prior_numax = np.exp(-(np.log(fb)-mu)**2 / (2*sigma**2)) # prior is 1 at numax
    return prior_numax

def compute_H1_posterior(fb, pb, bb, dfb, Nbin, teff, numax):
    """ Compute the H1 posterior at each frequency
    
    
    
    """
 
    
    Ptot, Ptot_pred, Btot = compute_summed_powers(fb, pb, bb, dfb, teff)
    
    SNR = Ptot / Btot
    
    SNR_pred = Ptot_pred / Btot
    
    like_H0, like_H1, post_H1 = compute_likelihoods(SNR, SNR_pred, Nbin)
    
    prior_thresh = compute_threshold_prior(SNR_pred, Nbin)
    
    prior_numax = compute_numax_prior(fb, np.log(numax), 1)
    
    prior_H1 = prior_thresh * prior_numax
    
    prior_H0 = 1 - prior_H1
    
    post_H1_w_tn_prior = prior_H1*like_H1 / (prior_H0*like_H0 + prior_H1*like_H1)
    
    return Ptot, Ptot_pred, SNR, SNR_pred, like_H0, like_H1, post_H1, post_H1_w_tn_prior, prior_H0, prior_H1



N = 10
tgtList = pd.read_csv('the_big_solar_like_oscillator_list.csv').sample(N, random_state=42)

downloadDir = '/home/nielsemb/.lightkurve-cache'
resultsDir = '/home/nielsemb/work/repos/PSM128/detection/Results'

#fig, ax = subplots(2,2, figsize = (16,9))

for i in tgtList.index:
    
    ID = tgtList.loc[i, 'ID']
    
    print(ID)
    numaxGuess = tgtList.loc[i,'numax'] # This is just used for the prior on numax. Should be replaced with photometric numax estimate.
    
    cadence = tgtList.loc[i, 'cadence']        
    
    teff = tgtList.loc[i, 'teff'] 

    f, p, b = tsToPsd(ID, download_dir=downloadDir, mission='Kepler', cadence=cadence)
  
    fb, pb, bb, dfb, Nbin = binEverything(f, p, b, freq_bin_size = 3)

    Ptot, Ptot_pred, SNR, SNR_pred, like_H0, like_H1, post_H1, post_H1_w_tn_prior, prior_H0, prior_H1 = compute_H1_posterior(fb, pb, bb, dfb, Nbin, teff, numaxGuess)

    idx = post_H1_w_tn_prior > 0.68 

    ax[0,0].loglog(fb, pb, label = 'Binned power', color = 'C0')
    ax[0,0].loglog(fb, bb, label = 'Binned background', color = 'C1')
    ax[0,0].loglog(fb[idx], pb[idx], color = 'C2')
    ax[0,0].legend()
    ax[0,0].set_xlim(min(fb), max(fb))
    ax[0,0].set_ylabel(r'PSD [ppm$^2/\mu$Hz]')
    
    snr_ratio = (1+SNR)/(1+SNR_pred)
    ax[0,1].loglog(fb, snr_ratio, label = '(1+SNR)/(1+SNR_pred)')
    ax[0,1].axhline(1, color = 'k', alpha = 0.4)
    ax[0,1].legend()
    ax[0,1].set_xlim(min(fb),max(fb))
    ax[0,1].set_ylim(1e-2, max([max(snr_ratio), 1.1]))
    ax[0,1].set_xlabel(r'Frequency [$\mu$Hz]')
    ax[0,1].set_ylabel(r'Ratio')  

    ax[1,0].plot(fb, post_H1, c= 'k', label = r'H1 posterior without prior')
    ax[1,0].plot(fb, post_H1_w_tn_prior, ls = 'dashed', c= 'k', label = 'H1 posterior with prior')
    ax[1,0].plot(fb, prior_H1, color = 'C0', label = r'$p(H1)$')
    ax[1,0].plot(fb, prior_H0, color = 'C1', label = r'$p(H0)$')
    ax[1,0].set_xscale('log')
    ax[1,0].set_ylabel('Probability')
    ax[1,0].set_xlabel('Frequency [$\mu$Hz]')
    ax[1,0].set_xlim(min(fb),max(fb))
    ax[1,0].legend()
    
    ax[1,1].plot(fb, like_H0, label = '$\mathcal{L}(H0)$')
    ax[1,1].plot(fb, like_H1, label = '$\mathcal{L}(H1)$')
    ax[1,1].legend()
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlabel('Frequency [$\mu$Hz]')
    ax[1,1].set_ylabel('Likelihood')
    ax[1,1].set_xlim(min(fb),max(fb))
       
    fig.savefig(os.path.join(*[resultsDir, f'{ID}_detection_probability.png']))
    
    for a in ax.flatten():
        a.cla()