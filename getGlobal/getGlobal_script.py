from matplotlib.pyplot import *
import numpy as np
import corner, logging
import globalDetFunctions as gd
import pandas as pd


downloadDir = '/home/nielsemb/work/Bluebear_data/data'

N = 10

df = pd.read_csv('/home/nielsemb/work/the_big_solar_like_oscillator_list.csv').sample(N, random_state=42).reset_index()

df['numax_p16'] = np.nan 
df['numax_p50'] = np.nan
df['numax_p84'] = np.nan
df['dnu_p16'] = np.nan
df['dnu_p50'] = np.nan
df['dnu_p84'] = np.nan

nwalkers, ndim = 50, 5
bsteps = 500
nsteps = 200

figT, axT = subplots(3,2, figsize = (16,7))

for k in df.index:
    print(df.loc[k, 'ID'])
#    try:
    
    ID = df.loc[k, 'ID']
    Teff = df.loc[k,'teff']
    numaxGuess = df.loc[k,'numax']
    cadence = df.loc[k, 'cadence']

    print('Getting psd')
    f, p, b = gd.tsToPsd(ID, downloadDir, mission='Kepler', cadence=cadence)
    
    print('Binning the spectrum')
    fb, pb, bb, dfb, Nbin = gd.binTheSpectrum(f, p, b, freqBinSize = 1)
    
    # Trim the ACF to be at most a few times the maximum possible dnu given nyquist frequency as numax
    print('Computing ACF')
    lags, acf = gd.getTrimmedACF(f, p, b, freqBinSize=0.1) 
    C = np.std(acf[-5:])
    
    print('Running the sampler')
    samples = gd.runSampler(fb, pb, bb, Nbin, f[-1], lags, acf, C, Teff, numaxGuess, gd.logProbEmcee, ndim, progress=True)

    fsamples = samples.reshape((-1, ndim))
    
    percs = np.percentile(fsamples, [16, 50, 84], axis = 0)
    df.at[k,'numax_p16'] = percs[0,0]
    df.at[k,'numax_p50'] = percs[1,0]
    df.at[k,'numax_p84'] = percs[2,0]
    df.at[k,'dnu_p16'] = percs[0,1]
    df.at[k,'dnu_p50'] = percs[1,1]
    df.at[k,'dnu_p84'] = percs[2,1]
    
    df.to_csv('Results/measured_numax.csv', index=False)
      
    logging.disable(logging.WARNING)
    figC = corner.corner(fsamples, truths = [numaxGuess, df.loc[k,'dnu'], None, None, None], labels = ['numax', 'dnu', 'H', 'w', 'B']);
    logging.getLogger().setLevel(logging.WARNING)
    figC.savefig(f'Results/{df.loc[k,"ID"]}_corn.png')            
    for ax in figC.get_axes():
        ax.clear()
    close(figC)    
    
    axT[0,0].plot(fb, pb/bb)
    axT[0,0].set_xlim(min(fb), max(fb))
    axT[0,0].set_ylim(min(pb/bb), max(pb/bb))
    axT[0,0].axvline(df.loc[k,'numax'], color = 'C4')
    axT[0,0].fill_betweenx([0, max(pb/bb)], percs[0,0], percs[2,0], color = 'C3', alpha = 0.2)
        
    axT[0,1].plot(lags, acf)
    for m in np.random.randint(0, len(fsamples[0]), 50):
        _dnu, _H, _w, _B = fsamples[m,1:]
        _w = 10**_w
        axT[0,1].plot(lags, gd.lorentz(lags, acf, _dnu, _H, _w, _B), color = 'C3', alpha = 0.1)
    #axT[0,1].set_ylim(min(acf), max(acf))
    axT[0,1].axvline(df.loc[k,'dnu'], color = 'C4')
    axT[0,1].set_xlim(min(lags), min([10*df.loc[k,'dnu'], max(lags)]))
    axT[0,1].fill_betweenx([min(acf), max(acf)], percs[0,1], percs[2,1], color = 'C3', alpha = 0.1)

    axT[2,1].plot(range(0,nsteps), samples[:, :, 1], "k", alpha=0.3)
    axT[2,1].set_xlim(0, len(samples))
    axT[2,1].set_ylabel("dnu")   
    axT[2,1].set_xlabel("step number");   
    axT[2,1].axhline(df.loc[k,'dnu'], color = 'C4')
    
    for nu in fb:
        lnpn = gd.numaxLogProbability(nu, fb, pb, bb, Teff, dfb, Nbin, fb[-1], numaxGuess) 
        axT[1,0].plot(nu, np.sum(lnpn),'.', color = 'k', ms = 2)
        
    axT[1,0].set_xlim(0,fb[-1])
    
    axT[2,0].plot(range(0,nsteps), samples[:, :, 0], "k", alpha=0.3)
    axT[2,0].set_xlim(0, len(samples))
    axT[2,0].set_ylabel("numax")   
    axT[2,0].set_xlabel("step number");   
    axT[2,0].axhline(df.loc[k,'numax'], color = 'C4')
    
    
    figT.savefig(f'Results/{df.loc[k,"ID"]}_res.png')

    for ax in figT.get_axes():
        ax.clear()
                
#    except Exception as ex:
#             message = "Star {0} produced an exception of type {1} occurred. Arguments:\n{2!r}".format(df.loc[k,'ID'], type(ex).__name__, ex.args)
#             print(message)

close(figT)      
