from LightkurveCacheAccess import perform_search, check_lc_cache, load_fits
from get_dnu import get_dnu
import pandas as pd
from astropy.io import ascii
import numpy as np

cache_dir = '/home/nielsemb/work/Bluebear_data/data/'

pdata = pd.read_csv('prior_data.csv')

yu1 = ascii.read('table1.dat', format='cds', readme="ReadMe")
yu2 = ascii.read('table2.dat', format='cds',readme="ReadMe")

kics = ['KIC'+str(x) for x in yu1['KIC']]
numaxs = [(x, y) for x,y in yu1[['numax', 'e_numax']]]
teffs = [(x, y) for x,y in yu2[['Teff', 'e_Teff']]]

dnus = np.zeros(len(kics))-np.inf

N = 1 #len(dnus)

for i in range(N):
    search = perform_search(kics[i], download_dir = cache_dir)
    files = check_lc_cache(search, download_dir = cache_dir)
    lc = load_fits(files)
    pg = lc.to_periodogram(normalization='psd').flatten()
    f, s = pg.frequency.value, pg.power.value
    try:
        dnus[i] = get_dnu(numaxs[i], teffs[i], f, s, pdata, KDEsize = 500)
    except:
        continue

pd.DataFrame({'kics': kics, 'dnus': dnus}).to_csv('dnus.csv')