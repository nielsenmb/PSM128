from LightkurveCacheAccess import perform_search, check_lc_cache, load_fits
from get_dnu import get_dnu
import pandas as pd
from astropy.io import ascii
import numpy as np


cache_dir = '/rds/projects/b/ballwh-tess-yield/data'

pdata = pd.read_csv('/rds/homes/n/nielsemb/repos/PSM128/getdnu/data/prior_data.csv')

#yu1 = ascii.read('/rds/homes/n/nielsemb/repos/PSM128/getdnu/data/table1.dat', format='cds', readme="ReadMe")
#yu2 = ascii.read('/rds/homes/n/nielsemb/repos/PSM128/getdnu/data/table2.dat', format='cds',readme="ReadMe")

df = pd.read_csv('/rds/homes/n/nielsemb/repos/PSM128/getdnu/yu_comparison.csv')

kics = ['KIC'+str(x) for x in df['KIC']]
numaxs = [(x, y) for x,y in df[['numax', 'e_numax']].values]
teffs = [(x, y) for x,y in df[['Teff', 'e_Teff']].values]

N = len(df)

for i in range(N):
    if np.isnan(df.loc[i,'dnu_acf']):
        #try:
        search = perform_search(kics[i], download_dir = cache_dir)
        files = check_lc_cache(search, download_dir = cache_dir)
        lc = load_fits(files)
        pg = lc.to_periodogram(normalization='psd').flatten()
        f, s = pg.frequency.value, pg.power.value
        df.at[i,'dnu_acf'] = get_dnu(numaxs[i], teffs[i], f, s, pdata, KDEsize = 500)
        #except:
        #    continue
    if i%100:
        df.to_csv('/rds/homes/n/nielsemb/repos/PSM128/getdnu/yu_comparison.csv', index=False)

df.to_csv('/rds/homes/n/nielsemb/repos/PSM128/getdnu/yu_comparison.csv', index=False)