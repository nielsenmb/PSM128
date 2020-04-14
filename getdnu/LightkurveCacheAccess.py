

import numpy as np
import lightkurve as lk
import glob, pickle, warnings, os
from datetime import datetime


def search_and_dump(ID, cadence, mission, search_cache, store_date):
    
    search = lk.search_lightcurvefile(ID, cadence=cadence, mission=mission)

    kplr = 'kplr'+'0'*(9-len(ID.strip('KIC')))+ID.strip('KIC')
    idx = np.zeros(len(search.table['obs_id']), dtype = bool)
    for i in range(len(idx)):
        if kplr in search.table['obs_id'][i]:
            idx[i] = 1
    search.table = search.table[idx]

    fname = os.path.join(*[search_cache, f"{ID}_{mission}_{cadence}_{store_date}.lksearchresult"])
    pickle.dump(search, open(fname, "wb"))
    return search


def perform_search(ID, cadence='long', mission='Kepler', download_dir=None, cache_expire = 30):
    """ Find filenames related to target
    
    Preferentially accesses cached search results, otherwise searches the 
    MAST archive.
    
    Parameters
    ----------
    ID : str
        Target ID (must be KIC, TIC, or ktwo prefixed)
    cadence : str
        Cadence of the observations, 'short' or 'long'
    mission : str
        Mission 
    download_dir : str
        Directory for fits file and search results caches. Default is ~/.lightkurve-cache. 
    cache_expire : int
        Expiration time for the search cache results. Files older than this will be 
        
    Returns
    -------
    search : lightkurve.search.SearchResult
        Search result from MAST. 
    
    """
    current_date = datetime.now().isoformat()
    store_date = current_date[:current_date.index('T')].replace('-','')
    
    # Set default lightkurve cache if nothing else is given
    if download_dir is None:
        download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])
    
    # Make search_cache dir if it doesn't exist
    cachepath = download_dir
    for x in ['SearchResults', mission]:
        cachepath = os.path.join(*[cachepath, x])
        if not os.path.isdir(cachepath):
            os.mkdir(cachepath)

    # Search the search_cache for relevant file
    wildpath = os.path.join(*[cachepath, f"{ID}_{mission}_{cadence}_20*.lksearchresult"])
    files = glob.glob(wildpath)
   
    # Load cached search result if it exists.
    if len(files)==1:
        fdate = files[0].replace('.lksearchresult','').split('_')[-1]
        ddate = datetime.now() - datetime(int(fdate[:4]), int(fdate[4:6]), int(fdate[6:]))
        
        # If file is saved more than cache_expire days ago, a new search is performed
        if ddate.days < cache_expire:
            search = pickle.load(open(files[0], "rb"))
        else:
            search = search_and_dump(ID, cadence, mission, cachepath, store_date)
    elif len(files) == 0:
        search = search_and_dump(ID, cadence, mission, cachepath, store_date)
    else:
        raise ValueError('Too many files found, clean up the cache!')
    
    return search


def load_fits(files):
    """ Read fitsfiles into a Lightkurve object
    
    Parameters
    ----------
    files : list
        List of pathnames to fits files
    
    Returns
    -------
    lc : lightkurve.lightcurve.KeplerLightCurve object
        Lightkurve light curve object containing the concatenated set of 
        quarters.
        
    """
    
    lcs = [lk.lightcurvefile.KeplerLightCurveFile(file) for file in files]
    lccol = lk.collections.LightCurveFileCollection(lcs)
    lc = lccol.PDCSAP_FLUX.stitch().remove_outliers(4)
    return lc


def check_lc_cache(search, download_dir=None):
    """ Query cache directory or download fits files.
    
    Searches the Lightkurve cache directory set by download_dir for fits files
    matching the search query, and returns a list of path names of the fits
    files.
    
    If not cache either doesn't exist or doesn't contain all the files in the
    search, all the fits files will be downloaded again.
    
    Parameters
    ----------
    search : lightkurve.search.SearchResult
        Search result from MAST. 
    download_dir : str
        Top level of the Lightkurve cache directory. default is 
        ~/.lightkurve-cache
        
    Returns
    -------
    files_in_cache : list
        List of path names to the fits files in the cache directory
    
    """
    
    mission = search.table['obs_collection'][0]
    
    # Set main cache directory.
    if download_dir is None:
        download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])
     
    subdir_name = search.table['obs_id'][0]
    
    tgt_dir = os.path.join(*[download_dir, 'mastDownload', mission, subdir_name])

    fnames = search.table['productFilename']
    files_in_cache = []
        
    if os.path.isdir(tgt_dir):
    
        # Check if filenames are in the cache directory and append to output
        for i, fname in enumerate(fnames):
            file = os.path.join(*[tgt_dir, fname])
            if os.path.exists(file):
                files_in_cache.append(file)

        # If list of files in cache doesn't match search results. Download everything in the search.
        if len(files_in_cache) != len(search):
            warnings.warn('Some fits files appear to be missing compared to search results. Downloading everything again.')
            search.download_all(download_dir = download_dir)
            files_in_cache = [os.path.join(*[download_dir, 'mastDownload', mission, subdir_name, fname]) for fname in fnames]
    else:
        search.download_all(download_dir = download_dir)
        files_in_cache = [os.path.join(*[download_dir, 'mastDownload', mission, subdir_name, fname]) for fname in fnames]

    return files_in_cache
