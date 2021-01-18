"""
Some helping function that apply on catalogueconstructor (apply steps, sumamry, ...)

"""
import time
import copy
from pprint import pprint

from .cltools import HAVE_PYOPENCL
from .cluster import HAVE_ISOSPLIT5

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    

#TODO debug this when no peak at all
def apply_all_catalogue_steps(catalogueconstructor, params, verbose=True):
    """
    Helper function to call seuquentialy all catalogue steps with dict of params as input.
    Used by offline mainwinwod and OnlineWindow.
    Also used by spikeinterface.sorters.
    
    
    Usage:
      
      cc = CatalogueConstructor(dataio, chan_grp=0)
      params = get_auto_params_for_catalogue(dataio, chan_grp=0)
      apply_all_catalogue_steps(cc, params)
    """
    #~ if verbose:
        #~ print('apply all catalogue steps')
        #~ pprint(params)
    
    #~ pprint(params)
    
    cc = catalogueconstructor
    
    # global params
    d = {k:params[k] for k in ('chunksize', 'sparse_threshold', 'mode', 'memory_mode', 'n_jobs', 'n_spike_for_centroid', ) if k in params}
    
    cc.set_global_params(**d)
    
    # params preprocessor
    d = dict(params['preprocessor'])
    cc.set_preprocessor_params(**d)
    
    # params peak detector
    d = dict(params['peak_detector'])
    cc.set_peak_detector_params(**d)
    
    dataio = cc.dataio
    
    #TODO offer noise esatimation duration somewhere
    noise_duration = min(10., params['duration'], dataio.get_segment_length(seg_num=0)/dataio.sample_rate*.99)
    #~ print('noise_duration', noise_duration)
    t1 = time.perf_counter()
    cc.estimate_signals_noise(seg_num=0, duration=noise_duration)
    t2 = time.perf_counter()
    if verbose:
        print('estimate_signals_noise', t2-t1)
    
    t1 = time.perf_counter()
    cc.run_signalprocessor(duration=params['duration'])
    t2 = time.perf_counter()
    if verbose:
        print('run_signalprocessor', t2-t1)
    
    #~ t1 = time.perf_counter()
    #~ cc.extract_some_waveforms(**params['extract_waveforms'], recompute_all_centroid=False)
    #~ t2 = time.perf_counter()
    #~ if verbose:
        #~ print('extract_some_waveforms', t2-t1)
    
    #~ t1 = time.perf_counter()
    #~ cc.clean_waveforms(**params['clean_waveforms'], recompute_all_centroid=False)
    #~ t2 = time.perf_counter()
    #~ if verbose:
        #~ print('clean_waveforms', t2-t1)
    
    #~ t1 = time.perf_counter()
    #~ n_left, n_right = cc.find_good_limits(mad_threshold = 1.1,)
    #~ t2 = time.perf_counter()
    #~ print('find_good_limits', t2-t1)

    cc.set_waveform_extractor_params(**params['extract_waveforms'])
    
    t1 = time.perf_counter()
    cc.clean_peaks(**params['clean_peaks'])
    t2 = time.perf_counter()
    if verbose:
        print('clean_peaks', t2-t1)
    
    t1 = time.perf_counter()
    cc.sample_some_peaks(**params['peak_sampler'])
    t2 = time.perf_counter()
    if verbose:
        print('sample_some_peaks', t2-t1)
    
    t1 = time.perf_counter()
    cc.extract_some_noise(**params['noise_snippet'])
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_noise', t2-t1)
    
    #~ print(cc)
    
    t1 = time.perf_counter()
    cc.extract_some_features(method=params['feature_method'], **params['feature_kargs'])
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_features', t2-t1)
    
    #~ print(cc)
    
    t1 = time.perf_counter()
    cc.find_clusters(method=params['cluster_method'], recompute_centroid=False, order=False, **params['cluster_kargs'])
    t2 = time.perf_counter()
    if verbose:
        print('find_clusters', t2-t1)
    
    t1 = time.perf_counter()
    cc.cache_some_waveforms()
    t2 = time.perf_counter()
    if verbose:
        print('cache_some_waveforms', t2-t1)
    
    t1 = time.perf_counter()
    cc.compute_all_centroid()
    t2 = time.perf_counter()
    if verbose:
        print('compute_all_centroid', t2-t1)

    #~ cc.order_clusters(by='waveforms_rms')
    #~ t2 = time.perf_counter()
    #~ if verbose:
        #~ print('order_clusters', t2-t1)

    # clean cluster steps
    pclean = params['clean_cluster']
    
    if pclean['apply_auto_split']:
        t1 = time.perf_counter()
        cc.auto_split_cluster()
        if verbose:
            t2 = time.perf_counter()
            print('auto_split_cluster', t2-t1)
    
    if pclean['apply_trash_not_aligned']:
        t1 = time.perf_counter()
        cc.trash_not_aligned()
        if verbose:
            t2 = time.perf_counter()
            print('trash_not_aligned', t2-t1)
    
    if pclean['apply_auto_merge_cluster']:
        t1 = time.perf_counter()
        cc.auto_merge_cluster()
        if verbose:
            t2 = time.perf_counter()
            print('auto_merge_cluster', t2-t1)
    
    if pclean['apply_trash_low_extremum']:
        t1 = time.perf_counter()
        cc.trash_low_extremum()
        if verbose:
            t2 = time.perf_counter()
            print('trash_low_extremum', t2-t1)

    if pclean['apply_trash_small_cluster']:
        t1 = time.perf_counter()
        cc.trash_small_cluster()
        if verbose:
            t2 = time.perf_counter()
            print('trash_small_cluster', t2-t1)
    
    cc.order_clusters(by='waveforms_rms')
    
    t1 = time.perf_counter()
    cc.make_catalogue_for_peeler(**params['make_catalogue'])
    t2 = time.perf_counter()
    if verbose:
        print('make_catalogue_for_peeler', t2-t1)
    

