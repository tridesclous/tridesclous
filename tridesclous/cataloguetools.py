"""
Some helping function that apply on catalogueconstructor (apply steps, sumamry, ...)

"""
import time
import copy



#TODO debug this when no peak at all
def apply_all_catalogue_steps(catalogueconstructor, params, verbose=True):
    """
    Helper function to call seuquentialy all catalogue steps with dict of params as input.
    Used by offline mainwinwod and OnlineWindow
    
    
    Usage:
      
      catalogueconstructor = CatalogueConstructor(dataio, chan_grp=0)
      
    params = {
        'duration': 300.,
        'preprocessor': {
            'highpass_freq': 300.,
            'lowpass_freq': 5000.,
            'smooth_size': 0,
            'chunksize': 1024,
            'lostfront_chunksize': -1,   # this auto
            'signalpreprocessor_engine': 'numpy',
            'common_ref_removal':False,
        },
        'peak_detector': {
            'peakdetector_engine': 'numpy',
            'peak_sign': '-',
            'relative_threshold': 5.,
            'peak_span_ms': .3,
        },
        'noise_snippet': {
            'nb_snippet': 300,
        },
        'extract_waveforms': {
            'wf_left_ms': -0.2,
            'wf_right_ms': 3.0,
            'mode': 'rand',
            'nb_max': 20000.,
            'align_waveform': False,
        },
        'clean_waveforms': {
            'alien_value_threshold': 100.,
        },
        'feature_method': 'peak_max',
        'feature_kargs': {},
        'cluster_method': 'sawchaincut',
        'cluster_kargs': {'kde_bandwith': 1.},
        'clean_cluster' : False,
        'clean_cluster_kargs' : {},
    }
      
      apply_all_catalogue_steps(catalogueconstructor, params)
    """
    
    cc = catalogueconstructor
    
    p = {}
    p.update(params['preprocessor'])
    p.update(params['peak_detector'])
    cc.set_preprocessor_params(**p)
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

    t1 = time.perf_counter()
    cc.extract_some_waveforms(**params['extract_waveforms'])
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_waveforms', t2-t1)
    
    t1 = time.perf_counter()
    #~ duration = d['duration'] if d['limit_duration'] else None
    #~ d['clean_waveforms']
    cc.clean_waveforms(**params['clean_waveforms'])
    t2 = time.perf_counter()
    if verbose:
        print('clean_waveforms', t2-t1)
    
    #~ t1 = time.perf_counter()
    #~ n_left, n_right = cc.find_good_limits(mad_threshold = 1.1,)
    #~ t2 = time.perf_counter()
    #~ print('find_good_limits', t2-t1)

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
        print('project', t2-t1)
    
    t1 = time.perf_counter()
    cc.find_clusters(method=params['cluster_method'], **params['cluster_kargs'])
    t2 = time.perf_counter()
    if verbose:
        print('find_clusters', t2-t1)
    
    if params['clean_cluster']:
        cc.clean_cluster(**params['clean_cluster_kargs'])


_default_catalogue_params = {
    'duration': 300.,
    'preprocessor': {
        'highpass_freq': 300.,
        'lowpass_freq': 5000.,
        'smooth_size': 0,
        'chunksize': 1024,
        'lostfront_chunksize': -1,   # this auto
        'signalpreprocessor_engine': 'numpy',
        'common_ref_removal':False,
    },
    'peak_detector': {
        'peakdetector_engine': 'numpy',
        'peak_sign': '-',
        'relative_threshold': 5.,
        'peak_span_ms': .3,
    },
    'noise_snippet': {
        'nb_snippet': 300,
    },
    'extract_waveforms': {
        'wf_left_ms': -0.2,
        'wf_right_ms': 3.0,
        'mode': 'rand',
        'nb_max': 20000.,
        'align_waveform': False,
    },
    'clean_waveforms': {
        'alien_value_threshold': 100.,
    },
    'feature_method': 'peak_max',
    'feature_kargs': {},
    'cluster_method': 'sawchaincut',
    'cluster_kargs': {'kde_bandwith': 1.},
    
    'clean_cluster' : False,
    'clean_cluster_kargs' : {},
    
}







def get_auto_params_for_catalogue(dataio, chan_grp=0):
    """
    Automatic selection of parameters.
    This totally empiric paramerters.
    """
    params = copy.deepcopy(_default_catalogue_params)
    
    nb_chan = dataio.nb_channel(chan_grp=chan_grp)
    
    # TODO make this more complicated
    #  * by detecting if dense array or not.
    #  * better method sleection
    
    if nb_chan <=8:
        params['feature_method'] = 'global_pca'
        params['feature_kargs'] = {'n_components' : int(nb_chan*1.5) }
        params['cluster_method'] = 'dbscan'
        params['cluster_kargs'] = {'eps': 2.5}
        params['clean_cluster'] = True
        params['clean_cluster_kargs'] = {'too_small' : 20 }
        
    else:
        params['feature_method'] = 'peak_max'
        params['feature_kargs'] = {}
        params['cluster_method'] = 'sawchaincut'
        params['cluster_kargs'] = {'kde_bandwith': 1.}
    
    return params


