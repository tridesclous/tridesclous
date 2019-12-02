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
      
      catalogueconstructor = CatalogueConstructor(dataio, chan_grp=0)
      
    params = {
        'duration': 300.,
        'preprocessor': {
            'highpass_freq': 300.,
            'lowpass_freq': 5000.,
            'smooth_size': 0,
            'chunksize': 1024,
            'lostfront_chunksize': -1,   # this auto
            'engine': 'numpy',
            'common_ref_removal':False,
        },
        'peak_detector': {
            'method' : 'global',
            'engine': 'numpy',
            'peak_sign': '-',
            'relative_threshold': 5.,
            'peak_span_ms': .3,
        },
        'noise_snippet': {
            'nb_snippet': 300,
        },
        'extract_waveforms': {
            'wf_left_ms': -1.5,
            'wf_right_ms': 2.5,
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
    if verbose:
        print('apply all catalogue steps')
        pprint(params)
    
    cc = catalogueconstructor
    
    # global params
    d = {k:params[k] for k in ('chunksize', 'mode', 'adjacency_radius_um')}
    
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

    t1 = time.perf_counter()
    cc.extract_some_waveforms(**params['extract_waveforms'], recompute_all_centroid=False)
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_waveforms', t2-t1)
    
    t1 = time.perf_counter()
    #~ duration = d['duration'] if d['limit_duration'] else None
    #~ d['clean_waveforms']
    cc.clean_waveforms(**params['clean_waveforms'], recompute_all_centroid=False)
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
        t1 = time.perf_counter()
        cc.clean_cluster(**params['clean_cluster_kargs'])
        t2 = time.perf_counter()
        if verbose:
            print('clean_cluster', t2-t1)
    
    cc.order_clusters(by='waveforms_rms')


_default_catalogue_params = {
    'duration': 300.,
    
    'chunksize': 1024,
    'mode': 'dense', # 'sparse'
    'adjacency_radius_um': None, # None when sparse
    'sparse_threshold': None, # 1.5
    
    'preprocessor': {
        'highpass_freq': 300.,
        'lowpass_freq': 5000.,
        'smooth_size': 0,
        
        'lostfront_chunksize': -1,   # this auto
        'engine': 'numpy',
        'common_ref_removal':False,
    },
    'peak_detector': {
        'method' : 'global',
        'engine': 'numpy',
        'peak_sign': '-',
        'relative_threshold': 5.,
        'peak_span_ms': .7,
        #~ 'adjacency_radius_um' : None,
    },
    'noise_snippet': {
        'nb_snippet': 300,
    },
    'extract_waveforms': {
        'wf_left_ms': -1.5,
        'wf_right_ms': 2.5,
        'mode': 'rand',
        'nb_max': 20000,
    },
    'clean_waveforms': {
        'alien_value_threshold': None,
    },
    'feature_method': 'peak_max',
    'feature_kargs': {},
    'cluster_method': 'sawchaincut',
    'cluster_kargs': {},
    
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

    # auto chunsize of 100 ms
    params['chunksize'] = int(dataio.sample_rate * 0.1)
    
    
    #~ if nb_chan <=8:
    #~ if nb_chan <=1:
    if nb_chan <=4:
    
        
        params['mode'] = 'dense'
        params['adjacency_radius_um'] = 0.
        params['sparse_threshold'] = 1.5
        
        params['peak_detector']['method'] = 'global'
        params['peak_detector']['engine'] = 'numpy'
        
        params['feature_method'] = 'global_pca'
        
        if nb_chan in (1,2):
            n_components = 3
        elif nb_chan in (3, 4):
            n_components = 5
        else:
            n_components = int(nb_chan)
        
        params['feature_kargs'] = {'n_components' : n_components }
        #~ if HAVE_ISOSPLIT5:
            #~ params['cluster_method'] = 'isosplit5'
            #~ params['cluster_kargs'] = {}
        #~ else:
            #~ params['cluster_method'] = 'dbscan'
            #~ params['cluster_kargs'] = {'eps': 3,  'metric':'euclidean', 'algorithm':'brute'}

        params['cluster_method'] = 'hdbscan'
        params['cluster_kargs'] = {'min_cluster_size': 20}
        

        params['clean_cluster'] = True
        params['clean_cluster_kargs'] = {'too_small' : 20 }
        

    else:
        params['mode'] = 'sparse'
        params['adjacency_radius_um'] = 200.
        params['sparse_threshold'] = 1.5

        if nb_chan > 32 and HAVE_PYOPENCL:
            params['preprocessor']['engine'] = 'opencl'

        params['peak_detector']['method'] = 'geometrical'
        
        #~ numba
        if HAVE_PYOPENCL:
            params['peak_detector']['engine'] = 'opencl'
        elif HAVE_NUMBA:
            params['peak_detector']['engine'] = 'numba'
        else:
            print('WARNING : peakdetector will be slow install opencl')
            params['peak_detector']['engine'] = 'numpy'
        
        params['extract_waveforms']['nb_max'] = max(20000, nb_chan * 300)
        
        
        #~ params['feature_method'] = 'peak_max'
        #~ params['feature_kargs'] = {}
        #~ params['cluster_method'] = 'sawchaincut'
        #~ params['cluster_kargs']['kde_bandwith'] = 1.
        #~ if nb_chan<32:
            #~ params['cluster_kargs']['max_loop'] = 10000
        #~ elif nb_chan>=32:
            #~ params['cluster_kargs']['max_loop'] = nb_chan * 400

        params['feature_method'] = 'pca_by_channel'
        params['feature_kargs'] = {'n_components_by_channel':3}
        params['cluster_method'] = 'pruningshears'
        params['cluster_kargs']['max_loop'] = max(1000, nb_chan * 10)
        params['cluster_kargs']['min_cluster_size'] = 20

        
        #~ else:
            #~ # default one already
            #~ params['cluster_kargs']['max_loop'] = 1000
        
    
    
    return params


