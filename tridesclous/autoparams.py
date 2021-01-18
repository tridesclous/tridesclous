import time
import copy
from pprint import pprint

import numpy as np

from .cltools import HAVE_PYOPENCL
from .cluster import HAVE_ISOSPLIT5

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


# nb of channel to become sparse
limit_dense_sparse = 4


_default_catalogue_params = {
    'duration': 300.,
    
    'chunksize': 1024,
    'mode': 'dense', # 'sparse'
    'sparse_threshold': None, # 1.5
    'memory_mode': 'memmap',
    'n_spike_for_centroid':350,
    'n_jobs' :-1,

    'preprocessor': {
        'highpass_freq': 300.,
        'lowpass_freq': 5000.,
        'smooth_size': 0,
        
        'pad_width': -1,   # this auto
        'engine': 'numpy',
        'common_ref_removal':False,
    },
    'peak_detector': {
        'method' : 'global',
        'engine': 'numpy',
        'peak_sign': '-',
        'relative_threshold': 5.,
        'peak_span_ms': .7,
        'adjacency_radius_um' : None,
        'smooth_radius_um' : None,
        
    },
    'noise_snippet': {
        'nb_snippet': 300,
    },
    'extract_waveforms': {
        'wf_left_ms': -1.,
        'wf_right_ms': 1.5,
        'wf_left_long_ms': -2.5,
        'wf_right_long_ms': 3.5,
    },
    'clean_peaks': {
        #~ 'alien_value_threshold': None,
        'alien_value_threshold':-1., # equivalent to None (normally)
        #~ 'alien_value_threshold': np.nan,
        'mode': 'extremum_amplitude',
    },
    'peak_sampler': {
        'mode': 'rand',
        'nb_max': 20000,
        #~ 'nb_max_by_channel': None,
        'nb_max_by_channel': 600.,
    },
    #~ 'clean_waveforms': {
        #~ 'alien_value_threshold': None,
    #~ },
    'feature_method': 'pca_by_channel',
    'feature_kargs': {},
    'cluster_method': 'pruningshears',
    'cluster_kargs': {},
    
    
    'clean_cluster' :{
        'apply_auto_split': True,
        'apply_trash_not_aligned': True,
        'apply_auto_merge_cluster': True,
        'apply_trash_low_extremum': True,
        'apply_trash_small_cluster': True,
    },
    
    
    'make_catalogue':{
        'inter_sample_oversampling':False,
        'subsample_ratio': 'auto',
        'sparse_thresh_level2': 1.5,
        'sparse_thresh_level2': 3,
    }
}





def get_auto_params_for_catalogue(dataio=None, chan_grp=0,
                            nb_chan=None, sample_rate=None,
                            context='offline'):
    """
    Automatic selection of parameters.
    This totally empiric paramerters.
    """
    
    assert context in ('offline', 'online')
    
    params = copy.deepcopy(_default_catalogue_params)

    # TODO make this more complicated
    #  * by detecting if dense array or not.
    #  * better method sleection


    if dataio is None:
        # case online no DataIO
        assert nb_chan is not None
        assert sample_rate is not None
        # use less because chunksize is set outsied
        params['chunksize'] = int(sample_rate * 0.05)
        params['duration'] = 10. # debug
    else:
        # case offline with a DataIO
        nb_chan = dataio.nb_channel(chan_grp=chan_grp)
        sample_rate = dataio.sample_rate
        total_duration = sum(dataio.get_segment_length(seg_num=seg_num) / dataio.sample_rate for seg_num in range(dataio.nb_segment))
        # auto chunsize of 100 ms
        params['chunksize'] = int(sample_rate * 1.0)
        params['duration'] = 601.
        
        # segment durartion is not so big then take the whole duration
        # to avoid double preprocessing (catalogue+peeler)
        if params['duration'] * 2 > total_duration:
            params['duration'] = total_duration
    
    #~ if nb_chan == 1:
        #~ params['mode'] = 'dense'
        #~ params['adjacency_radius_um'] = 0.
        #~ params['sparse_threshold'] = 1.5
        
        #~ params['peak_detector']['method'] = 'global'
        #~ params['peak_detector']['engine'] = 'numpy'
        #~ params['peak_detector']['smooth_radius_um' ] = None


        #~ params['peak_sampler']['mode'] = 'rand'
        #~ params['peak_sampler']['nb_max'] = 20000
        
        #~ params['feature_method'] = 'global_pca'
        #~ params['feature_kargs'] = {'n_components' : 4 }
        
        #~ params['cluster_method'] = 'dbscan_with_noise'
        #~ params['cluster_kargs'] = {}
        

        
        #~ params['clean_cluster_kargs'] = {'too_small' : 20 }        
    
    #~ elif nb_chan <=4:
    if nb_chan <= limit_dense_sparse:
    #~ if nb_chan <=8:
    
        params['mode'] = 'dense'
        #~ params['adjacency_radius_um'] = 0.
        params['sparse_threshold'] = 1.5
        
        #~ params['peak_detector']['method'] = 'global'
        #~ params['peak_detector']['engine'] = 'numpy'

        params['peak_detector']['method'] = 'geometrical'
        params['peak_detector']['engine'] = 'numba'
        
        params['peak_detector']['adjacency_radius_um'] = 200. # useless
        params['peak_detector']['smooth_radius_um' ] = None


        params['peak_sampler']['mode'] = 'rand'
        params['peak_sampler']['nb_max'] = 20000
        
        params['feature_method'] = 'global_pca'
        if nb_chan in (1,2):
            n_components = 5
        else:
            n_components = int(nb_chan*2)
        params['feature_kargs'] = {'n_components' : n_components }
        
        
        params['cluster_method'] = 'pruningshears'
        params['cluster_kargs']['max_loop'] = max(1000, nb_chan * 10)
        params['cluster_kargs']['min_cluster_size'] = 20
        params['cluster_kargs']['adjacency_radius_um'] = 0.
        params['cluster_kargs']['high_adjacency_radius_um'] = 0.
        
        # necessary for peeler classic
        #~ params['make_catalogue']['inter_sample_oversampling'] = True


        

    else:
        params['mode'] = 'sparse'
        #~ params['adjacency_radius_um'] = 200.
        params['sparse_threshold'] = 1.5

        if nb_chan >= 32 and HAVE_PYOPENCL:
            params['preprocessor']['engine'] = 'opencl'
        params['peak_detector']['method'] = 'geometrical'
        params['peak_detector']['adjacency_radius_um'] = 200.
        #~ params['peak_detector']['smooth_radius_um' ] = 10
        params['peak_detector']['smooth_radius_um' ] = None
        
        
        if HAVE_PYOPENCL:
            params['peak_detector']['engine'] = 'opencl'
        elif HAVE_NUMBA:
            params['peak_detector']['engine'] = 'numba'
        else:
            print('WARNING : peakdetector will be slow install opencl or numba')
            params['peak_detector']['engine'] = 'numpy'
        
        params['peak_sampler']['mode'] = 'rand_by_channel'
        #~ params['extract_waveforms']['nb_max_by_channel'] = 700
        params['peak_sampler']['nb_max_by_channel'] = 1000
        #~ params['peak_sampler']['nb_max_by_channel'] = 1500
        #~ params['peak_sampler']['nb_max_by_channel'] = 3000
        
        
        params['feature_method'] = 'pca_by_channel'
        # TODO change n_components_by_channel depending on channel density
        #~ params['feature_kargs'] = {'n_components_by_channel':5}
        params['feature_kargs'] = {'n_components_by_channel': 3,
                                                        'adjacency_radius_um' :50.,  # this should be greater than cluster 'adjacency_radius_um'
                                                        }
        
        params['cluster_method'] = 'pruningshears'
        params['cluster_kargs']['max_loop'] = max(1000, nb_chan * 20)
        params['cluster_kargs']['min_cluster_size'] = 20
        params['cluster_kargs']['adjacency_radius_um'] = 50.
        params['cluster_kargs']['high_adjacency_radius_um'] = 30.

    if context == 'online':
        params['n_jobs' ] = 1
    
    return params



def get_auto_params_for_peelers(dataio, chan_grp=0):
    nb_chan = dataio.nb_channel(chan_grp=chan_grp)
    params = {}
    
    params['chunksize'] = int(dataio.sample_rate * 1)
    #~ params['chunksize'] = int(dataio.sample_rate * 0.5)
    #~ params['chunksize'] = int(dataio.sample_rate * 0.1)
    #~ params['chunksize'] = int(dataio.sample_rate * 0.033)

    if nb_chan <= limit_dense_sparse:
        params['engine'] = 'geometrical'
        
    else:
        if HAVE_PYOPENCL:
            params['engine'] = 'geometrical_opencl'
        else:
            params['engine'] = 'geometrical'

    
    # DEBUG force 'geometrical'
    params['engine'] = 'geometrical'
        

    return params
