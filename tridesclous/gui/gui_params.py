from collections import OrderedDict


preprocessor_params = [
    {'name': 'highpass_freq', 'type': 'float', 'value':400., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
    {'name': 'lowpass_freq', 'type': 'float', 'value':5000., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
    {'name': 'smooth_size', 'type': 'int', 'value':0},
    {'name': 'common_ref_removal', 'type': 'bool', 'value':False},
    {'name': 'chunksize', 'type': 'int', 'value':1024, 'decilmals':5},
    {'name': 'lostfront_chunksize', 'type': 'int', 'value':128, 'decilmals':0},
    
]

peak_detector_params = [
    {'name': 'peakdetector_engine', 'type': 'list', 'value' : 'numpy', 'values':['numpy', 'opencl']},
    {'name': 'peak_sign', 'type': 'list', 'values':['-', '+']},
    {'name': 'relative_threshold', 'type': 'float', 'value': 6., 'step': .1,},
    {'name': 'peak_span', 'type': 'float', 'value':0.0002, 'step': 0.0001, 'suffix': 's', 'siPrefix': True},
]

waveforms_params = [
    {'name': 'n_left', 'type': 'int', 'value':-20},
    {'name': 'n_right', 'type': 'int', 'value':30},
    {'name': 'mode', 'type': 'list', 'values':['rand', 'all']},
    {'name': 'nb_max', 'type': 'int', 'value':20000},
    {'name': 'align_waveform', 'type': 'bool', 'value':True},
    #~ {'name': 'subsample_ratio', 'type': 'int', 'value':20},
]





features_params_by_methods = OrderedDict([
    ('pca',  [{'name' : 'n_components', 'type' : 'int', 'value' : 5}]),
    ('peak_max',  []),
    ('pca_by_channel',  [{'name' : 'n_components_by_channel', 'type' : 'int', 'value' : 3}]),
    ('peakmax_and_pca',  [{'name' : 'n_components', 'type' : 'int', 'value' : 5}]),
    ('tsne',  []),
])


cluster_params_by_methods = OrderedDict([
    ('kmeans', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5}]),
    ('gmm', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5},
                    {'name' : 'covariance_type', 'type' : 'list', 'values' : ['full']},
                    {'name' : 'n_init', 'type' : 'int', 'value' : 10}]),
    ('dbscan', [{'name' : 'eps', 'type' : 'float', 'value' : 0.5}]),
    ('dirtycut', []),
])


fullchain_params = [
    {'name':'duration', 'type': 'float', 'value':300., 'suffix': 's', 'siPrefix': True},
    {'name':'preprocessor', 'type':'group', 'children': preprocessor_params},
    {'name':'peak_detector', 'type':'group', 'children': peak_detector_params},
    {'name':'extract_waveforms', 'type':'group', 'children' : waveforms_params},
    #~ {'name':'features', 'type':'group', 
        #~ 'children':[{'name': 'method', 'type': 'list', 'values': features_params_by_methods.keys()}],
    #~ },
    #~ {'name':'find_cluster', 'type':'group', 
        #~ 'children':[{'name': 'method', 'type': 'list', 'values':cluster_params_by_methods.keys()}],
    #~ },
]


peeler_params = [
    {'name':'limit_duration', 'type': 'bool', 'value':True},
    {'name':'duration', 'type': 'float', 'value':60., 'suffix': 's', 'siPrefix': True},
    {'name': 'n_peel_level', 'type': 'int', 'value':2},
]
