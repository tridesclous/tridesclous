"""
Some details on parameters for each step.

Preprocessor
---------------------

    * highpass_freq (float): frequency of high pass filter typically 250~500Hz. This remove LFP component in the signal
      Theorically a low value is better (250Hz) but if the signal contain oscillation at high freqyencies 
      (during sleep for insctance) theye must removed so 400Hz should be OK.
    * lowpass_freq (float): low pass frequency (typically 3000~10000Hz) This remove noise in high freuqnecy. This help
       to smooth the spike for peak alignement. This must not exceed niquist frequency (sample_rate/2)
    * smooth_size (int): other possibility to smooth signal. This apply a kernel (more or less triangle) the *smooth_size**
       width in sample. This is like a lowpass filter. If you don't known put 0.
    * common_ref_removal (bool): this substracts sample by sample the median across channels
       When there is a strong noise that appears on all channels (sometimes due to reference) you
       can substract it. This is as if all channels would re referenced numerically to there medians.
    * chunksize (int): the whole processing chain is applied chunk by chunk, this is the chunk size in sample. Typically 1024.
       The smaller size lead to less memory but more CPU comsuption in Peeler. For online, this will be more or less the latency.
    * pad_width (int): size in sample of the margin at the front edge for each chunk to avoid border effect in backward filter.
       In you don't known put None then pad_width will be int(sample_rate/highpass_freq)*3 which is quite robust (<5% error)
       compared to a true offline filtfilt.
    * engine (str): 'numpy' or 'opencl'. There is a double implementation for signal preprocessor : With numpy/scipy
      flavor (and so CPU) or opencl with home made CL kernel (and so use GPU computing). If you have big fat GPU and are able to install
      "opencl driver" (ICD) for your platform the opencl flavor should speedup the peeler because pre processing signal take a quite
      important amoung of time.
    
Peak detector
----------------------

  * peakdetector_engine (str): 'numpy' or 'opencl'.  See signal_preprocessor_engine. Here the speedup is small.
  * peak_sign (str) : sign of the peak ('+' or '-'). The double detection ('+-') is intentionaly NOT implemented is tridesclous
    because it lead to many mistake for users in multi electrode arrays where the same cluster is seen both on negative peak
    and positive rebounce.
  * relative_threshold (str): the threshold without sign with MAD units (robust standard deviation). See :ref:`important_details`.
  * peak_span_ms (float) : this avoid double detection of the same peak in a short span. The units is millisecond.
  
Waveform extraction
--------------------------------

  * wf_left_ms (float); size in ms of the left sweep from the peak index. This number is negative.
  * wf_right_ms (float): size in ms of the right sweep from the peak index. This number is positive.
  * mode (str): 'rand' or 'all' With 'all' all detected peaks are extracted. With 'all' only an randomized subset is taken.
     Note that if you use tridesclous with the script/notebook method you can also choose by yourself which peak are 
     choosen for waveform extraction. This can be usefull to avoid electrical/optical stimlation periods or force peak around
     stimulus periods.


Clean peaks
-------------------

  * alien_value_threshold (float): units=one mad. above this threshold the waveforms is tag as "Alien" and not use for features and clustering
  * mode 'extremum_amplitude' or 'full_waveform': use only the peak value (fast) or the whole waevform (slower)

  * 

    {'name':'extract_waveforms', 'type':'group', 'children' : waveforms_params},
    {'name':'clean_peaks', 'type':'group', 'children' : clean_peaks_params},
    {'name':'peak_sampler', 'type':'group', 'children' : peak_sampler_params},

    {'name': 'alien_value_threshold', 'type': 'float', 'value':100.},
    {'name': 'mode', 'type': 'list', 'limits':[, ]}, # full_waveform
]

Peak sampler
--------------------

This step must be carrefully inspected. It select some peak mongs all peaks to make features and clustering.
This highly depend on : the duration on which the catalogue constructor is done + the number of channel (and so the number
of cells) + the density (firing rate) fo each cluster. Since this can't be known in advance, the user must explore cluster and
extract again while changing theses parameters given dense enough clusters.
This have a strong imptact of the CPU and RAM. So do not choose too big numbers.
But you there are too few spikes, cluster could be not detected.
    
  * mode : 'rand', 'rand_by_channel' or 'all' Startegy to select some peak to then make feature and clustering.
         'rand' take a global number of spike indepently of channels detection
         'rand_by_channel' take a random number of spike per channel
         'all' take all spike
         internally with script 'force' allow to select manually which spike we want. For instance to force
         when there is a stimulus.
  * nb_max: when mode='rand'
  * nb_max_by_channel when mode='rand_by_channel'

Noise snippet extraction
--------------------------------------

  * nb_snippet (int):  the number of noise snippet taken in the signal in between peaks.
  

Features extraction
-------------------------------
 
Several methods possible. See :ref:`important_details`.


  * **global_pca**: good option for tetrode.
  
    * n_components (int): number of components of the pca for all the channel.
    

  * **peak_max** : good options when clustering is **sawchaincut**
  
  * **pca_by_channel**: good option for high channel counts

    * n_components_by_channel (int): number of component for each channel.


Cluster
-----------

Several methods possible. See :ref:`important_details`.

  * **kmeans** : `kmeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ implemented in sklearn
    
    * n_clusters (int): number of cluster
    
  * **onecluster** no clustering. All label set to 0.
  
  * **gmm** `gaussian mixture model <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_ implemented in sklearn 
    
    * n_clusters (int): number of cluster
    * covariance_type (str): 'full', 'tied', 'diag', 'spherical'
    * n_init (int) The number of initializations to perform.
  
  * **agglomerative** `AgglomerativeClustering <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ implemented in sklearn 
  
    * n_clusters: number of cluster
  
  * **dbscan** `DBSCAN <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ implemented in sklearn
     
    * eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.

  * **hdbscan** `HDBSCAN <https://hdbscan.readthedocs.io>`_  density base clustering without the problem of the **eps**

  * **isosplit** `ISOSPLIT5 <https://github.com/flatironinstitute/isosplit5>`_ develop for moutainsort (another sorter)

  * **optics** `OPTICS <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html>`_ implemented in sklearn
     
    * min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
  
  * **sawchaincut** Home made automatic clustering, usefull for dense arrays. Autodetect well isolated cluster
    and put to trash ambiguous things.

  * **pruningshears** Another home made automatic clustering. Internaly use hdbscan. Have better performance than **sawcahincut**
    but it is slower.


"""

from collections import OrderedDict
import numpy as np

# TODO copy form .._default_catalogue_params all default value to be consistent

preprocessor_params = [
    {'name': 'highpass_freq', 'type': 'float', 'value':300., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
    {'name': 'lowpass_freq', 'type': 'float', 'value':5000., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
    {'name': 'smooth_size', 'type': 'int', 'value':0},
    {'name': 'common_ref_removal', 'type': 'bool', 'value':False},
    #~ {'name': 'chunksize', 'type': 'int', 'value':1024, 'decimals':10},
    {'name': 'pad_width', 'type': 'int', 'value':-1, 'decimals':10, 'limits': (-1, np.inf),},
    {'name': 'engine', 'type': 'list', 'value' : 'numpy', 'limits':['numpy', 'opencl']},
]

peak_detector_params = [
    {'name': 'method', 'type': 'list', 'value' : 'global', 'limits':['global', 'geometrical']},
    {'name': 'engine', 'type': 'list', 'value' : 'numpy', 'limits':['numpy', 'opencl', 'numba']},
    {'name': 'peak_sign', 'type': 'list',  'value':'-', 'limits':['-', '+']},
    {'name': 'relative_threshold', 'type': 'float', 'value': 5., 'step': .1,},
    {'name': 'peak_span_ms', 'type': 'float', 'value':0.5, 'step': 0.05, 'suffix': 'ms', 'siPrefix': False},
    {'name':'adjacency_radius_um', 'type': 'float', 'value':200., 'suffix': 'µm', 'siPrefix': False},    
]

waveforms_params = [
    {'name': 'wf_left_ms', 'type': 'float', 'value':-1.0, 'suffix': 'ms', 'step': .1,},
    {'name': 'wf_right_ms', 'type': 'float', 'value': 1.5,  'suffix': 'ms','step': .1,},
    {'name': 'wf_left_long_ms', 'type': 'float', 'value':-2.5, 'suffix': 'ms', 'step': .1,},
    {'name': 'wf_right_long_ms', 'type': 'float', 'value': 3.5,  'suffix': 'ms','step': .1,},
]


clean_peaks_params = [
    {'name': 'alien_value_threshold', 'type': 'float', 'value': np.nan},
    {'name': 'mode', 'type': 'list', 'limits':['extremum_amplitude', 'full_waveform']}, # 
]


peak_sampler_params = [
    {'name': 'mode', 'type': 'list', 'limits':['rand', 'rand_by_channel', 'all']},
    {'name': 'nb_max', 'type': 'int', 'value':20000},
    {'name': 'nb_max_by_channel', 'type': 'int', 'value':600},
]

#~ clean_waveforms_params =[
    #~ {'name': 'alien_value_threshold', 'type': 'float', 'value':100.},
#~ ]



noise_snippet_params = [
    {'name': 'nb_snippet', 'type': 'int', 'value':300},
]




features_params_by_methods = OrderedDict([
    ('global_pca',  [{'name' : 'n_components', 'type' : 'int', 'value' : 5}]),
    ('peak_max',  []),
    ('pca_by_channel',  [{'name' : 'n_components_by_channel', 'type' : 'int', 'value' : 3},
                                     {'name':'adjacency_radius_um', 'type': 'float', 'value':50., 'suffix': 'µm', 'siPrefix': False},
                                    ]),
    #~ ('neighborhood_pca',  [{'name' : 'n_components_by_neighborhood', 'type' : 'int', 'value' : 3}, 
                                        #~ {'name' : 'radius_um', 'type' : 'float', 'value' : 300., 'step':50.}, 
                                        #~ ]),
])


cluster_params_by_methods = OrderedDict([
    ('pruningshears', [{'name' : 'min_cluster_size', 'type' : 'int', 'value' : 20},
                                {'name':'adjacency_radius_um', 'type': 'float', 'value':50., 'suffix': 'µm', 'siPrefix': False},
                                {'name':'high_adjacency_radius_um', 'type': 'float', 'value':30., 'suffix': 'µm', 'siPrefix': False},
                                ]),
    ('kmeans', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5}]),
    ('onecluster', []),
    ('gmm', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5},
                    {'name' : 'covariance_type', 'type' : 'list', 'limits' : ['full']},
                    {'name' : 'n_init', 'type' : 'int', 'value' : 10}]),
    ('agglomerative', [{'name' : 'n_clusters', 'type' : 'int', 'value' : 5}]),
    ('dbscan', [{'name' : 'eps', 'type' : 'float', 'value' : 3},
                        {'name' : 'metric', 'type' : 'list', 'limits' : ['euclidean', 'l1', 'l2']},
                        {'name' : 'algorithm', 'type' : 'list', 'limits' : ['brute', 'auto', 'ball_tree', 'kd_tree', 'brute']},
                    ]),
    ('optics', [{'name' : 'min_samples', 'type' : 'int', 'value' : 5}]),
    ('hdbscan', [{'name' : 'min_cluster_size', 'type' : 'int', 'value' : 20}]),
    ('isosplit5', []),
    ('sawchaincut', [{'name' : 'max_loop', 'type' : 'int', 'value' : 1000},
                                {'name' : 'nb_min', 'type' : 'int', 'value' : 20},
                                {'name' : 'break_nb_remain', 'type' : 'int', 'value' : 30},
                                {'name' : 'kde_bandwith', 'type' : 'float', 'value' : 1., 'step':0.1},
                                {'name' : 'auto_merge_threshold', 'type' : 'float', 'value' : 2., 'step':0.1},
                                {'name':'print_debug', 'type': 'bool', 'value':False},
                            ]),
    
])

clean_cluster_params = [

    {'name':'apply_auto_split', 'type': 'bool', 'value':True},
    
    
    {'name':'apply_trash_not_aligned', 'type': 'bool', 'value':True},
    
    {'name':'apply_auto_merge_cluster', 'type': 'bool', 'value':True},
    
    
    {'name':'apply_trash_low_extremum', 'type': 'bool', 'value':True},
    
    
    {'name':'apply_trash_small_cluster', 'type': 'bool', 'value':True},
    
]


make_catalogue_params = [
    {'name':'inter_sample_oversampling', 'type': 'bool', 'value':False},
    {'name' : 'sparse_thresh_level1', 'type' : 'float', 'value' : 1.5, 'step':0.1},
    {'name' : 'sparse_thresh_level2', 'type' : 'float', 'value' : 3., 'step':0.1},
    
    
]



fullchain_params = [

    # global params
    {'name':'duration', 'type': 'float', 'value':300., 'suffix': 's', 'siPrefix': True},
    {'name': 'chunksize', 'type': 'int', 'value':1024, 'decimals':10},
    {'name' : 'mode', 'type' : 'list', 'limits' : ['dense', 'sparse']},
    {'name':'sparse_threshold', 'type': 'float', 'value':1.5},
    {'name' : 'n_spike_for_centroid', 'type' : 'int', 'value' : 350},
    {'name' : 'n_jobs', 'type' : 'int', 'value' : -1},
    
    
    # params for each steps
    {'name':'preprocessor', 'type':'group', 'children': preprocessor_params},
    {'name':'peak_detector', 'type':'group', 'children': peak_detector_params},
    {'name':'noise_snippet', 'type':'group', 'children': noise_snippet_params},
    {'name':'extract_waveforms', 'type':'group', 'children' : waveforms_params},
    {'name':'clean_peaks', 'type':'group', 'children' : clean_peaks_params},
    {'name':'peak_sampler', 'type':'group', 'children' : peak_sampler_params},
    
    #~ {'name':'clean_cluster', 'type': 'bool', 'value':True},
    {'name':'clean_cluster', 'type':'group', 'children' : clean_cluster_params},
    
    {'name':'make_catalogue', 'type':'group', 'children' : make_catalogue_params},
    
]

metrics_params = [
    {'name': 'spike_waveforms_similarity', 'type': 'list', 'limits' : [ 'cosine_similarity']},
    {'name': 'cluster_similarity', 'type': 'list', 'limits' : [ 'cosine_similarity_with_max']},
    {'name': 'cluster_ratio_similarity', 'type': 'list', 'limits' : [ 'cosine_similarity_with_max']},
    {'name': 'size_max', 'type': 'int', 'value':10000000},
]


_common_peeler_params = [
    #~ {'name':'limit_duration', 'type': 'bool', 'value': False},
    {'name': 'chunksize', 'type': 'int', 'value':1024, 'decimals':10},
    {'name':'duration', 'type': 'float', 'value':60., 'suffix': 's', 'siPrefix': True},
    
    
    
    {'name': 'maximum_jitter_shift', 'type': 'int', 'value':4, 'decimals':10},
    {'name':'save_bad_label', 'type': 'bool', 'value':False},
    
]

#~ _classic_peeler_params = _common_peeler_params + [
    #~ {'name': 'argmin_method', 'type': 'list', 'limits' : [ 'numpy', 'opencl', 'numba',]},
#~ ]

_geometrical_peeler_params = _common_peeler_params + [
    #~ {'name':'adjacency_radius_um', 'type': 'float', 'value':100., 'suffix': 'µm', 'siPrefix': False},
]

_geometrical_opencl_peeler_params = _common_peeler_params + [
    {'name':'adjacency_radius_um', 'type': 'float', 'value':100., 'suffix': 'µm', 'siPrefix': False},
]


peeler_params_by_methods = OrderedDict([
    #~ ('classic', _classic_peeler_params),
    ('geometrical', _geometrical_peeler_params),
    ('geometrical_opencl', _geometrical_opencl_peeler_params),
])



possible_tags = ['', 'so_bad', 'bad', 'not_so_bad','not_so_good','good', 'so_good', 'better_than_dreams']




