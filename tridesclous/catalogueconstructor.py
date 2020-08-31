"""

.. autoclass:: CatalogueConstructor
   :members:


"""


import os
import json
from collections import OrderedDict
import time
import pickle
import itertools
import datetime
import shutil
from pprint import pprint

import numpy as np
import scipy.signal
import scipy.interpolate
import seaborn as sns
sns.set_style("white")

import sklearn
import sklearn.metrics

from . import signalpreprocessor
from . import  peakdetector
from . import decomposition
from . import cluster 
from . import metrics

from .tools import median_mad, get_pairs_over_threshold, int32_to_rgba, rgba_to_int32, make_color_dict
from . import cleancluster


from .iotools import ArrayCollection

import matplotlib.pyplot as plt

from . import labelcodes

from .cataloguetools import apply_all_catalogue_steps

from .waveformtools import get_normalized_centroids, compute_sparse_mask, compute_shared_channel_mask


_global_params_attr = ('chunksize', 'memory_mode', 'internal_dtype', 'mode', 'sparse_threshold', 'n_spike_for_centroid', 'n_jobs') # 'adjacency_radius_um'

_persistent_metrics = ('spike_waveforms_similarity', 'cluster_similarity',
                        'cluster_ratio_similarity', 'spike_silhouette')

_centroids_arrays = ('centroids_median', 'centroids_median_long', 'centroids_mad', 'centroids_mean', 'centroids_std', 'centroids_sparse_mask')
# centroids_sparse_mask is for plotting only


_reset_after_peak_sampler = ('some_features', 'channel_to_features', 'some_noise_snippet',
                'some_noise_index', 'some_noise_features',) + _persistent_metrics + _centroids_arrays

#~ _reset_after_peak_arrays = ('some_peaks_index', 'some_waveforms', 'some_features',
                        #~ 'channel_to_features', 
                        #~ 'some_noise_index', 'some_noise_snippet', 'some_noise_features',
                        #~ ) + _persistent_metrics + _centroids_arrays

# 'some_waveforms', 'some_waveforms_sparse_mask'
_reset_after_peak_arrays = ('some_peaks_index', ) + _reset_after_peak_sampler



_persitent_arrays = ('all_peaks', 'signals_medians','signals_mads', 'clusters') + \
                ('some_waveforms', 'some_waveform_index') + \
                _reset_after_peak_arrays


_dtype_peak = [('index', 'int64'), ('cluster_label', 'int64'), ('channel', 'int64'),  ('segment', 'int64'), ('extremum_amplitude', 'float64'),]


_dtype_cluster = [('cluster_label', 'int64'), ('cell_label', 'int64'), 
            ('extremum_channel', 'int64'), ('extremum_amplitude', 'float64'),
            ('waveform_rms', 'float64'), ('nb_peak', 'int64'), 
            ('tag', 'U16'), ('annotations', 'U32'), ('color', 'uint32')]

_keep_cluster_attr_on_new = ['cell_label', 'tag','annotations', 'color']


_default_n_spike_for_centroid = 350



class CatalogueConstructor:
    __doc__ = """
    
    The goal of CatalogueConstructor is to construct a catalogue of template (centroids)
    for the Peeler.
    
    For so the CatalogueConstructor will:
      * preprocess a duration of data
      * detect peaks
      * extract some waveform
      * compute some feature from these waveforms
      * find cluster
      * compute some metrics ontheses clusters
      * enable some manual trash/merge/split
    
    At the end we can **make_catalogue_for_peeler**, this construct everything usefull for
    the Peeler : centroids (median for each cluster: the centroids, first and secnd derivative,
    median and mad of noise, ...
    
    You need to have one CatalogueConstructor for each channel_group.

    Since all operation are more or less slow each internal arrays is by default persistent
    on the disk (memory_mode='memmap'). With this when you instanciate a CatalogueConstructor
    you load all existing arrays already computed. For that tridesclous mainly use the numpy
    structured arrays (array of struct) with memmap. So, hacking internal arrays of the
    CatalogueConstructor should be easy outside python and tridesclous.

    You can explore/save/reload internal state by borwsing this directory:
    *path_of_dataset/channel_group_XXX/catalogue_constructor*
    
    
    
    **Usage**::
    
        from tridesclous  import Dataio, CatalogueConstructor
        dirname = '/path/to/dataset'
        dataio = DataIO(dirname=dirname)
        cc = CatalogueConstructor(dataio, chan_grp=0)
        
        # preprocessing
        cc.set_preprocessor_params(chunksize=1024,
            highpass_freq=300.,
            lowpass_freq=5000.,
            common_ref_removal=True,
            lostfront_chunksize=64,
            peak_sign='-', 
            relative_threshold=5.5,
            peak_span_ms=0.3,
            )
        cc.estimate_signals_noise(seg_num=0, duration=10.)
        cc.run_signalprocessor(duration=60.)
        
        # waveform/feature/cluster
        cc.extract_some_waveforms(n_left=-25, n_right=40, mode='rand')
        cc.clean_waveforms(alien_value_threshold=55.)
        cc.project(method='global_pca', n_components=7)
        cc.find_clusters(method='kmeans', n_clusters=7)
        
        # manual stuff
        cc.trash_small_cluster(n=5)
        cc.order_clusters()
        
        # do this before peeler
        cc.make_catalogue_for_peeler()
    
    
    For more example see examples.
    
    
    
    **Persitent atributes**:
    
    CatalogueConstructor have a mechanism to store/load some attributes in the
    dirname of dataio. This is usefull to continue previous woprk on a catalogue.
    
    The attributes are almost all numpy arrays and stored using the numpy.memmap
    mechanism. So prefer local fast fast storage like ssd.
    
    Here the list of theses attributes with shape and dtype. **N** is the total 
    number of peak detected. **M** is the number of selected peak for
    waveform/feature/cluser. **C** is the number of clusters
      * all_peaks (N, ) dtype = {0}
      * signals_medians (nb_sample, nb_channel, ) float32
      * signals_mads (nb_sample, nb_channel, ) float32
      * clusters (c, ) dtype= {1}
      * some_peaks_index (M) int64
      * some_waveforms (M, width, nb_channel) float32
      * some_features (M, nb_feature) float32
      * channel_to_features (nb_chan, nb_component) bool
      * some_noise_snippet (nb_noise, width, nb_channel) float32
      * some_noise_index (nb_noise, ) int64
      * some_noise_features (nb_noise, nb_feature) float32
      * centroids_median (C, width, nb_channel) float32
      * centroids_mad (C, width, nb_channel) float32
      * centroids_mean (C, width, nb_channel) float32
      * centroids_std (C, width, nb_channel) float32
      * spike_waveforms_similarity (M, M) float32
      * cluster_similarity (C, C) float32
      * cluster_ratio_similarity (C, C) float32

    """.format(_dtype_peak, _dtype_cluster)
    def __init__(self, dataio, chan_grp=None, name='catalogue_constructor'):
        """
        Parameters
        ----------
        dataio: tirdesclous.DataIO
            the dataio object for describe the dataset
        chan_grp: int
            The channel group key . See PRB file. None by default = take the first
            channel group (often 0 but sometimes 1!!)
        name: str a name
            The name of the CatalogueConstructor. You can have several name for the
            same dataset but several catalogue and try different options.
        """
        self.dataio = dataio
        
        if chan_grp is None:
            chan_grp = min(self.dataio.channel_groups.keys())
        self.chan_grp = chan_grp
        self.nb_channel = self.dataio.nb_channel(chan_grp=self.chan_grp)
        self.geometry = self.dataio.get_geometry(chan_grp=self.chan_grp)

        self.channel_distances = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
        
        self.catalogue_path = os.path.join(self.dataio.channel_group_path[chan_grp], name)
        
        if not os.path.exists(self.catalogue_path):
            os.mkdir(self.catalogue_path)
        
        self.arrays = ArrayCollection(parent=self, dirname=self.catalogue_path)
        
        self.info_filename = os.path.join(self.catalogue_path, 'info.json')
        self.load_info()
        
        for name in _persitent_arrays:
            # this set attribute to class if exsits
            self.arrays.load_if_exists(name)
            
        if self.all_peaks is not None:
            self.memory_mode='memmap'
        
        self.projector = None
    
    def flush_info(self):
        """ Flush info (mainly parameters) to json files.
        """
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f, indent=4)
    
    def load_info(self):
        if not os.path.exists(self.info_filename):
            #first init
            self.info = {}
            self.flush_info()
        else:
            with open(self.info_filename, 'r', encoding='utf8') as f:
                self.info = json.load(f)
            if 'n_spike_for_centroid' not in self.info:
                self.info['n_spike_for_centroid'] = _default_n_spike_for_centroid
    
        for k in _global_params_attr:
            if k in self.info:
                setattr(self, k, self.info[k])
    
    def __repr__(self):
        t = "CatalogueConstructor\n"
        t += '  ' + self.dataio.channel_group_label(chan_grp=self.chan_grp) + '\n'
        if self.all_peaks is None:
            t += '  Signal pre-processing not done yet'
            return t
        
        #~ t += "  nb_peak: {}\n".format(self.nb_peak)
        nb_peak_by_segment = [ np.sum(self.all_peaks['segment']==i)  for i in range(self.dataio.nb_segment)]
        t += '  nb_peak_by_segment: '+', '.join('{}'.format(n) for n in nb_peak_by_segment)+'\n'

        #~ if self.some_waveforms is not None:
            #~ if self.mode == 'sparse':
                #~ sparsity = self.some_waveforms_sparse_mask.sum() / self.some_waveforms_sparse_mask.size
                #~ sparsity = '(sparse {:.2f})'.format(sparsity)
            #~ else:
                #~ sparsity = ''
            #~ t += '  some_waveforms.shape: {} {}\n'.format(self.some_waveforms.shape, sparsity)
            
        if self.some_features is not None:
            t += '  some_features.shape: {}\n'.format(self.some_features.shape)
        
        if hasattr(self, 'cluster_labels'):
            if self.cluster_labels.size < 8:
                labels = self.cluster_labels
            else:
                labels = '[' + ' '.join( str(l) for l in self.cluster_labels[:3]) + ' ... ' + ' '.join( str(l) for l in self.cluster_labels[-2:]) + ']'
            n = self.positive_cluster_labels.size
            t += '  cluster_labels {} {}\n'.format(n, labels)
        
        
        return t

    @property
    def nb_peak(self):
        if self.all_peaks is None:
            return 0
        return self.all_peaks.size

    @property
    def cluster_labels(self):
        if self.clusters is not None:
            return self.clusters['cluster_label']
        else:
            return np.array([], dtype='int64')
    
    @property
    def positive_cluster_labels(self):
        return self.cluster_labels[self.cluster_labels>=0] 

    
    def reload_data(self):
        """Reload persitent arrays.
        """
        if not hasattr(self, 'memory_mode') or not self.memory_mode=='memmap':
            return
        
        for name in _persitent_arrays:
            # this set attribute to class if exsits
            self.arrays.load_if_exists(name)

    def _reset_arrays(self, list_arrays):
        
        #TODO fix this need to delete really this arrays but size=0 is buggy
        for name in list_arrays:
            self.arrays.detach_array(name)
            setattr(self, name, None)
    
    def set_global_params(self, 
            chunksize=1024,
            memory_mode='memmap',   # 'memmap' or 'ram' 
            internal_dtype = 'float32',  # TODO "int16"
            mode='dense',
            #~ adjacency_radius_um=None,
            sparse_threshold=1.5,
            n_spike_for_centroid=_default_n_spike_for_centroid,
            n_jobs=-1,
            ):
        """
        
        
        Parameters
        ----------
        chunksize: int. default 1024
            Length of each chunk for processing.
            For real time, the latency is between chunksize and 2*chunksize.
        memory_mode: 'memmap' or 'ram' default 'memmap'
            By default all arrays are persistent with memmap but you can
            also have everything in ram.
        internal_dtype:  'float32' (or 'float64')
            Internal dtype for signals/waveforms/features.
            Support of intteger signal and waveform is planned for one day and should
            boost the process!!
        mode: str dense or sparse
            Choose the global mode dense or sparse.
        adjacency_radius_um: float or None
            When mode='sparse' then this must not be None.
        """

        #~ if mode == 'sparse':
            #~ assert adjacency_radius_um is not None
            #~ assert adjacency_radius_um > 0. 
        #~ elif mode == 'dense':
            #~ # chosse a radius that connect all channels
            #~ if self.dataio.nb_channel(self.chan_grp) >1:
                #~ channel_distances = self.dataio.get_channel_distances(chan_grp=self.chan_grp)
                #~ # important more than 2 times because
                #~ # adjacency_radius_um is use for waveform with this radius
                #~ # but also in pruningshears with half radius
                #~ adjacency_radius_um = np.max(channel_distances) * 2.5
        
        self.info['chunksize'] = chunksize
        self.info['memory_mode'] = memory_mode
        self.info['internal_dtype'] = internal_dtype
        self.info['mode'] = mode
        #~ self.info['adjacency_radius_um'] = adjacency_radius_um
        self.info['sparse_threshold'] = sparse_threshold
        self.info['n_spike_for_centroid'] = n_spike_for_centroid
        self.info['n_jobs'] = n_jobs

        self.flush_info()
        self.load_info() # this make attribute

        #~ self.chunksize = chunksize
        #~ self.memory_mode = memory_mode
        #~ self.internal_dtype = internal_dtype
        #~ self.mode = mode
        #~ self.adjacency_radius_um = adjacency_radius_um
        
        self._reset_arrays(_persitent_arrays)


    def set_preprocessor_params(self, 
    
            #signal preprocessor
            engine='numpy',
            highpass_freq=300., 
            lowpass_freq=None,
            smooth_size=0,
            common_ref_removal=False,
            
            lostfront_chunksize=None,
            
            
            #peak detector
            
            ):
        """
        Set parameters for the preprocessor engine
        
        Parameters
        ----------
        engine='numpy' or 'opencl'
            If you have pyopencl installed and correct ICD installed you can try
            'opencl' for high channel count some critial part of the processing is done on
            the GPU.
        highpass_freq: float dfault 300
            High pass cut frquency of the filter. Can be None if the raw 
            dataset is already filtered.
        lowpass_freq: float default is None
            Low pass frequency of the filter. None by default.
            For high samplign rate a low pass at 5~8kHz can be a good idea
            for smoothing a bit waveform and avoid high noise.
        smooth_size: int default 0
            This a simple smooth convolution kernel. More or less act
            like a low pass filter. Can be use instead lowpass_freq.
        common_ref_removal: bool. False by dfault.
            The remove the median of all channel sample by sample.
        lostfront_chunksize: int. default None
            size in sample of the margin at the front edge for each chunk to avoid border effect in backward filter.
            In you don't known put None then lostfront_chunksize will be int(sample_rate/highpass_freq)*3 which is quite robust (<5% error)
            compared to a true offline filtfilt.
        """

        
        self._reset_arrays(_persitent_arrays)
        
        # set default lostfront_chunksize if none is provided
        if lostfront_chunksize is None or lostfront_chunksize<=0:
            assert highpass_freq is not None, 'lostfront_chunksize=None needs a highpass_freq'
            lostfront_chunksize = int(self.dataio.sample_rate/highpass_freq*3)
        
        self.signal_preprocessor_params = dict(highpass_freq=highpass_freq, lowpass_freq=lowpass_freq, 
                        smooth_size=smooth_size, common_ref_removal=common_ref_removal,
                        lostfront_chunksize=lostfront_chunksize, output_dtype=self.internal_dtype,
                        engine=engine)
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[engine]
        self.signalpreprocessor = SignalPreprocessor_class(self.dataio.sample_rate, self.nb_channel, self.chunksize, self.dataio.source_dtype)
        
        for i in range(self.dataio.nb_segment):
            self.dataio.reset_processed_signals(seg_num=i, chan_grp=self.chan_grp, dtype=self.internal_dtype)
        
        # put all params in info
        self.info['signal_preprocessor_params'] = self.signal_preprocessor_params
        self.flush_info()
    
    def set_peak_detector_params(self, 
            method='global', engine='numpy',
            peak_sign='-', relative_threshold=7, peak_span_ms=0.3,
            adjacency_radius_um=None, smooth_radius_um=None):
        """
        Set parameters for the peak_detector engine
        
        Parameters
        ----------
        method : 'global' or 'geometrical'
            Method for detection.
        engine: 'numpy' or 'opencl' or 'numba'
            Engine for peak detection.
        peak_sign: '-' or '+'
            Signa of peak.
        relative_threshold: int default 7
            Threshold for peak detection. The preprocessed signal have units
            expressed in MAD (robust STD). So 7 is MAD*7.
        peak_span_ms: float default 0.3
            Peak span to avoid double detection. In millisecond.
        
        """
        
        if self.mode == 'sparse':
            assert method in ('geometrical', )
            assert adjacency_radius_um is not None

        self.peak_detector_params = dict(peak_sign=peak_sign, relative_threshold=relative_threshold,
                    peak_span_ms=peak_span_ms, engine=engine, method=method,
                    adjacency_radius_um=adjacency_radius_um, smooth_radius_um=smooth_radius_um)
        
        PeakDetector_class = peakdetector.get_peak_detector_class(method, engine)
        
        geometry = self.dataio.get_geometry(self.chan_grp)
        self.peakdetector = PeakDetector_class(self.dataio.sample_rate, self.nb_channel,
                                                        self.chunksize, self.internal_dtype, geometry)

        p = dict(self.peak_detector_params)
        p.pop('engine')
        p.pop('method')
        self.peakdetector.change_params(**p)

        self.info['peak_detector_params'] = self.peak_detector_params
        self.flush_info()
    
    def estimate_signals_noise(self, seg_num=0, duration=10.):
        """
        This estimate the median and mad on processed signals on 
        a short duration. This will be necessary for normalisation
        in next steps.
        
        Note that if the noise is stable even a short duratio is OK.
        
        Parameters
        ----------
        seg_num: int
            segment index
        duration: float
            duration in seconds
        
        """
        length = int(duration*self.dataio.sample_rate)
        length -= length%self.chunksize
        
        assert length<self.dataio.get_segment_length(seg_num), 'duration exeed size'
        
        name = 'filetered_sigs_for_noise_estimation_seg_{}'.format(seg_num)
        shape=(length - self.signal_preprocessor_params['lostfront_chunksize'], self.nb_channel)
        filtered_sigs = self.arrays.create_array(name, self.info['internal_dtype'], shape, 'memmap')
        
        params2 = dict(self.signal_preprocessor_params)
        params2.pop('engine')
        params2['normalize'] = False
        self.signalpreprocessor.change_params(**params2)
        
        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=self.chan_grp, chunksize=self.chunksize, i_stop=length,
                                                    signal_type='initial')
        for pos, sigs_chunk in iterator:
            pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
            if preprocessed_chunk is not None:
                filtered_sigs[pos2-preprocessed_chunk.shape[0]:pos2, :] = preprocessed_chunk

        #create  persistant arrays
        self.arrays.create_array('signals_medians', self.info['internal_dtype'], (self.nb_channel,), 'memmap')
        self.arrays.create_array('signals_mads', self.info['internal_dtype'], (self.nb_channel,), 'memmap')
        
        self.signals_medians[:] = signals_medians = np.median(filtered_sigs[:pos2], axis=0)
        self.signals_mads[:] = np.median(np.abs(filtered_sigs[:pos2]-signals_medians),axis=0)*1.4826
        
        #detach filetered signals even if the file remains.
        self.arrays.detach_array(name)
        

    #~ def signalprocessor_one_chunk(self, pos, sigs_chunk, seg_num, detect_peak=True):

        #~ pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        #~ if preprocessed_chunk is  None:
            #~ return
        
        #~ self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num, chan_grp=self.chan_grp,
                        #~ i_start=pos2-preprocessed_chunk.shape[0], i_stop=pos2, signal_type='processed')
        
        #~ if detect_peak:
            #~ index_chunk_peaks, peak_chan_index = self.peakdetector.process_data(pos2, preprocessed_chunk)
            
            #~ # chan_index can be None for some method
            
            #~ if index_chunk_peaks is not None:
                #~ peaks = np.zeros(index_chunk_peaks.size, dtype=_dtype_peak)
                #~ peaks['index'] = index_chunk_peaks
                #~ peaks['segment'][:] = seg_num
                #~ peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
                #~ if peak_chan_index is None:
                    #~ peaks['channel'][:] = -1
                #~ else:
                    #~ peaks['channel'][:] = peak_chan_index
                #~ self.arrays.append_chunk('all_peaks',  peaks)
        
    
    
    def run_signalprocessor_loop_one_segment(self, seg_num=0, duration=60., detect_peak=True):
        
        if detect_peak:
            assert 'peak_detector_params' in self.info
            assert len(self.info['peak_detector_params'])>0
        
        length = int(duration*self.dataio.sample_rate)
        length = min(length, self.dataio.get_segment_length(seg_num))
        
        #~ length -= length%self.chunksize

        #initialize engines
        p = dict(self.signal_preprocessor_params)
        p.pop('engine')
        p['normalize'] = True
        p['signals_medians'] = self.signals_medians
        p['signals_mads'] = self.signals_mads
        self.signalpreprocessor.change_params(**p)
        
        self.peakdetector.reset_fifo_index()
        
        # iterate a bit more on the rigth border
        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=self.chan_grp,
                        chunksize=self.chunksize, i_stop=length, signal_type='initial',
                        pad_width=self.info['signal_preprocessor_params']['lostfront_chunksize'],
                        with_last_chunk=True)
        for pos, sigs_chunk in iterator:
            #~ print(seg_num, pos, sigs_chunk.shape)
            #~ self.signalprocessor_one_chunk(pos, sigs_chunk, seg_num, detect_peak=detect_peak)
            
            pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
            if preprocessed_chunk is  None:
                continue
            
            
            if detect_peak:
                index_chunk_peaks, peak_chan_index, peak_val_peaks = self.peakdetector.process_data(pos2, preprocessed_chunk)
                
                # chan_index can be None for some method
                # peak_chan_indexcan be None for some method
                
                if index_chunk_peaks is not None:
                    peaks = np.zeros(index_chunk_peaks.size, dtype=_dtype_peak)
                    peaks['index'] = index_chunk_peaks
                    peaks['segment'][:] = seg_num
                    peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
                    if peak_chan_index is None:
                        peaks['channel'][:] = -1
                    else:
                        peaks['channel'][:] = peak_chan_index
                    if peak_val_peaks is None:
                        peaks['extremum_amplitude'][:] = 0.
                    else:
                        peaks['extremum_amplitude'][:] = peak_val_peaks
                    self.arrays.append_chunk('all_peaks',  peaks)

            if pos2>length:
                # clip writting
                preprocessed_chunk = preprocessed_chunk[:-(pos2-length), :]
                pos2 = length
                
            self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num, chan_grp=self.chan_grp,
                            i_start=pos2-preprocessed_chunk.shape[0], i_stop=pos2, signal_type='processed')
        
        self.dataio.flush_processed_signals(seg_num=seg_num, chan_grp=self.chan_grp, processed_length=int(pos2))
    
    
    def run_signalprocessor(self, duration=60., detect_peak=True):
        """
        this run (chunk by chunk), the signal preprocessing chain on
        all segments.
        
        The duration can be clip for very long recording. For catalogue 
        construction the user must have the intuition of how signal we must
        have to get enough spike to detect clusters. If the duration is too short clusters
        will not be visible. If too long the processing time will be unacceptable.
        This totally depend on the dataset (nb channel, spike rate ...)
        
        This also detect peak to avoid to useless access to the storage.
        
        Parameters
        ----------
        duration: float
            total duration in seconds for all segment
        detect_peak: bool (default True)
            Also detect peak.
        
        """
        self.arrays.initialize_array('all_peaks', self.memory_mode,  _dtype_peak, (-1, ))
        
        #~ duration_per_segment = []
        #~ total_duration = duration
        #~ for seg_num in range(self.dataio.nb_segment):
            #~ dur = self.dataio.get_segment_length(seg_num=seg_num) / self.dataio.sample_rate
            #~ if total_duration ==0:
                #~ duration_per_segment.append(0.)
            #~ elif dur <=total_duration:
                #~ duration_per_segment.append(dur)
                #~ total_duration -= dur
            #~ else:
                #~ duration_per_segment.append(total_duration)
                #~ total_duration = 0.
        duration_per_segment = self.dataio.get_duration_per_segments(duration)
        #~ print(duration_per_segment)
        for seg_num in range(self.dataio.nb_segment):
            self.run_signalprocessor_loop_one_segment(seg_num=seg_num, duration=duration_per_segment[seg_num], detect_peak=detect_peak)
        
        # flush peaks
        self.arrays.finalize_array('all_peaks')
        
        self._reset_arrays(_reset_after_peak_arrays)
        self.on_new_cluster()
    
    
    def re_detect_peak(self, **kargs):
        """
        Peak are detected while **run_signalprocessor**.
        But in some case for testing other threshold we can **re-detect peak** without signal processing.
        
        Parameters
        ----------
        method : 'global' or 'geometrical'
            Method for detection.
        engine: 'numpy' or 'opencl' or 'numba'
            Engine for peak detection.
        peak_sign: '-' or '+'
            Signa of peak.
        relative_threshold: int default 7
            Threshold for peak detection. The preprocessed signal have units
            expressed in MAD (robust STD). So 7 is MAD*7.
        peak_span_ms: float default 0.3
            Peak span to avoid double detection. In second.
        
        """
        self.set_peak_detector_params(**kargs)
        
        self.arrays.initialize_array('all_peaks', self.memory_mode,  _dtype_peak, (-1, ))
        
        for seg_num in range(self.dataio.nb_segment):
            
            self.peakdetector.reset_fifo_index()
            
            i_stop = self.dataio.get_processed_length(seg_num=seg_num, chan_grp=self.chan_grp)
            
            iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=self.chan_grp,
                            chunksize=self.info['chunksize'], i_stop=i_stop, signal_type='processed')
            for pos, preprocessed_chunk in iterator:
                index_chunk_peaks, peak_chan_index, peak_val_peaks = self.peakdetector.process_data(pos, preprocessed_chunk)
                
                # peak_chan_index can be None
                # peak_val_peaks can be None
            
                if index_chunk_peaks is not None:
                    peaks = np.zeros(index_chunk_peaks.size, dtype=_dtype_peak)
                    peaks['index'] = index_chunk_peaks
                    peaks['segment'][:] = seg_num
                    peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
                    if peak_chan_index is None:
                        peaks['channel'][:] = -1
                    else:
                        peaks['channel'][:] = peak_chan_index
                    if peak_val_peaks is None:
                        peaks['extremum_amplitude'][:] = 0.
                    else:
                        peaks['extremum_amplitude'][:] = peak_val_peaks
                    self.arrays.append_chunk('all_peaks',  peaks)

        self.arrays.finalize_array('all_peaks')
        #~ self._reset_waveform_and_features()
        self._reset_arrays(_reset_after_peak_arrays)
        self.on_new_cluster()
    
    
    def set_waveform_extractor_params(self, n_left=None, n_right=None,
                            wf_left_ms=None, wf_right_ms=None,
                            wf_left_long_ms=None, wf_right_long_ms=None,
                            ):
        if n_left is None or n_right is None:
            if 'waveform_extractor_params' in self.info:
                n_left = self.info['waveform_extractor_params']['n_left']
                n_right = self.info['waveform_extractor_params']['n_right']
            elif wf_left_ms is not None and wf_right_ms is not None:
                n_left = int(wf_left_ms / 1000. * self.dataio.sample_rate)
                n_right = int(wf_right_ms / 1000. * self.dataio.sample_rate)
            else:
                raise(ValueError('Must provide wf_left_ms/wf_right_ms'))
        if wf_left_long_ms is None:
            n_left_long = 2 * n_nleft
        else:
            n_left_long = int(wf_left_long_ms / 1000. * self.dataio.sample_rate)
        if wf_right_long_ms is None:
            n_right_long = 2 * n_right
        else:
            n_right_long = int(wf_right_long_ms / 1000. * self.dataio.sample_rate)
        
        assert n_left < n_right
        assert n_left_long < n_left
        assert n_right_long > n_right
        
        self.info['waveform_extractor_params'] = dict(n_left=n_left, n_right=n_right, n_left_long=n_left_long, n_right_long=n_right_long)
        self.flush_info()
        
        self.projector = None
        self._reset_arrays(_reset_after_peak_sampler)
    
    def clean_peaks(self, alien_value_threshold=200, mode='extremum_amplitude'):
        """
        Detect alien peak with very high values.
        Tag then with alien (label=-9)
        This prevent then to be selected.
        Usefull to remove artifact.
        
        alien_value_threshold: float or None
            threhold for alien peak
        mode='peak_value' / 'full_waveform'
            'peak_value' use only the peak value very fast because already measured
            'full_waveform' use the full waveform for this and can be slow.
        """
        
        if alien_value_threshold is not None:
            peak_sign = self.info['peak_detector_params']['peak_sign']
            
            if mode == 'extremum_amplitude':
                if peak_sign == '+':
                    index_over, = np.nonzero(self.all_peaks['extremum_amplitude'] >= alien_value_threshold)
                elif peak_sign == '-':
                    index_over, = np.nonzero(self.all_peaks['extremum_amplitude'] <= -alien_value_threshold)
                
            elif mode == 'full_waveform':
                # need to load all waveforms chunk by chunk
                alien_mask = np.zeros(self.all_peaks.size, dtype='bool')
                n = 100000
                nloop = self.all_peaks.size // n
                if self.all_peaks.size % n:
                    nloop += 1
                for i in range(nloop):
                    i1, i2 = i, min(i+n, self.all_peaks.size)
                    peaks_index = np.arange(i1, i2, dtype='int64')
                    wfs = self.get_some_waveforms(peaks_index=peaks_index, channel_indexes=None)
                    over1 = np.any(wfs>alien_value_threshold, axis=(1,2))
                    over2 = np.any(wfs<-alien_value_threshold, axis=(1,2))
                    over = over1 | over2
                    alien_mask[i1:i2] = over
                
                index_over, = np.nonzero(alien_mask)
            
            self.all_peaks['cluster_label'][index_over] = labelcodes.LABEL_ALIEN

        self.info['clean_peaks_params'] = dict(alien_value_threshold=alien_value_threshold, mode=mode)
        self.flush_info()
        self._reset_arrays(_reset_after_peak_sampler)
        self.on_new_cluster()

    
    def sample_some_peaks(self, mode='rand', 
                            nb_max=10000, nb_max_by_channel=1000, index=None):
        
        if mode=='rand':
            if self.nb_peak > nb_max:
                some_peaks_index = np.random.choice(self.nb_peak, size=nb_max).astype('int64')
            else:
                some_peaks_index = np.arange(self.nb_peak, dtype='int64')
        elif mode=='rand_by_channel':
            assert self.mode == 'sparse'
            some_peaks_index = []
            for c in range(self.nb_channel):
                ind, = np.nonzero(self.all_peaks['channel'] == c)
                if ind.size > nb_max_by_channel:
                    take = np.random.choice(ind.size, size=nb_max_by_channel, replace=False)#.astype('int64')
                    ind = ind[take]
                some_peaks_index.append(ind)
            some_peaks_index = np.concatenate(some_peaks_index)
            
        elif mode=='all':
            some_peaks_index = np.arange(self.nb_peak, dtype='int64')
        elif mode =='force':
            assert index is not None, 'With mode=force you must give peak index'
            some_peaks_index = index.copy()
            
        else:
            raise(NotImplementedError, 'unknown mode')
        
        #remove alien
        valid = self.all_peaks['cluster_label'][some_peaks_index] != labelcodes.LABEL_ALIEN
        if mode in ('rand', 'rand_by_channel', 'all'):
            some_peaks_index = some_peaks_index[valid]
        else:
            if np.any(~valid):
                print('WARNING : sample_some_peaks with mode "froce" but take some alien peaks')

        n_left = self.info['waveform_extractor_params']['n_left']
        n_right = self.info['waveform_extractor_params']['n_right']
        peak_width = n_right - n_left
        n_left_long = self.info['waveform_extractor_params']['n_left_long']
        n_right_long = self.info['waveform_extractor_params']['n_right_long']
        peak_width_long = n_right_long - n_left_long
            
        
        # this is important to not take 2 times the sames, this leads to bad mad/median
        some_peaks_index = np.unique(some_peaks_index)

        # remove peak_index near border
        keep = np.zeros(some_peaks_index.size, dtype='bool')
        seg_nums = np.unique(self.all_peaks['segment'])
        
        for seg_num in seg_nums:
            in_seg_mask = self.all_peaks[some_peaks_index]['segment'] == seg_num
            indexes  = self.all_peaks[some_peaks_index]['index']
            in_seg_keep = (indexes > peak_width_long) & (indexes < self.dataio.get_segment_length(seg_num) - peak_width_long)
            keep |= in_seg_mask & in_seg_keep
        some_peaks_index = some_peaks_index[keep]        
        
        # make persistent
        self.arrays.add_array('some_peaks_index', some_peaks_index, self.memory_mode)
        
        alien_index, = np.nonzero(self.all_peaks['cluster_label'] == labelcodes.LABEL_ALIEN)
        self.all_peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
        self.all_peaks['cluster_label'][alien_index] = labelcodes.LABEL_ALIEN
        self.all_peaks['cluster_label'][self.some_peaks_index] = 0
        
        self.info['peak_sampler_params'] = dict(mode=mode, nb_max=nb_max, nb_max_by_channel=nb_max_by_channel)
        self.flush_info()
        
        self.on_new_cluster()
        
    
    
    def get_some_waveforms(self, peaks_index=None, channel_indexes=None, n_left=None, n_right=None):
        """
        Get waveforms accros segment based on internal self.some_peak_index
        forced peaks_index.
        
        """
        if peaks_index is None:
            assert self.some_peaks_index is not None
            peaks_index = self.some_peaks_index

        nb = peaks_index.size
        
        if n_left is None:
            n_left = self.info['waveform_extractor_params']['n_left']
        if n_right is None:
            n_right = self.info['waveform_extractor_params']['n_right']
        peak_width = n_right - n_left
        
        if channel_indexes is None:
            nb_chan = self.nb_channel
        else:
            nb_chan = len(channel_indexes)            
        
        shape = (nb, peak_width, nb_chan)
        some_waveforms = np.zeros(shape, dtype=self.info['internal_dtype'])
        
        peak_mask = np.zeros(self.nb_peak, dtype='bool')
        peak_mask[peaks_index] = True
        
        seg_nums = self.all_peaks[peak_mask]['segment']
        peak_sample_indexes = self.all_peaks[peak_mask]['index']
        self.dataio.get_some_waveforms(seg_nums=seg_nums, chan_grp=self.chan_grp,
                                                peak_sample_indexes=peak_sample_indexes, n_left=n_left, n_right=n_right,
                                                waveforms=some_waveforms, channel_indexes=channel_indexes)
        
        return some_waveforms


    def cache_some_waveforms(self):
        n_left_long = self.info['waveform_extractor_params']['n_left_long']
        n_right_long = self.info['waveform_extractor_params']['n_right_long']
        peak_width_long = n_right_long - n_left_long
        
        if 'some_waveforms' in self.arrays.keys():
            self.arrays.delete_array('some_waveforms')
        if 'some_waveform_index' in self.arrays.keys():
            self.arrays.delete_array('some_waveform_index')
        
        selected_indexes = []
        for label in self.positive_cluster_labels:
            selected, = np.nonzero(self.all_peaks['cluster_label']==label)
            if selected.size>self.n_spike_for_centroid:
                keep = np.random.choice(selected.size, self.n_spike_for_centroid, replace=False)
                selected = selected[keep]
            selected_indexes.append(selected)
        
        selected_indexes = np.concatenate(selected_indexes)
        
        n = selected_indexes.size
        shape = (n, peak_width_long, self.nb_channel)
        self.arrays.create_array('some_waveforms',  self.info['internal_dtype'] , shape, self.memory_mode)
        self.arrays.create_array('some_waveform_index', 'int64', n, self.memory_mode)
        
        
        seg_nums = self.all_peaks[selected_indexes]['segment']
        peak_sample_indexes = self.all_peaks[selected_indexes]['index']
        
        self.dataio.get_some_waveforms(seg_nums=seg_nums, chan_grp=self.chan_grp,
                                                peak_sample_indexes=peak_sample_indexes, n_left=n_left_long, n_right=n_right_long,
                                                waveforms=self.some_waveforms, channel_indexes=None)
        self.some_waveform_index[:] = selected_indexes

    def extend_cached_waveforms(self, label):
        n_left_long = self.info['waveform_extractor_params']['n_left_long']
        n_right_long = self.info['waveform_extractor_params']['n_right_long']
        
        selected, = np.nonzero(self.all_peaks['cluster_label']==label)
        if selected.size>self.n_spike_for_centroid:
            keep = np.random.choice(selected.size, self.n_spike_for_centroid, replace=False)
            selected = selected[keep]

        seg_nums = self.all_peaks[selected]['segment']
        peak_sample_indexes = self.all_peaks[selected]['index']

        new_waveforms = self.dataio.get_some_waveforms(seg_nums=seg_nums, chan_grp=self.chan_grp,
                                                peak_sample_indexes=peak_sample_indexes, n_left=n_left_long, n_right=n_right_long,
                                                waveforms=None, channel_indexes=None)

        self.arrays.append_array('some_waveforms',  new_waveforms)
        self.arrays.append_array('some_waveform_index', selected)

    def get_cached_waveforms(self, label, long=False):
        assert self.some_waveforms is not None, 'run cc.cache_some_waveforms() first'
        
        cached_labels = self.all_peaks[self.some_waveform_index]['cluster_label']
        
        mask = cached_labels == label
        if long:
            wfs = self.some_waveforms[mask]
        else:
            n_left = self.info['waveform_extractor_params']['n_left']
            n_right = self.info['waveform_extractor_params']['n_right']
            n_left_long = self.info['waveform_extractor_params']['n_left_long']
            n_right_long = self.info['waveform_extractor_params']['n_right_long']
            i0 = n_left - n_left_long
            i1 = n_right_long - n_right
            wfs = self.some_waveforms[mask][:, i0:-i1]
        
        # avoid memmap ref for ploting
        wfs = wfs.copy()
        return wfs
    
    def delete_waveforms_cache(self):
        if 'some_waveforms' in self.arrays.keys():
            self.arrays.delete_array('some_waveforms')
        if 'some_waveform_index' in self.arrays.keys():
            self.arrays.delete_array('some_waveform_index')        

    def extract_some_noise(self, nb_snippet=300):
        """
        Find some snipet of signal that are not overlap with peak waveforms.
        
        Usefull to project this noise with the same tranform as real waveform
        and see the distinction between waveforma and noise in the subspace.
        """
        #~ 'some_noise_index', 'some_noise_snippet', 
        assert  'waveform_extractor_params' in self.info
        n_left = self.info['waveform_extractor_params']['n_left']
        n_right = self.info['waveform_extractor_params']['n_right']
        peak_width = n_right - n_left
        
        
        #~ self.all_peaks
        #~ _dtype_peak = [('index', 'int64'), ('cluster_label', 'int64'), ('segment', 'int64'),]
        
        some_noise_index = []
        n_by_seg = nb_snippet//self.dataio.nb_segment
        for seg_num in range(self.dataio.nb_segment):
            
            # tdc can process only the begining of the segment so
            length = min(self.dataio.get_segment_length(seg_num),
                        self.dataio.get_processed_length(seg_num, chan_grp=self.chan_grp))
            if length == 0:
                continue
            
            possibles = np.ones(length, dtype='bool')
            possibles[:peak_width] = False
            possibles[-peak_width:] = False
            peaks = self.all_peaks[self.all_peaks['segment']==seg_num]
            for peak in peaks:
                possibles[peak['index']+n_left:peak['index']+n_right] = False
            possible_indexes, = np.nonzero(possibles)
            if possible_indexes.size == 0:
                print('WARNING no noise snipet possible!!! Take random sigs instead', seg_num)
                # take random places
                #~ print('length', length)
                inds = np.sort(np.random.choice(length, size=n_by_seg))
            else:
                inds = possible_indexes[np.sort(np.random.choice(possible_indexes.size, size=n_by_seg))]
            noise_index = np.zeros(n_by_seg, dtype=_dtype_peak)
            noise_index['index'] = inds
            noise_index['cluster_label'] = labelcodes.LABEL_NOISE
            noise_index['segment'][:] = seg_num
            some_noise_index.append(noise_index)
        some_noise_index = np.concatenate(some_noise_index)
        
        #make it persistent
        self.arrays.add_array('some_noise_index', some_noise_index, self.memory_mode)
        
        #create snipet
        shape=(self.some_noise_index.size, peak_width, self.nb_channel)
        self.arrays.create_array('some_noise_snippet', self.info['internal_dtype'], shape, self.memory_mode)
        #~ n = 0
        for n, ind in enumerate(self.some_noise_index):
        #~ for seg_num in range(self.dataio.nb_segment):
            #~ insegment_indexes  = self.some_noise_index[(self.some_noise_index['segment']==seg_num)]
            #~ for ind in insegment_indexes:
            i_start = ind['index']+n_left
            i_stop = i_start+peak_width
            snippet = self.dataio.get_signals_chunk(seg_num=ind['segment'], chan_grp=self.chan_grp, i_start=i_start, i_stop=i_stop, signal_type='processed')
            #~ print(i_start, i_stop, self.some_noise_snippet.shape, self.dataio.get_segment_length(ind['segment']))
            self.some_noise_snippet[n, :, :] = snippet
                #~ n +=1

    def extract_some_features(self, method='global_pca',  selection=None, **params):
        """
        Extract feature from waveforms.
        
        If selection is None then all peak from some_peak_index are taken.
        
        Else selection is mask bool size all_peaks used for fit and  then the tranform is applied on all.
        
        """

        # global feature log it
        self.info['feature_method'] = method
        self.info['feature_kargs'] = params
        self.flush_info()
        
        features, channel_to_features, self.projector = decomposition.project_waveforms(catalogueconstructor=self,method=method, selection=selection, **params)
        
        if features is None:
            for name in ['some_features', 'channel_to_features', 'some_noise_features']:
                self.arrays.detach_array(name)
                setattr(self, name, None)            
        else:
            # make it persistant
            self.arrays.add_array('some_features', features.astype(self.info['internal_dtype']), self.memory_mode)
            self.arrays.add_array('channel_to_features', channel_to_features, self.memory_mode)
            
            if self.some_noise_snippet is not None:
                some_noise_features = self.projector.transform(self.some_noise_snippet)
                self.arrays.add_array('some_noise_features', some_noise_features.astype(self.info['internal_dtype']), self.memory_mode)
        
            #~ print('extract_some_features', self.some_features.shape)
    
    #ALIAS TODO remove it
    #~ project = extract_some_features
    
    #~ def apply_projection(self):
        #~ assert self.projector is not None
        #~ features = self.projector.transform(self.some_waveforms)
        
        #trick to make it persistant
        #~ self.arrays.create_array('some_features', self.info['internal_dtype'], features.shape, self.memory_mode)
        #~ self.some_features[:] = features
        #~ self.arrays.add_array('some_features', some_features.astype(self.info['internal_dtype']), self.memory_mode)
        
    
    
    def find_clusters(self, method='kmeans', selection=None, order=True, recompute_centroid=True, **kargs):
        """
        Find cluster for peaks that have a waveform and feature.
        
        selection is mask bool size all_peaks
        """
        #done in a separate module cluster.py
    
        self.delete_waveforms_cache()
        
        if selection is not None:
            old_labels = np.unique(self.all_peaks['cluster_label'][selection])
            #~ print(old_labels)
            
        
        labels = cluster.find_clusters(self, method=method, selection=selection, **kargs)
        
        if selection is None:
            self.on_new_cluster()
            if recompute_centroid:
                self.compute_all_centroid(n_spike_for_centroid=self.n_spike_for_centroid)
            
            if order:
                self.order_clusters(by='waveforms_rms')

            # global cluster log it
            self.info['cluster_method'] = method
            self.info['cluster_kargs'] = kargs
            self.flush_info()
                
        else:
            new_labels = np.unique(labels)
            for new_label in new_labels:
                if new_label not in self.clusters['cluster_label'] and new_label>=0:
                    self.add_one_cluster(new_label)
                if new_label>=0:
                    self.compute_one_centroid(new_label, n_spike_for_centroid=self.n_spike_for_centroid)
            
            for old_label in old_labels:
                ind = self.index_of_label(old_label)
                nb_peak = np.sum(self.all_peaks['cluster_label']==old_label)
                if nb_peak == 0:
                    self.pop_labels_from_cluster([old_label])
                else:
                    self.clusters['nb_peak'][ind] = nb_peak
                    self.compute_one_centroid(old_label, n_spike_for_centroid=self.n_spike_for_centroid)
                    

    def on_new_cluster(self):
        if self.all_peaks is None:
            return
        cluster_labels = np.unique(self.all_peaks['cluster_label'])
        clusters = np.zeros(cluster_labels.shape, dtype=_dtype_cluster)
        clusters['cluster_label'][:] = cluster_labels
        clusters['cell_label'][:] = cluster_labels
        clusters['extremum_channel'][:] = -1
        clusters['extremum_amplitude'][:] = np.nan
        clusters['waveform_rms'][:] = np.nan
        for i, k in enumerate(cluster_labels):
            clusters['nb_peak'][i] = np.sum(self.all_peaks['cluster_label']==k)
        
        if self.clusters is not None:
            #get previous _keep_cluster_attr_on_new
            for i, c in enumerate(clusters):
                #~ print(i, c)
                if c['cluster_label'] in self.clusters['cluster_label']:
                    j = np.nonzero(c['cluster_label']==self.clusters['cluster_label'])[0][0]
                    for attr in _keep_cluster_attr_on_new:
                        #~ self.clusters[j]['cell_label'] in cluster_labels
                        #~ clusters[i]['cell_label'] = self.clusters[j]['cell_label']
                        clusters[attr][i] = self.clusters[attr][j]
                        
                    #~ print('j', j)
        
        #~ if clusters.size>0:
        self.arrays.add_array('clusters', clusters, self.memory_mode)
    
    def add_one_cluster(self, label):
        assert label not in self.clusters['cluster_label']
        
        clusters = np.zeros(self.clusters.size+1, dtype=_dtype_cluster)
        
        if label>=0:
            pos_insert = -1
            clusters[:-1] = self.clusters
        else:
            pos_insert = 0
            clusters[1:] = self.clusters
        
        clusters['cluster_label'][pos_insert] = label
        clusters['cell_label'][pos_insert] = label
        clusters['extremum_channel'][pos_insert] = -1
        clusters['extremum_amplitude'][pos_insert] = np.nan
        clusters['waveform_rms'][pos_insert] = np.nan
        clusters['nb_peak'][pos_insert] = np.sum(self.all_peaks['cluster_label']==label)
        
        self.arrays.add_array('clusters', clusters, self.memory_mode)
        
        for name in _centroids_arrays:
            arr = getattr(self, name).copy()
            
            if arr.ndim == 3:
                new_arr = np.zeros((arr.shape[0]+1, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            else:
                # special case "centroids_sparse_mask"
                new_arr = np.zeros((arr.shape[0]+1, arr.shape[1]), dtype=arr.dtype)
            
            if label>=0:
                new_arr[:-1] = arr
            else:
                new_arr[1:] = arr
            
            self.arrays.add_array(name, new_arr, self.memory_mode)
        
        #TODO set one color
        self.refresh_colors(reset=False)
        
        self.extend_cached_waveforms(label)
        self.compute_one_centroid(label)
    
    def index_of_label(self, label):
        ind = np.nonzero(self.clusters['cluster_label']==label)[0][0]
        return ind
    
    def remove_one_cluster(self, label):
        print('WARNING remove_one_cluster')
        # This should not be called any more
        # because name ambiguous
        self.pop_labels_from_cluster([label])
    
    def pop_labels_from_cluster(self, labels):
        # this reduce the array clusters by removing some labels
        # warning all_peak are touched
        if isinstance(labels, int):
            labels = [labels]
        keep = np.ones(self.clusters.size, dtype='bool')
        for k in labels:
            ind = self.index_of_label(k)
            keep[ind] = False

        clusters = self.clusters[keep].copy()
        self.arrays.add_array('clusters', clusters, self.memory_mode)
        
        for name in _centroids_arrays:
            new_arr = getattr(self, name)[keep].copy()  # first dim is cluster for all
            self.arrays.add_array(name, new_arr, self.memory_mode)
        
    def move_cluster_to_trash(self, labels):
        if isinstance(labels, int):
            labels = [labels]
        
        mask = np.zeros(self.all_peaks.size, dtype='bool')
        for k in labels:
            mask |= self.all_peaks['cluster_label']== k
        self.change_spike_label(mask, labelcodes.LABEL_TRASH)
    
    def compute_one_centroid(self, k, flush=True, n_spike_for_centroid=None):
        #~ t1 = time.perf_counter()
        if n_spike_for_centroid is None:
            n_spike_for_centroid = self.n_spike_for_centroid
        
        ind = self.index_of_label(k)
        
        n_left = self.info['waveform_extractor_params']['n_left']
        n_right = self.info['waveform_extractor_params']['n_right']
        n_left_long = int(self.info['waveform_extractor_params']['n_left_long'])
        n_right_long = int(self.info['waveform_extractor_params']['n_right_long'])
        peak_sign = self.info['peak_detector_params']['peak_sign']
        
        
        if self.some_waveforms is None:
            # waveforms not cached
            selected, = np.nonzero(self.all_peaks['cluster_label'][self.some_peaks_index]==k)
            if selected.size>n_spike_for_centroid:
                keep = np.random.choice(selected.size, n_spike_for_centroid, replace=False)
                selected = selected[keep]
            wf = self.get_some_waveforms(peaks_index=self.some_peaks_index[selected], channel_indexes=None,
                        n_left=n_left_long, n_right=n_right_long)
        else:
            wf = self.get_cached_waveforms(k, long=True)
        
        median_long, mad_long = median_mad(wf, axis = 0)
        # mean, std = np.mean(wf, axis=0), np.std(wf, axis=0) # TODO rome the mean/std
        if peak_sign == '-':
            extremum_channel = np.argmin(median_long[-n_left_long,:], axis=0)
        elif peak_sign == '+':
            extremum_channel = np.argmax(median_long[-n_left_long,:], axis=0)

        i0 = n_left - n_left_long
        i1 = n_right_long - n_right
        median = median_long[i0:-i1, :]
        mad = mad_long[i0:-i1, :]
        
        # to persistant arrays
        self.centroids_median_long[ind, :, :] = median_long
        self.centroids_median[ind, :, :] = median
        self.centroids_mad[ind, :, :] = mad
        #~ self.centroids_mean[ind, :, :] = mean
        #~ self.centroids_std[ind, :, :] = std
        self.centroids_mean[ind, :, :] = 0
        self.centroids_std[ind, :, :] = 0
        
        
        self.centroids_sparse_mask[ind, :] = np.any(np.abs(median) > self.sparse_threshold, axis=0)
        
        self.clusters['extremum_channel'][ind] = extremum_channel
        self.clusters['extremum_amplitude'][ind] = median[-n_left, extremum_channel]
        self.clusters['waveform_rms'][ind] = np.sqrt(np.mean(median**2))

        if flush:
            for name in ('clusters',) + _centroids_arrays:
                self.arrays.flush_array(name)
        
        #~ t2 = time.perf_counter()
        #~ print('compute_one_centroid',k, t2-t1)
    
    def compute_several_centroids(self, labels, n_spike_for_centroid=None):
        # TODO make this in paralell
        for k in labels:
            self.compute_one_centroid(k, flush=False, n_spike_for_centroid=n_spike_for_centroid)
        
        # one global flush
        for name in ('clusters',) + _centroids_arrays:
            self.arrays.flush_array(name)
        
    
    def compute_all_centroid(self, n_spike_for_centroid=None):
        t1 = time.perf_counter()
        if self.some_peaks_index is None:
            for name in _centroids_arrays:
                self.arrays.detach_array(name)
                setattr(self, name, None)
            return
        
        n_left = self.info['waveform_extractor_params']['n_left']
        n_right = self.info['waveform_extractor_params']['n_right']
        n_left_long = int(self.info['waveform_extractor_params']['n_left_long'])
        n_right_long = int(self.info['waveform_extractor_params']['n_right_long'])
        
        

        for name in ('centroids_median', 'centroids_mad', 'centroids_mean', 'centroids_std',):
            empty = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.info['internal_dtype'])
            self.arrays.add_array(name, empty, self.memory_mode)

        empty = np.zeros((self.cluster_labels.size, n_right_long - n_left_long, self.nb_channel), dtype=self.info['internal_dtype'])
        self.arrays.add_array('centroids_median_long', empty, self.memory_mode)

        mask = np.zeros((self.cluster_labels.size, self.nb_channel), dtype='bool')
        self.arrays.add_array('centroids_sparse_mask', mask, self.memory_mode)
        
        self.compute_several_centroids(self.positive_cluster_labels, n_spike_for_centroid=n_spike_for_centroid)
    
    def get_one_centroid(self, label, metric='median', long=False):
        ind = self.index_of_label(label)
        if long:
            centroid = self.centroids_median_long[ind, :, :].copy()
        else:
            attr = getattr(self, 'centroids_'+metric)
            # make a copy to avoid too much reference on the memmap object
            centroid = attr[ind, :, :].copy()
        
        return centroid
    
    def change_sparse_threshold(self, sparse_threshold=1.5):
        self.info['sparse_threshold'] = sparse_threshold
        self.flush_info()
        self.load_info() # this make attribute
        
        for k in self.cluster_labels:
            if k <0: continue
            ind = self.index_of_label(k)
            median = self.centroids_median[ind, :, :]
            
            self.centroids_sparse_mask[ind, :] = np.any(np.abs(median) > self.sparse_threshold, axis=0)            
            
        
        self.centroids_sparse_mask[ind, :] = np.any(np.abs(median) > self.sparse_threshold, axis=0)
        self.arrays.flush_array('centroids_sparse_mask')
        
        
    
    def refresh_colors(self, reset=True, palette='husl', interleaved=True):
        
        labels = self.positive_cluster_labels
        
        if reset:
            n = labels.size
            if interleaved and n>1:
                n1 = np.floor(np.sqrt(n))
                n2 = np.ceil(n/n1)
                n = int(n1*n2)
                n1, n2 = int(n1), int(n2)
        else:
            n = np.sum((self.clusters['cluster_label']>=0) & (self.clusters['color']==0))

        if n>0:
            colors_int32 = np.array([rgba_to_int32(r,g,b) for r,g,b in sns.color_palette(palette, n)])
            
            if reset and interleaved and n>1:
                colors_int32 = colors_int32.reshape(n1, n2).T.flatten()
                colors_int32 = colors_int32[:labels.size]
            
            if reset:
                mask = self.clusters['cluster_label']>=0
                self.clusters['color'][mask] = colors_int32
            else:
                mask = (self.clusters['cluster_label']>=0) & (self.clusters['color']==0)
                self.clusters['color'][mask] = colors_int32
        
        #Make colors accessible by key
        self.colors = make_color_dict(self.clusters)
        

    def change_spike_label(self, mask, label):
        is_new = label not in self.clusters['cluster_label']
        
        label_changed = np.unique(self.all_peaks['cluster_label'][mask]).tolist()
        self.all_peaks['cluster_label'][mask] = label
        
        if is_new:
            self.add_one_cluster(label) # this also compute centroid
        else:
            label_changed = label_changed + [label]
        
        to_remove = []
        for k in label_changed:
            ind = self.index_of_label(k)
            nb_peak = np.sum(self.all_peaks['cluster_label']==k)
            self.clusters['nb_peak'][ind] = nb_peak
            if k>=0:
                if nb_peak>0:
                    self.compute_one_centroid(k)
                else:
                    to_remove.append(k)
        
        self.pop_labels_from_cluster(to_remove)
        
        self.arrays.flush_array('all_peaks')
        self.arrays.flush_array('clusters')
        
    
    def split_cluster(self, label, method='kmeans',  **kargs):
        """
        This split one cluster by applying a new clustering method only on the subset.
        """
        mask = self.all_peaks['cluster_label']==label
        self.find_clusters(method=method, selection=mask, **kargs)
    
    def auto_split_cluster(self, **kargs):
        cleancluster.auto_split(self, n_spike_for_centroid=self.n_spike_for_centroid, n_jobs=self.n_jobs, **kargs)
    
    def trash_not_aligned(self, **kargs):
        cleancluster.trash_not_aligned(self, **kargs)
        
    def auto_merge_cluster(self, **kargs):
        cleancluster.auto_merge(self, **kargs)
    
    def trash_low_extremum(self, **kargs):
        cleancluster.trash_low_extremum(self, **kargs)
    
    def trash_small_cluster(self, **kargs):
        cleancluster.trash_small_cluster(self, **kargs)

    def compute_cluster_similarity(self, method='cosine_similarity_with_max'):
        if self.centroids_median is None:
            self.compute_all_centroid()
        
        #~ t1 = time.perf_counter()
        
        labels = self.cluster_labels
        mask = labels>=0
        
        wfs = self.centroids_median[mask, :,  :]
        wfs = wfs.reshape(wfs.shape[0], -1)
        
        if wfs.size == 0:
            cluster_similarity = None
        else:
            cluster_similarity = metrics.cosine_similarity_with_max(wfs)

        if cluster_similarity is None:
            self.arrays.detach_array('cluster_similarity')
            self.cluster_similarity = None
        else:
            self.arrays.add_array('cluster_similarity', cluster_similarity.astype('float32'), self.memory_mode)

        #~ t2 = time.perf_counter()
        #~ print('compute_cluster_similarity', t2-t1)

            

    def detect_high_similarity(self, threshold=0.95):
        if self.cluster_similarity is None:
            self.compute_cluster_similarity()
        pairs = get_pairs_over_threshold(self.cluster_similarity, self.positive_cluster_labels, threshold)
        return pairs

    def auto_merge_high_similarity(self, threshold=0.95):
        """Recursively merge all pairs with similarity hihger that a given threshold
        """
        pairs = self.detect_high_similarity(threshold=threshold)
        already_merge = {}
        for k1, k2 in pairs:
            # merge if k2 still exists
            if k1 in already_merge:
                k1 = already_merge[k1]
            #~ print('auto_merge', k1, 'with', k2)
            mask = self.all_peaks['cluster_label'] == k2
            self.all_peaks['cluster_label'][mask] = k1
            already_merge[k2] = k1
            self.pop_labels_from_cluster([k2])

    def compute_cluster_ratio_similarity(self, method='cosine_similarity_with_max'):
        #~ print('compute_cluster_ratio_similarity')
        if self.centroids_median is None:
            self.compute_all_centroid()
        
        #~ if not hasattr(self, 'centroids'):
            #~ self.compute_centroid()            
        
        #~ t1 = time.perf_counter()
        labels = self.positive_cluster_labels
        
        #TODO: this is stupid because cosine_similarity is the same at every scale!!!
        #so this is identique to compute_similarity()
        #find something else
        wf_normed = []
        for ind, k in enumerate(self.clusters['cluster_label']):
            if k<0: continue
            #~ chan = self.centroids[k]['extremum_channel']
            chan = self.clusters['extremum_channel'][ind]
            #~ median = self.centroids[k]['median']
            median = self.centroids_median[ind, :, :]
            n_left = int(self.info['waveform_extractor_params']['n_left'])
            wf_normed.append(median/np.abs(median[-n_left, chan]))
        wf_normed = np.array(wf_normed)
        
        if wf_normed.size == 0:
            cluster_ratio_similarity = None
        else:
            wf_normed_flat = wf_normed.swapaxes(1, 2).reshape(wf_normed.shape[0], -1)
            #~ cluster_ratio_similarity = metrics.compute_similarity(wf_normed_flat, 'cosine_similarity')
            cluster_ratio_similarity = metrics.cosine_similarity_with_max(wf_normed_flat)

        if cluster_ratio_similarity is None:
            self.arrays.detach_array('cluster_ratio_similarity')
            self.cluster_ratio_similarity = None
        else:
            self.arrays.add_array('cluster_ratio_similarity', cluster_ratio_similarity.astype('float32'), self.memory_mode)
        #~ return labels, ratio_similarity, wf_normed_flat


        #~ t2 = time.perf_counter()
        #~ print('compute_cluster_ratio_similarity', t2-t1)        

    def detect_similar_waveform_ratio(self, threshold=0.9):
        if self.cluster_ratio_similarity is None:
            self.compute_cluster_ratio_similarity()
        pairs = get_pairs_over_threshold(self.cluster_ratio_similarity, self.positive_cluster_labels, threshold)
        return pairs
    
    
    
    def compute_spike_silhouette(self, size_max=1e7):
        #~ t1 = time.perf_counter()
        
        spike_silhouette = None
        if self.some_peaks_index is not None:
            wf = self.get_some_waveforms(peaks_index=self.some_peaks_index)
            wf = wf.reshape(wf.shape[0], -1)
            labels = self.all_peaks['cluster_label'][self.some_peaks_index]
            if wf.size<size_max:
                spike_silhouette = metrics.compute_silhouette(wf, labels, metric='euclidean')

        if spike_silhouette is None:
            self.arrays.detach_array('spike_silhouette')
            self.spike_silhouette = None
        else:
            self.arrays.add_array('spike_silhouette', spike_silhouette.astype('float32'), self.memory_mode)


        #~ t2 = time.perf_counter()
        #~ print('compute_spike_silhouette', t2-t1)                
    
    def tag_same_cell(self, labels_to_group):
        """
        In some situation in spike burst the amplitude change baldly.
        In theses cases we can have 2 distinct cluster but the user suspect
        that it is the same cell. In such cases the user must not merge 
        the 2 clusters because the centroids will represent nothing and 
        and  the Peeler.will fail.
        
        Instead we tag the 2 differents cluster as "same cell"
        so same cell_label.
        
        This is a manual action.
        
        """
        inds, = np.nonzero(np.in1d(self.clusters['cluster_label'], labels_to_group))
        self.clusters['cell_label'][inds] = min(labels_to_group)
    

    def order_clusters(self, by='waveforms_rms'):
        """
        This reorder labels from highest rms to lower rms.
        The higher rms the smaller label.
        Negative labels are not reassigned.
        """
        if self.centroids_median is None:
            self.compute_all_centroid()
        
        if np.any(np.isnan(self.clusters[self.clusters['cluster_label']>=0]['waveform_rms'])):
            # MUST compute centroids
            #~ print('MUST compute centroids because nan')
            #~ self.compute_centroid()
            for ind, k in enumerate(self.clusters['cluster_label']):
                if k<0:
                    continue
                if np.isnan(self.clusters[ind]['waveform_rms']):
                    self.compute_one_centroid(k)
        

        clusters = self.clusters.copy()
        neg_clusters = clusters[clusters['cluster_label']<0]
        pos_clusters = clusters[clusters['cluster_label']>=0]
        
        #~ print('order_clusters', by)
        if by=='waveforms_rms':
            order = np.argsort(pos_clusters['waveform_rms'])[::-1]
        elif by=='extremum_amplitude':
            order = np.argsort(np.abs(pos_clusters['extremum_amplitude']))[::-1]
        else:
            raise(NotImplementedError)
        
        sorted_labels = pos_clusters['cluster_label'][order]
        
        
        #reassign labels for peaks and clusters
        if len(sorted_labels)>0:
            N = int(max(sorted_labels)*10)
        else:
            N = 0
        self.all_peaks['cluster_label'][self.all_peaks['cluster_label']>=0] += N
        for new, old in enumerate(sorted_labels+N):
            self.all_peaks['cluster_label'][self.all_peaks['cluster_label']==old] = new
        
        pos_clusters = pos_clusters[order].copy()
        n = pos_clusters.size
        pos_clusters['cluster_label'] = np.arange(n)
        
        #this shoudl preserve previous identique cell_label
        pos_clusters['cell_label'] += N
        for i in range(n):
            k = pos_clusters['cell_label'][i]
            inds, = np.nonzero(pos_clusters['cell_label']==k)
            if (len(inds)>=1) and (inds[0] == i):
                pos_clusters['cell_label'][inds] = i
        
        new_cluster = np.concatenate((neg_clusters, pos_clusters))
        self.clusters[:] = new_cluster
        
        # reasign centroids
        mask_pos= (self.clusters['cluster_label']>=0)
        for name in _centroids_arrays:
            arr = getattr(self, name)
            arr_pos = arr[mask_pos, ].copy() # first dim for all
            arr[mask_pos, ] = arr_pos[order, ]
        
        self.refresh_colors(reset=True)
        
        #~ self._reset_metrics()
        self._reset_arrays(_persistent_metrics)



    
    #~ def compute_nearest_neighbor_similarity(self):
        #~ from .waveformtools import nearest_neighbor_similarity
        
        #~ snn = nearest_neighbor_similarity(self)
    
    def compute_boundaries(self, sparse_thresh_level1=1.0, full_hyperplane=False):
        assert self.some_waveforms is not None, 'run cc.cache_some_waveforms() first'


        keep = self.cluster_labels>=0
        cluster_labels = self.clusters[keep]['cluster_label'].copy()
        centroids = self.centroids_median[keep, :, :].copy()
        
        #~ centroids[np.abs(centroids)<0.5] = 0.

        thresh = self.info['peak_detector_params']['relative_threshold']
        sparse_mask_level1 = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=sparse_thresh_level1)
        #~ sparse_mask_level3 = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=thresh)
        
        #~ share_channel_mask = compute_shared_channel_mask(centroids, self.mode,  sparse_thresh_level1)

        # sparsify centroids
        #~ sparse_centroids = centroids.copy()
        #~ for cluster_idx0, label0 in enumerate(cluster_labels):
            #~ rem = ~sparse_mask_level1[cluster_idx0, :]
            #~ sparse_centroids[cluster_idx0, :, rem] = 0.
            

        #~ flat_centroids = centroids.reshape(centroids.shape[0], -1).T.copy()
        
        n = len(cluster_labels)
        #~ print('n', n)
        
        flat_shape = n, centroids.shape[1] * centroids.shape[2]
        projections = np.zeros(flat_shape, dtype='float32')
        projections_3d = projections.reshape(centroids.shape)
        scalar_products = np.zeros((n+1, n+1), dtype=object)
        boundaries = np.zeros((n, 4), dtype='float32')
        neighbors = {}
        
        for cluster_idx0, label0 in enumerate(cluster_labels):
            #~ if cluster_idx0 != 13:
                #~ continue
            #~ print()
            #~ print('cluster_idx0', cluster_idx0)
            
        
            # all channel
            if full_hyperplane:
                # all centroids / all channelss
                chan_mask = np.ones(self.nb_channel, dtype='bool')
                clus_mask = np.ones(n, dtype='bool')
                local_idx0 = cluster_idx0
            else:
                # only centroids / channel that are on sparse mask
                chan_mask = sparse_mask_level1[cluster_idx0, :]
                
                # case1 sharing chan_mask
                #~ nover = np.sum(sparse_mask_level1[:, chan_mask], axis=1)
                #~ clus_mask = (nover > 0)
                
                # case2 5 nearest template
                #~ distances = np.sum((centroids - centroids[[cluster_idx0], :, :])**2, axis=(1,2))
                #~ order = np.argsort(distances)
                #~ nearest = order[:8]
                #~ clus_mask = np.zeros(n, dtype='bool')
                #~ clus_mask[nearest] = True
                
                # case3 all cluster
                clus_mask = np.ones(n, dtype='bool')
                
                # case 4 : sharing chan_mask + N nearest
                #~ nover = np.sum(sparse_mask_level1[:, chan_mask], axis=1)
                #~ sharing_mask = (nover > 0)
                #~ distances = np.sum((centroids - centroids[[cluster_idx0], :, :])**2, axis=(1,2))
                #~ distances[~sharing_mask] = np.inf
                #~ order = np.argsort(distances)
                #~ nearest = order[:8]
                #~ clus_mask = np.zeros(n, dtype='bool')
                #~ clus_mask[nearest] = True
                
                local_indexes0, = np.nonzero(clus_mask)
                local_idx0 = np.nonzero(local_indexes0 == cluster_idx0)[0][0]
                
            
            local_nclus = np.sum(clus_mask)
            local_chan = np.sum(chan_mask)
            flat_centroids = centroids[clus_mask, :, :][:, :, chan_mask].reshape(local_nclus, -1).T.copy()
            #~ print('local_nclus', local_nclus, 'local_chan', local_chan)
            
            
            flat_centroid0 = flat_centroids[:, local_idx0]
            #~ flat_centroid0 = centroids[cluster_idx0, :, :][:, chan_mask].flatten()
            
            other_mask = np.ones(local_nclus, dtype='bool')
            other_mask[local_idx0] = False
            
            if np.sum(other_mask) > 0:
                other_centroids = flat_centroids[:, other_mask]
                
                # without noise in hyper plan
                #~ shift = -other_centroids[:, 0]
                #~ centerred_other_centroids = other_centroids.copy()
                #~ centerred_other_centroids += shift[:, np.newaxis]
                #~ centerred_other_centroids = centerred_other_centroids[:, 1:]
                #~ q, r = np.linalg.qr(centerred_other_centroids, mode='reduced')
                #~ centroid0_proj = q @ q.T @ (flat_centroid0 + shift) - shift
                #~ ortho_complement = centroid0_proj - flat_centroid0
                #~ ortho_complement /= np.sum(ortho_complement**2)                

                # with noise (center) in hyper plan
                #~ centerred_other_centroids = other_centroids.copy()
                #~ q, r = np.linalg.qr(centerred_other_centroids, mode='reduced')
                #~ centroid0_proj = q @ q.T @ (flat_centroid0)
                #~ ortho_complement = centroid0_proj - flat_centroid0
                #~ ortho_complement /= np.sum(ortho_complement**2)                
                

                #TEST1
                #~ centerred_other_centroids = other_centroids.copy()
                #~ ind_min = np.argmin(np.abs(centerred_other_centroids - flat_centroid0[:, np.newaxis]), axis=1)
                #~ centroid0_proj = centerred_other_centroids[np.arange(centerred_other_centroids.shape[0], dtype='int'), ind_min]
                #TEST2
                #~ centerred_other_centroids = other_centroids.copy()
                #~ ind_min = np.argmin(np.sum((centerred_other_centroids - flat_centroid0[:, np.newaxis])**2, axis=0))
                #~ centroid0_proj = centerred_other_centroids[:, ind_min]
                #TEST3
                #~ fig, ax = plt.subplots()
                #~ dists = np.sum((other_centroids - flat_centroid0[:, np.newaxis])**2, axis=0)
                #~ ax.plot(dists)
                #~ plt.show()
                
                ind_min = np.argmin(np.sum((other_centroids - flat_centroid0[:, np.newaxis])**2, axis=0))
                other_select = [ind_min]
                
                #~ print('ind_min', ind_min, np.nonzero(other_mask)[0][ind_min])
                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(flat_centroid0.reshape(-1, local_chan).T.flatten(), color='k')
                #~ ax.plot(other_centroids[:, ind_min].reshape(-1, local_chan).T.flatten(), color='m')
                #~ plt.show()
                
                #~ last_loop = False
                while True:
                    # TODO : try to add them one by one!!!!!!!
                    if len(other_select) == 1:
                        centroid0_proj = other_centroids[:, other_select[0]]
                    else:
                        #~ shift = -other_centroids[:, 0]
                        centerred_other_centroids = other_centroids[:, other_select].copy()
                        shift = -centerred_other_centroids[:, 0]
                        centerred_other_centroids += shift[:, np.newaxis]
                        centerred_other_centroids = centerred_other_centroids[:, 1:]
                        
                        #~ q, r = np.linalg.qr(centerred_other_centroids, mode='reduced')
                        #~ centroid0_proj = q @ q.T @ (flat_centroid0 + shift) - shift
                        
                        #~ u,s,vh = np.linalg.svd(centerred_other_centroids, full_matrices=False)
                        u,s,vh = scipy.linalg.svd(centerred_other_centroids, full_matrices=False)
                        centroid0_proj = u @ u.T @ (flat_centroid0 + shift) - shift
                        
                        
                    local_projector = centroid0_proj - flat_centroid0
                    local_projector /= np.sum(local_projector**2)                
                    other_feat = (other_centroids - flat_centroid0[:, np.newaxis]).T @ local_projector
                    
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(other_feat)
                    #~ other_idx = np.nonzero(other_mask)[0][other_select]
                    #~ title = f'{cluster_idx0} {other_idx}'
                    #~ ax.set_title(title)
                    #~ ax.set_ylim(-2, 2)
                    #~ plt.show()
                    
                    #~ if last_loop:
                        #~ ind,  = np.nonzero(np.abs(other_feat) < 1.)
                    #~ else:
                        #~ ind,  = np.nonzero(np.abs(other_feat) < 0.5)
                    #~ if np.all(np.in1d(ind, other_select)):
                        #~ if last_loop:
                            #~ break
                        #~ else:
                            #~ last_loop = True
                    #~ other_select.extend(ind)
                    #~ other_select = list(np.unique(other_select))


                    ind,  = np.nonzero(np.abs(other_feat) < 1.)

                    if ind.size == 0:
                        break
                    else:
                        not_in = np.ones(other_feat.size, dtype='bool')
                        not_in[other_select] = False
                        too_small = (np.abs(other_feat) < 1.)
                        others_candidate, = np.nonzero(not_in & too_small)
                        if others_candidate.size ==0:
                            break
                        smallest_ind = np.argmin(np.abs(other_feat[others_candidate]))
                        other_select.append(others_candidate[smallest_ind])
                    
                    
                    
                    
                    #~ print('other_select', other_select)
                    #~ print('other_feat', other_feat)
                
                #~ plt.show()
                
                #~ print('other_select', len(other_select))
                neighbors[cluster_idx0] = np.nonzero(other_mask)[0][other_select]
                
                
                
                #~ exit()
                ortho_complement = local_projector

                    
                    
                    
                    
                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(flat_centroid0.reshape(centroids.shape[1], -1).T.flatten())
                #~ ax.plot(centroid0_proj.reshape(centroids.shape[1], -1).T.flatten())
                #~ plt.show()
                
                #~ centerred_other_centroids = centerred_other_centroids[:, 1:]
                #~ q, r = np.linalg.qr(centerred_other_centroids, mode='reduced')
                #~ centroid0_proj = q @ q.T @ (flat_centroid0 + shift) - shift

                
                
            else:
                # alone one theses channels = make projection with noise
                ortho_complement = 0 - flat_centroid0
                ortho_complement /= np.sum(ortho_complement**2)                
            

            
            
            projections_3d[cluster_idx0, :, :][:, chan_mask] = ortho_complement.reshape(centroids.shape[1], local_chan)
            
            #~ projector = 
            

                
            
            #~ projector = ortho_complement
            projector =projections_3d[cluster_idx0, :][:, chan_mask].flatten()
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(ortho_complement)
            #~ plt.show()
            
            wf0 = self.get_cached_waveforms(label0)
            flat_centroid0 = centroids[cluster_idx0, :, :][:, chan_mask].flatten()
            
            
            wf0 = wf0[:, :, chan_mask].copy()
            flat_wf0 = wf0.reshape(wf0.shape[0], -1)
            feat_wf0 = (flat_wf0 - flat_centroid0) @ projector
            feat_centroid0 = (flat_centroid0 - flat_centroid0) @ projector
            
            #~ print('feat_centroid0', feat_centroid0)
            #~ print(feat_wf0)
            #~ feat_centroid0 = 0
            
            #~ inner_dist = np.sum((feat_wf0 - feat_centroid0)**2, axis=0)
            
            scalar_products[cluster_idx0, cluster_idx0] = feat_wf0
            
            
            for cluster_idx1, label1 in enumerate(cluster_labels):

                #~ channel_shared = share_channel_mask[cluster_idx0, cluster_idx1]
                
                #~ if not channel_shared:
                    #~ continue
                
                #~ print(' cluster_idx1', cluster_idx1, 'label1', label1)
                
                if cluster_idx0 == cluster_idx1:
                    #~ scalar_products[cluster_idx0, cluster_idx1] = None
                    continue
                
                centroid1 = centroids[cluster_idx1, :, :][:, chan_mask]
                wf1 = self.get_cached_waveforms(label1)
                wf1 = wf1[:, :, chan_mask].copy()
                flat_wf1 = wf1.reshape(wf1.shape[0], -1) 
                feat_centroid1 = (centroid1.flatten() - flat_centroid0) @ projector
                feat_wf1 = (flat_wf1- flat_centroid0) @ projector
                
                #~ cross_dist = np.sum((feat_wf1 - feat_centroid0)**2, axis=0)
                
                scalar_products[cluster_idx0, cluster_idx1] = feat_wf1
            
            noise = self.some_noise_snippet
            noise = noise[:, :, chan_mask].copy()
            flat_noise = noise.reshape(noise.shape[0], -1)
            feat_noise = (flat_noise - flat_centroid0) @ projector
            scalar_products[cluster_idx0, -1] = feat_noise
            
            
            
        
        for cluster_idx0, label0 in enumerate(cluster_labels):
            #~ if cluster_idx0 != 13:
                #~ continue
            
            #~ print(cluster_idx0)
            inner_sp = scalar_products[cluster_idx0, cluster_idx0]
            med, mad = median_mad(inner_sp)
            
            mad_factor = 6
            #~ mad_factor = 5
            #~ mad_factor = 4
            #~ mad_factor = 3.5
            #~ mad_factor = 3
            #~ mad_factor = 2.5
            #~ low = np.min(inner_sp)
            #~ low = max(med - mad_factor * mad, -0.5)
            low = med - mad_factor * mad
            initial_low = low
            #~ high = min(med + mad_factor * mad, 0.5)
            high = med + mad_factor * mad
            initial_high = high

            # method with factor
            #~ for cluster_idx1, label1 in enumerate(cluster_labels):
                #~ if cluster_idx1 == cluster_idx0:
                    #~ continue
                #~ cross_sp = scalar_products[cluster_idx0, cluster_idx1]
                #~ med, mad = median_mad(cross_sp)
                #~ if med > high and (med - mad_factor * mad) < high:
                    #~ middle = (initial_high + (med - mad_factor * mad)) * 0.5
                    #~ high = middle
                
                #~ if med < low and (med + mad_factor * mad) > low:
                    #~ middle = (low + (med + mad_factor * mad)) * 0.5
                    #~ low = middle
            #~ noise_sp = scalar_products[cluster_idx0, -1]
            #~ med, mad = median_mad(noise_sp)
            #~ if (med - mad_factor * mad) < high:
                #~ ## middle = (initial_high + (med - mad_factor * mad)) * 0.5
                #~ high = med - mad_factor * mad
            
            #~ boundaries[cluster_idx0, 0] = max(low, -0.5)
            #~ boundaries[cluster_idx0, 1] = min(high, 0.5)
            
            #~ boundaries[cluster_idx0, 2] = max(initial_low, -0.5)
            #~ boundaries[cluster_idx0, 3] = min(initial_high, 0.5)                    

            #~ if high <0:
                #~ # too complicated
                #~ print('warning boundary label=', cluster_labels[cluster_idx0], 'cluster_idx0=', cluster_idx0)
                #~ boundaries[cluster_idx0, 0] = 0.
                #~ boundaries[cluster_idx0, 1] = 0.
            
            # optimze boudaries with accuracy
            
            high_clust = []
            high_lim = []
            low_clust = []
            low_lim = []
            for cluster_idx1, label1 in enumerate(cluster_labels):
                # select dangerous cluster
                if cluster_idx1 == cluster_idx0:
                    continue
                cross_sp = scalar_products[cluster_idx0, cluster_idx1]
                med, mad = median_mad(cross_sp)
                if med > high and (med - mad_factor * mad) < high :
                    high_clust.append(cluster_idx1)
                    high_lim.append(med - mad_factor * mad)
                if med < low and (med + mad_factor * mad) > low:
                    low_clust.append(cluster_idx1)
                    low_lim.append(med + mad_factor * mad)

            noise_sp = scalar_products[cluster_idx0, -1]
            med, mad = median_mad(noise_sp)
            if med > high and (med - mad_factor * mad) < high :
                high_clust.append(-1)
                high_lim.append(med - mad_factor * mad)
            # TODO if noise in low limits
            #~ print()
            #~ print('initial_low', initial_low)
            #~ print('initial_high', initial_high)
            #~ print('high_clust', high_clust, 'high_lim', high_lim)
            #~ print('low_clust', low_clust, 'low_lim', low_lim)

            if len(high_clust) > 0:
                l0 = min(high_lim)
                l1 = initial_high
                step = (l1-l0)/20.
                
                all_sp = np.concatenate([scalar_products[cluster_idx0, idx1] for idx1 in high_clust])
                limits = np.arange(l0, l1, step) 
                accuracies = []
                for l in limits:
                    tp = scalar_products[cluster_idx0, cluster_idx0].size
                    fn = np.sum(scalar_products[cluster_idx0, cluster_idx0] > l)
                    fp = np.sum(all_sp < l)
                    accuracy = tp / (tp + fn + fp)
                    accuracies.append(accuracy)
                best_lim = limits[np.argmax(accuracies)]
                boundaries[cluster_idx0, 1] = min(best_lim, 0.5)
                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(limits, accuracies)
                #~ ax.axvline(best_lim)
                #~ ax.set_title(f'high {cluster_idx0}')
                #~ plt.show()
                
            else:
                boundaries[cluster_idx0, 1] = min(initial_high, 0.5)
            
            if len(low_clust) > 0:
                l1 = max(low_lim)
                l0 = initial_low
                step = (l1-l0)/20.
                
                all_sp = np.concatenate([scalar_products[cluster_idx0, idx1] for idx1 in low_clust])
                limits = np.arange(l0, l1, step) 
                accuracies = []
                for l in limits:
                    tp = scalar_products[cluster_idx0, cluster_idx0].size
                    fn = np.sum(scalar_products[cluster_idx0, cluster_idx0] <l)
                    fp = np.sum(all_sp > l)
                    accuracy = tp / (tp + fn + fp)
                    accuracies.append(accuracy)
                best_lim = limits[np.argmax(accuracies)]
                boundaries[cluster_idx0, 0] = max(best_lim, -0.5)
                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(limits, accuracies)
                #~ ax.axvline(best_lim)
                #~ ax.set_title(f'low {cluster_idx0}')
                #~ plt.show()

            else:
                boundaries[cluster_idx0, 0] = max(initial_low, -0.5)

            
            boundaries[cluster_idx0, 2] = max(initial_low, -0.5)
            boundaries[cluster_idx0, 3] = min(initial_high, 0.5)                    
            

            

        
        #~ if True:
        if False:
            
            colors_ = sns.color_palette('husl', n)
            colors = {i: colors_[i] for i, k in enumerate(cluster_labels)}
            for cluster_idx0, label0 in enumerate(cluster_labels):
                #~ if  cluster_idx0 != 13:
                    #~ continue
                colors_ = sns.color_palette('husl', n)
                colors = {i: colors_[i] for i, k in enumerate(cluster_labels)}
                
                print()
                print(cluster_idx0)
                print(neighbors[cluster_idx0])
                
                
                inner_sp = scalar_products[cluster_idx0, cluster_idx0]
                fig, ax = plt.subplots()
                count, bins = np.histogram(inner_sp, bins=150, density=True)
                ax.plot(bins[:-1], count, color=colors[cluster_idx0])
                low = boundaries[cluster_idx0, 0]
                high = boundaries[cluster_idx0, 1]
                ax.axvline(low, color='k')
                ax.axvline(high, color='k')

                low = boundaries[cluster_idx0, 2]
                high = boundaries[cluster_idx0, 3]
                ax.axvline(low, color='grey', ls='--')
                ax.axvline(high, color='grey', ls='--')
                
                
                #~ fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

                for cluster_idx1, label1 in enumerate(cluster_labels):
                    if cluster_idx1 == cluster_idx0:
                        continue
                    #~ if not cluster_idx1 in (2,3):
                        #~ continue
                    

                    #~ channel_shared = share_channel_mask[cluster_idx0, cluster_idx1]
                    
                    #~ if not channel_shared:
                        #~ continue

                    cross_sp = scalar_products[cluster_idx0, cluster_idx1]
                    count, bins = np.histogram(cross_sp, bins=150)
                    ax.plot(bins[:-1], count, color=colors[cluster_idx1])
                    
                    #~ ax2.plot(centroids[cluster_idx1, :, :].T.flatten(),  color=colors[cluster_idx1])
                    
                
                feat_noise = scalar_products[cluster_idx0, -1]
                count, bins = np.histogram(feat_noise, bins=150, density=True)
                ax.plot(bins[:-1], count, color='k')
                
                
                #~ share_channel_clus, = np.nonzero(share_channel_mask[cluster_idx0])
                nearest = neighbors[cluster_idx0]
                title = f'scalar_products {cluster_idx0} {nearest}'
                ax.set_title(title)
                
                #~ shape = centroids[cluster_idx0, :, :].shape
                #~ ax1.plot(projections[cluster_idx0, :].reshape(shape).T.flatten(), color='b')
                #~ ax2.plot(centroids[cluster_idx0, :, :].T.flatten(), color='k', ls='--')
                
                
                #~ plt.show()
            plt.show()

            
            
        return projections, boundaries
        
        

    def compute_boundaries_old(self, sparse_thresh_level1=1.5, n_nearest=5):
        assert self.some_waveforms is not None, 'run cc.cache_some_waveforms() first'
        
        
        n_left = self.info['waveform_extractor_params']['n_left']
        n_right = self.info['waveform_extractor_params']['n_right']
        peak_sign = self.info['peak_detector_params']['peak_sign']

        keep = self.cluster_labels>=0
        
        cluster_labels = self.clusters[keep]['cluster_label'].copy()
        
        #~ centers0 = np.zeros((len(cluster_labels), catalogue_width, self.nb_channel), dtype=self.info['internal_dtype'])
        centroids = self.centroids_median[keep, :, :].copy()

        thresh = self.info['peak_detector_params']['relative_threshold']
        sparse_mask_level1 = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=sparse_thresh_level1)
        sparse_mask_level3 = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=thresh)


        
        # sparsify centroids
        sparse_centroids = centroids.copy()
        for cluster_idx0, label0 in enumerate(cluster_labels):
            rem = ~sparse_mask_level1[cluster_idx0, :]
            sparse_centroids[cluster_idx0, :, rem] = 0.


        #~ fig, ax = plt.subplots()
        #~ ax.plot(centroids.swapaxes(1,2).reshape(centroids.shape[0], -1).T)
        #~ fig, ax = plt.subplots()
        #~ ax.plot(sparse_centroids.swapaxes(1,2).reshape(sparse_centroids.shape[0], -1).T)

        #~ fig, ax = plt.subplots()
        #~ ax.plot(centroids[-1].T.flatten())
        #~ ax.plot(sparse_centroids[-1].T.flatten())

        #~ plt.show()        

        flat_centroids = centroids.reshape(centroids.shape[0], -1).T.copy()
        mean_centroid = np.mean(flat_centroids, axis=1)
        #~ mean_centroid[:] = 0 # debug
        print('mean_centroid.Shape', mean_centroid.shape)
        flat_centroids -= mean_centroid[:, np.newaxis]
        print('flat_centroids.Shape', flat_centroids.shape)
        #~ u,s,vh = np.linalg.svd(flat_centroids)
        #~ print(u.shape, s.shape, vh.shape)
        u,s,vh = np.linalg.svd(flat_centroids, full_matrices=False)
        print(u.shape, s.shape, vh.shape)
        
        projector = u[:, :centroids.shape[0]].copy()
        print('projector.shape', projector.shape)
        #~ exit()
        

        n = len(cluster_labels)
        
        scalar_products = np.zeros((n+1, n+1), dtype=object)
        feat_scalar_products = np.zeros((n+1, n+1), dtype=object)
        feat_distances = np.zeros((n+1, n+1), dtype=object)
        wf_distances = np.zeros((n+1, n+1), dtype=object)
        
        
        feat_centroids = (centroids.reshape(centroids.shape[0], -1) - mean_centroid) @ projector
        #~ print(feat_centroids.shape)
        
        
        #~ share_channel_mask = np.zeros((n, n), dtype='bool')
        share_channel_mask = compute_shared_channel_mask(centroids, self.mode,  sparse_thresh_level1)
        #~ fig, ax = plt.subplots()
        #~ ax.matshow(share_channel_mask)
        #~ plt.show()
        
        n_nearest_idx = []
        for cluster_idx0, label0 in enumerate(cluster_labels):
            
            print('cluster_idx0', cluster_idx0)
            
            centroid0 = centroids[cluster_idx0, :, :]
            
            #~ feat_centroid0 = (centroid0.flatten() - mean_centroid) @ projector
            feat_centroid0 = feat_centroids[cluster_idx0, :]
            
            
            feat_dist_to_other = np.sum((feat_centroids - feat_centroid0)**2, axis=1)
            nearests = np.argsort(feat_dist_to_other)
            n_nearest_idx.append(nearests[1:n_nearest])
            
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(feat_centroid0)
            #~ ax.plot(feat_centroids[cluster_idx0, :], color='k', ls='--')
            #~ plt.show()
            
            
            
            
            # sample waveforms
            wf0 = self.get_cached_waveforms(label0)
            feat_wf0 = (wf0.reshape(wf0.shape[0], -1) - mean_centroid) @ projector
            
            inner_feat_dist = np.sum((feat_wf0 - feat_centroid0)**2, axis=1)
            feat_distances[cluster_idx0, cluster_idx0] = inner_feat_dist
            
            inner_wf_dist = np.sum((wf0 - centroid0)**2, axis=(1, 2))
            wf_distances[cluster_idx0, cluster_idx0] = inner_wf_dist
            
            for cluster_idx1, label1 in enumerate(cluster_labels):

                #~ channel_shared = share_channel_mask[cluster_idx0, cluster_idx1]
                #~ if not channel_shared:
                    #~ continue
                if cluster_idx1 not in n_nearest_idx[cluster_idx0]:
                    continue
                
                #~ print(' cluster_idx1', cluster_idx1, 'label1', label1)
                
                if cluster_idx0 == cluster_idx1:
                    #~ scalar_products[cluster_idx0, cluster_idx1] = None
                    continue
                
                centroid1 = centroids[cluster_idx1, :, :]
                wf1 = self.get_cached_waveforms(label1)
                
                #~ feat_centroid1 = (centroid1.flatten() - mean_centroid) @ projector
                feat_centroid1 = feat_centroids[cluster_idx1, :]
                feat_wf1 = (wf1.reshape(wf1.shape[0], -1) - mean_centroid) @ projector
                
                
                mask = sparse_mask_level3[cluster_idx0, :]
                vector_0_1 = (centroid1 - centroid0)
                vector_0_1 =vector_0_1[:, mask]
                vector_0_1 /= np.sum(vector_0_1**2)
                
                inner_sp = np.sum((wf0[:,:,mask] - centroid0[np.newaxis,:,:][:,:,mask]) * vector_0_1[np.newaxis,:,:], axis=(1,2))
                cross_sp = np.sum((wf1[:,:,mask] - centroid0[np.newaxis,:,:][:,:,mask]) * vector_0_1[np.newaxis,:,:], axis=(1,2))


                feat_vector_0_1 = (feat_centroid1 - feat_centroid0)
                feat_vector_0_1 /= np.sum(feat_vector_0_1**2)

                inner_feat_sp = np.sum((feat_wf0 - feat_centroid0[np.newaxis,:]) * feat_vector_0_1[np.newaxis,:], axis=1)
                cross_feat_sp = np.sum((feat_wf1 - feat_centroid0[np.newaxis,:]) * feat_vector_0_1[np.newaxis,:], axis=1)
                
                cross_feat_dist = np.sum((feat_wf1 - feat_centroid0)**2, axis=1)
                cross_wf_dist = np.sum((wf1 - centroid0)**2, axis=(1, 2))
                
                scalar_products[cluster_idx0, cluster_idx1] = (inner_sp, cross_sp)
                feat_scalar_products[cluster_idx0, cluster_idx1] = (inner_feat_sp, cross_feat_sp)
                feat_distances[cluster_idx0, cluster_idx1] = cross_feat_dist
                wf_distances[cluster_idx0, cluster_idx1] = cross_wf_dist
        
        
        feat_distance_boundaries = np.zeros(n, dtype='float32')
        nearest_idx = []
        for cluster_idx0, label0 in enumerate(cluster_labels):
            d = feat_distances[cluster_idx0, cluster_idx0] 
            inner_lim_max = np.quantile(d, 0.9)
            
            low_limits = []
            for cluster_idx1, label1 in enumerate(cluster_labels):
                
                if cluster_idx0 == cluster_idx1:
                    l = np.inf
                #~ elif not share_channel_mask[cluster_idx0, cluster_idx1]:
                elif cluster_idx1 not in n_nearest_idx[cluster_idx0]:
                    l = np.inf
                else:
                    d = feat_distances[cluster_idx0, cluster_idx1] 
                    l = np.quantile(d, 0.1)
                low_limits.append(l)
            print()
            print('low_limits', low_limits)
            nearest = np.argmin(low_limits)
            print('nearest', nearest)
            nearest_idx.append(nearest)
            #~ share_channel_clus, = np.nonzero(share_channel_mask[cluster_idx0])
            #~ print('share_channel_clus', share_channel_clus)
            
            print('cluster_idx0', cluster_idx0, 'nearest', nearest)
            #~ if low_limits[nearest] > inner_lim_max:
                #~ print('ok')
                #~ lim = (inner_lim_max + low_limits[nearest]) / 2.
            #~ else:
                #~ # TODO find something better
                #~ lim = low_limits[nearest]
            #~ if low_limits[nearest] > inner_lim_max:
                #~ print('ok')
                #~ lim = inner_lim_max + (low_limits[nearest] - inner_lim_max) * 0.1
                #~ lim = inner_lim_max
            #~ else:
                # TODO find something better
                #~ lim = low_limits[nearest]
            
            lim = np.sum(( feat_centroids[cluster_idx0, :]-feat_centroids[nearest, :])**2) / 2.

            

            feat_distance_boundaries[cluster_idx0] = lim
            
        #~ print(scalar_products)
        

        #~ if True:
        if False:
            
            colors_ = sns.color_palette('husl', n)
            colors = {i: colors_[i] for i, k in enumerate(cluster_labels)}
            for cluster_idx0, label0 in enumerate(cluster_labels):
                
                #~ if not cluster_idx0 in (12, 13):
                    #~ continue
                
                #~ share_channel_clus, = np.nonzero(share_channel_mask[cluster_idx0])
                n_nearest = n_nearest_idx[cluster_idx0]
                nearest = nearest_idx[cluster_idx0]
                title = f' cluster_idx {cluster_idx0} {n_nearest} {nearest}'
                
                fig, ax1 = plt.subplots()
                ax1.set_title('projection2by2' + title)
                
                fig, ax2 = plt.subplots()
                ax2.set_title('projection2by2' + title)

                fig, ax3 = plt.subplots()
                ax3.set_title('distance feat' + title)
                lim = feat_distance_boundaries[cluster_idx0]
                ax3.axvline(lim, color='k', lw=3)

                fig, ax4 = plt.subplots()
                ax4.set_title('distance wf' + title)

                
                for cluster_idx1, label1 in enumerate(cluster_labels):
                    
                    #~ if not share_channel_mask[cluster_idx0, cluster_idx1]:
                        #~ continue
                    if cluster_idx1 not in n_nearest_idx[cluster_idx0] and cluster_idx0 != cluster_idx1:
                        continue


                    feat_dist = feat_distances[cluster_idx0, cluster_idx1]
                    count, bins = np.histogram(feat_dist, bins=150)
                    ax3.plot(bins[:-1], count, color=colors[cluster_idx1])

                    feat_centroid0 = feat_centroids[cluster_idx0, :]
                    feat_centroid1 = feat_centroids[cluster_idx1, :]
                    dist_feat_centroid = np.sum((feat_centroid1-feat_centroid0)**2)
                    ax3.axvline(dist_feat_centroid, color=colors[cluster_idx1])
                    

                    wf_dist = wf_distances[cluster_idx0, cluster_idx1]
                    count, bins = np.histogram(wf_dist, bins=150)
                    ax4.plot(bins[:-1], count, color=colors[cluster_idx1])
                    
                    
                    if cluster_idx0 == cluster_idx1:
                        continue
                    
                    
                    inner_sp, cross_sp = scalar_products[cluster_idx0, cluster_idx1]
                    count, bins = np.histogram(inner_sp, bins=150)
                    ax1.plot(bins[:-1], count, color=colors[cluster_idx1])
                    count, bins = np.histogram(cross_sp, bins=150)
                    ax1.plot(bins[:-1], count, color=colors[cluster_idx1])
                    
                    
                    

                    inner_sp, cross_sp = feat_scalar_products[cluster_idx0, cluster_idx1]
                    count, bins = np.histogram(inner_sp, bins=150)
                    ax2.plot(bins[:-1], count, color=colors[cluster_idx1])
                    count, bins = np.histogram(cross_sp, bins=150)
                    ax2.plot(bins[:-1], count, color=colors[cluster_idx1])

                    
                    
                plt.show()


        #~ return distance_limit
        
        return mean_centroid, projector, feat_centroids, feat_distance_boundaries
    

    def make_catalogue(self,
                            inter_sample_oversampling=False,
                            #~ inter_sample_oversampling=True,
                            subsample_ratio='auto',
                            sparse_thresh_level1=1.,
                            sparse_thresh_level2=3,
                            sparse_channel_count=7,
                            #~ sparse_thresh_level2=5,
                            
                            #~ sparse_thresh_extremum=-5,
                            ):
        #TODO: offer possibility to resample some waveforms or choose the number
        #~ return
        #~ print('inter_sample_oversampling', inter_sample_oversampling)


        
        
        t1 = time.perf_counter()
        
        
        n_left = self.info['waveform_extractor_params']['n_left']
        n_right = self.info['waveform_extractor_params']['n_right']
        n_left_long = self.info['waveform_extractor_params']['n_left_long']
        n_right_long = self.info['waveform_extractor_params']['n_right_long']
        peak_sign = self.info['peak_detector_params']['peak_sign']
        
        self.catalogue = {}
        self.catalogue['chan_grp'] = self.chan_grp
        self.catalogue['inter_sample_oversampling'] = inter_sample_oversampling
        self.catalogue['mode'] = self.mode
        
        self.catalogue['n_left'] = n_left
        self.catalogue['n_right'] = n_right
        self.catalogue['n_left_long'] = n_left_long
        self.catalogue['n_right_long'] = n_right_long
        #~ self.catalogue['peak_width'] = self.catalogue['n_right'] - self.catalogue['n_left']
        
        
        #for colors
        self.refresh_colors(reset=False)
        
        
        
        # TODO this is redundant with clusters, this shoudl be removed but imply some change in peeler.
        #~ keep = self.cluster_labels>=0
        #order = np.argsort(self.clusters[keep]['waveform_rms'])[::-1]
        #~ cluster_labels = self.clusters[keep]['cluster_label'][order].copy()
        #~ self.catalogue['cluster_labels'] = cluster_labels
        #~ self.catalogue['clusters'] = self.clusters[keep][order].copy()
        #~ # take centroids for positive labems
        #~ keep = (self.cluster_labels >= 0)
        #~ centroids = self.centroids_median[keep, :, :][order,:,:].copy()
        
        keep = self.cluster_labels>=0
        cluster_labels = self.clusters[keep]['cluster_label'].copy()
        self.catalogue['cluster_labels'] = cluster_labels
        self.catalogue['clusters'] = self.clusters[keep].copy()
        centroids = self.centroids_median[keep, :, :].copy()
        centroids_long = self.centroids_median_long[keep, :, :].copy()

        
        
        #~ print(self.cluster_labels)
        #~ print(keep.size, np.sum(keep))
        #~ print(self.centroids_median.shape)
        #~ print(centroids.shape)        
        
        
        #~ n, full_width, nchan = self.some_waveforms.shape
        #~ n = self.some_peaks_index.size
        #~ full_width = self.info['waveform_extractor_params']['n_right'] - self.info['waveform_extractor_params']['n_left']
        
        catalogue_width = n_right - n_left
        catalogue_width_long = n_right_long - n_left_long
        
        full_width = catalogue_width + 4
        #~ nchan = self.nb_channel
        
        centers0 = np.zeros((len(cluster_labels), catalogue_width, self.nb_channel), dtype=self.info['internal_dtype'])
        self.catalogue['centers0'] = centers0 # median of wavforms
        centers0_long = np.zeros((len(cluster_labels), catalogue_width_long, self.nb_channel), dtype=self.info['internal_dtype'])
        self.catalogue['centers0_long'] = centers0_long # median of wavforms
        
        
        # normed and sparse centers0
        #~ centers0_normed = np.zeros((len(cluster_labels), catalogue_width, self.nb_channel), dtype=self.info['internal_dtype'])
        #~ self.catalogue['centers0_normed'] = centers0_normed
        
        

        
        
        #~ self.catalogue['distance_limit'] = np.zeros(len(cluster_labels), dtype=self.info['internal_dtype'])
        
        #~ mean_centroid, projector, feat_centroids, feat_distance_boundaries= self.compute_boundaries(sparse_thresh_level1=sparse_thresh_level1)
        #~ self.catalogue['feat_distance_boundaries']  = feat_distance_boundaries
        #~ self.catalogue['mean_centroid']  = mean_centroid
        #~ self.catalogue['feat_centroids']  = feat_centroids
        #~ self.catalogue['projector']  = projector

        projections, boundaries = self.compute_boundaries(sparse_thresh_level1=sparse_thresh_level1)
        self.catalogue['projections']  = projections
        self.catalogue['boundaries']  = boundaries
        
           
        
        #~ self.catalogue['sp_normed_limit'] = np.zeros((len(cluster_labels),2), dtype=self.info['internal_dtype'])
        
        nb_cluster = cluster_labels.size
        self.catalogue['extremum_channel'] = np.zeros(nb_cluster, dtype='int64')
        self.catalogue['extremum_amplitude'] = np.zeros(nb_cluster, dtype='float32')
        
        #~ self.catalogue['sparse_mask_level1'] = np.zeros((nb_cluster,self.nb_channel),  dtype='bool')
        #~ self.catalogue['sparse_mask_level2'] = np.zeros((nb_cluster,self.nb_channel),  dtype='bool')
        #~ self.catalogue['sparse_mask_level3'] = np.zeros((nb_cluster,self.nb_channel),  dtype='bool')
        #~ self.catalogue['sparse_mask_level4'] = np.zeros((nb_cluster,self.nb_channel),  dtype='bool')

        #~ sparse_mask_level3 = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=thresh)
        #~ sparse_mask_level4 = compute_sparse_mask(centroids, self.mode, method='nbest', nbest=5)
        
        self.catalogue['sparse_mask_level1'] = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=sparse_thresh_level1)
        self.catalogue['sparse_mask_level2'] = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=sparse_thresh_level2)
        thresh = self.info['peak_detector_params']['relative_threshold']
        self.catalogue['sparse_mask_level3'] = compute_sparse_mask(centroids, self.mode, method='thresh', thresh=thresh)
        #~ self.catalogue['sparse_mask_level4'] = compute_sparse_mask(centroids, self.mode, method='nbest', nbest=5)


        # weight for distances
        #~ template_weight = np.std(centroids, axis=0)
        #~ template_weight /= np.sum(template_weight, axis=0)
        #~ template_weight /= np.sum(template_weight)
        #~ self.catalogue['template_weight'] = template_weight
        
        self.catalogue['label_to_index'] = {}
        #~ self.catalogue['nearest_templates'] = {}
        
        
        
        # first loop to compute mask and centers_normed
        for i, k in enumerate(cluster_labels):
            self.catalogue['label_to_index'][k] = i
            center0 = centroids[i, :,:]
            centers0[i,:,:] = center0
            
            center0_long = centroids_long[i, :,:]
            centers0_long[i,:,:] = center0_long
            

            #~ if self.mode == 'dense':
                #~ # see notes in PeelerEngineBase for which sparse mask is for what
                #~ sparse_mask_level1 = np.ones(self.nb_channel, dtype='bool')
                #~ sparse_mask_level2 = np.ones(self.nb_channel, dtype='bool')
                #~ sparse_mask_level3 = np.ones(self.nb_channel, dtype='bool')
            #~ else:
                #~ sparse_mask_level1 = np.any(np.abs(center0) > sparse_thresh_level1, axis=0)
                #~ sparse_mask_level2 = np.any(np.abs(center0)>sparse_thresh_level2, axis=0)
                #~ thresh = self.info['peak_detector_params']['relative_threshold']
                #~ sparse_mask_level3 = np.any(np.abs(center0)>thresh, axis=0)
                #~ sparse_mask_level3 = np.ones(self.nb_channel, dtype='bool') # DEBUG
                #~ sparse_mask_level3 = np.any(np.abs(center0)>sparse_thresh_level2, axis=0) # DEBUG
                
                #~ sparse_mask_level4 = sparse_mask_level2.copy()
                #~ if np.sum(sparse_mask_level4) > sparse_channel_count:
                    #~ best_channel_order = np.argsort(np.max(np.abs(center0), axis=0))[::-1]
                    #~ best_channel_order = best_channel_order[:sparse_channel_count]
                    #~ sparse_mask_level4[:] = False
                    #~ sparse_mask_level4[best_channel_order] = True
            
            #~ self.catalogue['sparse_mask_level1'][i, :] = sparse_mask_level1
            #~ self.catalogue['sparse_mask_level2'][i, :] = sparse_mask_level2
            #~ self.catalogue['sparse_mask_level3'][i, :] = sparse_mask_level3
            #~ self.catalogue['sparse_mask_level4'][i, :] = sparse_mask_level4
            
            #~ sparse_mask_level1 = self.catalogue['sparse_mask_level1'][i, :]
            
            #~ center0_normed = center0.copy()
            #~ center0_normed[:, ~sparse_mask_level1] = 0
            # normalisaton par nrj
            #center0_normed /= np.sum(center0_normed**2)
            # normalisation by extrema
            #~ if peak_sign == '-':
                #~ center0_normed /= np.abs(np.min(center0_normed))
            #~ elif peak_sign == '+':
                #~ center0_normed /= np.abs(np.max(center0_normed))
            #~ center0_normed /= np.sum(center0_normed**2)
            #~ self.catalogue['centers0_normed'][i, :] = center0_normed
            
        #~ fig, ax = plt.subplots()
        #~ ax.plot(centers0_normed.swapaxes(1,2).reshape(centers0_normed.shape[0], -1).T)
        #~ plt.show()
        
        n = len(cluster_labels)
        # dim 3 = quantile 50, quantile 5, quantile 95
        #~ cross_scalar_products = np.zeros((n, n, 3))
        
        
        # loop to get some waveform again to compute:
        #  * derivative if necessary
        if inter_sample_oversampling:
            raise NotImplementedError # TODO propagate center_long to this section and peeler
            centers1 = np.zeros_like(centers0)
            centers2 = np.zeros_like(centers0)
            self.catalogue['centers1'] = centers1 # median of first derivative of wavforms
            self.catalogue['centers2'] = centers2 # median of second derivative of wavforms
        
            if subsample_ratio == 'auto':
                # upsample to 200kHz
                subsample_ratio = int(np.ceil(200000/self.dataio.sample_rate))
                #~ print('subsample_ratio auto', subsample_ratio)
        
            original_time = np.arange(full_width)
            #~ subsample_time = np.arange(1.5, full_width-2.5, 1./subsample_ratio)
            subsample_time = np.arange(0, (full_width-1), 1./subsample_ratio)
        
            self.catalogue['subsample_ratio'] = subsample_ratio
            interp_centers0 = np.zeros((len(cluster_labels), subsample_ratio*catalogue_width, self.nb_channel), dtype=self.info['internal_dtype'])
            self.catalogue['interp_centers0'] = interp_centers0            
            
            for i, k in enumerate(cluster_labels):

                # sample waveforms
                selected, = np.nonzero(self.all_peaks['cluster_label'][self.some_peaks_index]==k)
                if selected.size>self.n_spike_for_centroid:
                    keep = np.random.choice(selected.size, self.n_spike_for_centroid, replace=False)
                    selected = selected[keep]
                
                wf0_large = self.get_some_waveforms(peaks_index=self.some_peaks_index[selected], n_left=n_left-2, n_right=n_right+2)
                wf0 = wf0_large[:, 2:-2,:]

                # compute first and second derivative on dim=1 (time)
                wf1_large = np.zeros_like(wf0_large)
                wf1_large[:, 1:-1, :] = (wf0_large[:, 2:,: ] - wf0_large[:, :-2,: ])/2.
                wf2_large = np.zeros_like(wf1_large)
                wf2_large[:, 1:-1, :] = (wf1_large[:, 2:,: ] - wf1_large[:, :-2,: ])/2.
            
                #median and
                #eliminate margin because of border effect of derivative and reshape
                center0_large = np.median(wf0_large, axis=0)
                #~ center0 = center0_large[2:-2, :]
                #~ centers0[i,:,:] = center0
                centers1[i,:,:] = np.median(wf1_large, axis=0)[2:-2, :]
                centers2[i,:,:] = np.median(wf2_large, axis=0)[2:-2, :]
            
                #interpolate centers0 for reconstruction inbetween bsample when jitter is estimated
                f = scipy.interpolate.interp1d(original_time, center0_large, axis=0, kind='cubic', )
                oversampled_center = f(subsample_time)
                
                extremum_channel = self.catalogue['clusters'][i]['extremum_channel']
                
                #find max  channel for each cluster for peak alignement
                ind_max = np.argmax(np.abs(oversampled_center[(-n_left + 1)*subsample_ratio:(-n_left + 3)*subsample_ratio, extremum_channel]))
                ind_max += (-n_left +1)*subsample_ratio
                i1 = int(ind_max + (n_left-0.5) * subsample_ratio)
                i2 = i1 + subsample_ratio*catalogue_width
                interp_centers0[i, :, :] = oversampled_center[i1:i2, :]                
        
        
        #params propagation
        self.catalogue['signal_preprocessor_params'] = dict(self.info['signal_preprocessor_params'])
        self.catalogue['peak_detector_params'] = dict(self.info['peak_detector_params'])
        self.catalogue['clean_peaks_params'] = dict(self.info['clean_peaks_params'])
        self.catalogue['signals_medians'] = np.array(self.signals_medians, copy=True)
        self.catalogue['signals_mads'] = np.array(self.signals_mads, copy=True)
        
        
        #~ t2 = time.perf_counter()
        #~ print('make_catalogue', t2-t1)
        
        return self.catalogue
    
    def make_catalogue_for_peeler(self, catalogue_name='initial', **kargs):
        """
        Make and save catalogue in the working dir for the Peeler.
        
        """
        # DEBUG
        #~ return
        
        self.make_catalogue(**kargs)
        self.dataio.save_catalogue(self.catalogue, name=catalogue_name)
        
    def create_savepoint(self, name=None):
        """this create a copy of the entire catalogue_constructor subdir
        Usefull for the UI when the user wants to snapshot and try tricky merge/split.
        """
        if name is None:
            name = '{:%Y-%m-%d_%Hh%Mm%S}'.format(datetime.datetime.now())
        copy_path = self.catalogue_path + '_SAVEPOINT_' + name
        
        if not os.path.exists(copy_path):
            shutil.copytree(self.catalogue_path, copy_path)
            
        return copy_path

    def apply_all_steps(self, params, verbose=True):
        """
        
        """
        apply_all_catalogue_steps(self, params, verbose=verbose)
