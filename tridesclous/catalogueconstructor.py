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

import numpy as np
import scipy.signal
import scipy.interpolate
import seaborn as sns
sns.set_style("white")

import sklearn

from . import signalpreprocessor
from . import  peakdetector
from . import decomposition
from . import cluster 
from . import metrics

from .tools import median_mad, get_pairs_over_threshold, int32_to_rgba, rgba_to_int32, make_color_dict


from .iotools import ArrayCollection

import matplotlib.pyplot as plt

from . import labelcodes



_persistent_metrics = ('spike_waveforms_similarity', 'cluster_similarity',
                        'cluster_ratio_similarity', 'spike_silhouette')

_centroids_arrays = ('centroids_median', 'centroids_mad', 'centroids_mean', 'centroids_std',)


_reset_after_waveforms_arrays = ('some_features', 'channel_to_features', 'some_noise_snippet',
                'some_noise_index', 'some_noise_features',) + _persistent_metrics + _centroids_arrays

#~ _reset_after_peak_arrays = ('some_peaks_index', 'some_waveforms', 'some_features',
                        #~ 'channel_to_features', 
                        #~ 'some_noise_index', 'some_noise_snippet', 'some_noise_features',
                        #~ ) + _persistent_metrics + _centroids_arrays

_reset_after_peak_arrays = ('some_peaks_index', 'some_waveforms',) + _reset_after_waveforms_arrays



_persitent_arrays = ('all_peaks', 'signals_medians','signals_mads', 'clusters') + \
                _reset_after_peak_arrays


_dtype_peak = [('index', 'int64'), ('cluster_label', 'int64'), ('segment', 'int64'),]

_dtype_cluster = [('cluster_label', 'int64'), ('cell_label', 'int64'), 
            ('max_on_channel', 'int64'), ('max_peak_amplitude', 'float64'),
            ('waveform_rms', 'float64'), ('nb_peak', 'int64'), 
            ('tag', 'U16'), ('annotations', 'U32'), ('color', 'uint32')]

_keep_cluster_attr_on_new = ['cell_label', 'tag','annotations', 'color']





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
        
        self.catalogue_path = os.path.join(self.dataio.channel_group_path[chan_grp], name)
        
        if not os.path.exists(self.catalogue_path):
            os.mkdir(self.catalogue_path)
        
        self.arrays = ArrayCollection(parent=self, dirname=self.catalogue_path)
        
        self.info_filename = os.path.join(self.catalogue_path, 'info.json')
        if not os.path.exists(self.info_filename):
            #first init
            self.info = {}
            self.flush_info()
        else:
            with open(self.info_filename, 'r', encoding='utf8') as f:
                self.info = json.load(f)
        
        
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
    
    def __repr__(self):
        t = "CatalogueConstructor\n"
        t += '  ' + self.dataio.channel_group_label(chan_grp=self.chan_grp) + '\n'
        if self.all_peaks is None:
            t += '  Signal pre-processing not done yet'
            return t
        
        #~ t += "  nb_peak: {}\n".format(self.nb_peak)
        nb_peak_by_segment = [ np.sum(self.all_peaks['segment']==i)  for i in range(self.dataio.nb_segment)]
        t += '  nb_peak_by_segment: '+', '.join('{}'.format(n) for n in nb_peak_by_segment)+'\n'

        if self.some_waveforms is not None:
            t += '  some_waveforms.shape: {}\n'.format(self.some_waveforms.shape)
            
        if self.some_features is not None:
            t += '  some_features.shape: {}\n'.format(self.some_features.shape)
        
        if hasattr(self, 'cluster_labels'):
            t += '  cluster_labels {}\n'.format(self.cluster_labels)
        
        
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

    def set_preprocessor_params(self, chunksize=1024,
    
            memory_mode='memmap',
            
            internal_dtype = 'float32',
            
            #signal preprocessor
            signalpreprocessor_engine='numpy',
            highpass_freq=300., 
            lowpass_freq=None,
            smooth_size=0,
            common_ref_removal=False,
            
            lostfront_chunksize=None,
            
            
            #peak detector
            peakdetector_engine='numpy',
            peak_sign='-', relative_threshold=7, peak_span_ms=0.3,
            
            ):
        """
        Set parameters for the preprocessor engine
        
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
        signalpreprocessor_engine='numpy' or 'opencl'
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
        peakdetector_engine: 'numpy' or 'opencl'
            Engine for peak detection.
        peak_sign: '-' or '+'
            Signa of peak.
        relative_threshold: int default 7
            Threshold for peak detection. The preprocessed signal have units
            expressed in MAD (robust STD). So 7 is MAD*7.
        peak_span_ms: float default 0.3
            Peak span to avoid double detection. In millisecond.
        """

        
        self._reset_arrays(_persitent_arrays)
            
        
        self.chunksize = chunksize
        self.memory_mode = memory_mode
        
        # set default lostfront_chunksize if none is provided
        if lostfront_chunksize is None or lostfront_chunksize==0:
            assert highpass_freq is not None, 'lostfront_chunksize=None needs a highpass_freq'
            lostfront_chunksize = int(self.dataio.sample_rate/highpass_freq*3)
        
        #~ self.arrays.initialize_array('all_peaks', self.memory_mode,  _dtype_peak, (-1, ))
        
        self.signal_preprocessor_params = dict(highpass_freq=highpass_freq, lowpass_freq=lowpass_freq, 
                        smooth_size=smooth_size, common_ref_removal=common_ref_removal,
                        lostfront_chunksize=lostfront_chunksize, output_dtype=internal_dtype,
                        signalpreprocessor_engine=signalpreprocessor_engine)
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[signalpreprocessor_engine]
        self.signalpreprocessor = SignalPreprocessor_class(self.dataio.sample_rate, self.nb_channel, chunksize, self.dataio.source_dtype)
        
        
        self.peak_detector_params = dict(peak_sign=peak_sign, relative_threshold=relative_threshold, peak_span_ms=peak_span_ms)
        PeakDetector_class = peakdetector.peakdetector_engines[peakdetector_engine]
        self.peakdetector = PeakDetector_class(self.dataio.sample_rate, self.nb_channel,
                                                        self.chunksize, internal_dtype)
        
        #TODO make processed data as int32 ???
        for i in range(self.dataio.nb_segment):
            self.dataio.reset_processed_signals(seg_num=i, chan_grp=self.chan_grp, dtype=internal_dtype)
        
        #~ self.nb_peak = 0
        
        # put all params in info
        self.info['internal_dtype'] = internal_dtype
        self.info['chunksize'] = chunksize
        self.info['signal_preprocessor_params'] = self.signal_preprocessor_params
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
        params2.pop('signalpreprocessor_engine')
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
        

    def signalprocessor_one_chunk(self, pos, sigs_chunk, seg_num, detect_peak=True):

        pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        if preprocessed_chunk is  None:
            return
        
        self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num, chan_grp=self.chan_grp,
                        i_start=pos2-preprocessed_chunk.shape[0], i_stop=pos2, signal_type='processed')
        
        if detect_peak:
            n_peaks, chunk_peaks = self.peakdetector.process_data(pos2, preprocessed_chunk)
            
            if chunk_peaks is not None:
                peaks = np.zeros(chunk_peaks.size, dtype=_dtype_peak)
                peaks['index'] = chunk_peaks
                peaks['segment'][:] = seg_num
                peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
                self.arrays.append_chunk('all_peaks',  peaks)
    
    
    def run_signalprocessor_loop_one_segment(self, seg_num=0, duration=60., detect_peak=True):
        

        
        length = int(duration*self.dataio.sample_rate)
        length = min(length, self.dataio.get_segment_length(seg_num))
        length -= length%self.chunksize

        #TODO make this by segment
        self.info['processed_length'] = length
        self.flush_info()
        
        #initialize engines
        
        p = dict(self.signal_preprocessor_params)
        p.pop('signalpreprocessor_engine')
        p['normalize'] = True
        p['signals_medians'] = self.signals_medians
        p['signals_mads'] = self.signals_mads
        self.signalpreprocessor.change_params(**p)
        
        self.peakdetector.change_params(**self.peak_detector_params)
        
        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=self.chan_grp, chunksize=self.chunksize, i_stop=length,
                                                    signal_type='initial')
        for pos, sigs_chunk in iterator:
            #~ print(seg_num, pos, sigs_chunk.shape)
            self.signalprocessor_one_chunk(pos, sigs_chunk, seg_num, detect_peak=detect_peak)
            
            #maybe flush at each loop to avoid memory up but make it slower
            #~ self.dataio.flush_processed_signals(seg_num=seg_num, chan_grp=self.chan_grp)
    
    
    def finalize_signalprocessor_loop(self):
        self.arrays.finalize_array('all_peaks')
        #~ self._reset_waveform_and_features()
        self._reset_arrays(_reset_after_peak_arrays)
        self.on_new_cluster()
    
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
            duration in seconds for each segment
        detect_peak: bool (default True)
            Also detect peak.
        
        """
        self.arrays.initialize_array('all_peaks', self.memory_mode,  _dtype_peak, (-1, ))
        #~ for i in range(self.dataio.nb_segment):
            #~ self.dataio.reset_processed_signals(seg_num=i, chan_grp=self.chan_grp, dtype=internal_dtype)
        
        
        for seg_num in range(self.dataio.nb_segment):
            self.run_signalprocessor_loop_one_segment(seg_num=seg_num, duration=duration, detect_peak=detect_peak)
            self.dataio.flush_processed_signals(seg_num=seg_num, chan_grp=self.chan_grp)
            
        self.finalize_signalprocessor_loop()
    
    def re_detect_peak(self, peakdetector_engine='numpy', peak_sign='-', relative_threshold=7, peak_span_ms=0.3):
        """
        Peak are detected while **run_signalprocessor**.
        But in some case for testing other threshold we can **re-detect peak** without signal processing.
        
        Parameters
        ----------
        peakdetector_engine: 'numpy' or 'opencl'
            Engine for peak detection.
        peak_sign: '-' or '+'
            Signa of peak.
        relative_threshold: int default 7
            Threshold for peak detection. The preprocessed signal have units
            expressed in MAD (robust STD). So 7 is MAD*7.
        peak_span_ms: float default 0.3
            Peak span to avoid double detection. In second.
        
        """
        #TODO if not peak detector in class
        self.peak_detector_params = dict(peak_sign=peak_sign, relative_threshold=relative_threshold, peak_span_ms=peak_span_ms)
        PeakDetector_class = peakdetector.peakdetector_engines[peakdetector_engine]
        self.peakdetector = PeakDetector_class(self.dataio.sample_rate, self.nb_channel,
                                                        self.info['chunksize'], self.info['internal_dtype'])

        self.peakdetector.change_params(**self.peak_detector_params)
        
        self.info['peak_detector_params'] = self.peak_detector_params
        self.flush_info()
        
        self.arrays.initialize_array('all_peaks', self.memory_mode,  _dtype_peak, (-1, ))
        
        #TODO clip i_stop with duration ???
        
        for seg_num in range(self.dataio.nb_segment):
            
            self.peakdetector.change_params(**self.peak_detector_params)#this reset the fifo index
            
            iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=self.chan_grp,
                            chunksize=self.info['chunksize'], i_stop=None, signal_type='processed')
            for pos, preprocessed_chunk in iterator:
                n_peaks, chunk_peaks = self.peakdetector.process_data(pos, preprocessed_chunk)
            
                if chunk_peaks is not None:
                    peaks = np.zeros(chunk_peaks.size, dtype=_dtype_peak)
                    peaks['index'] = chunk_peaks
                    peaks['segment'][:] = seg_num
                    peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
                    self.arrays.append_chunk('all_peaks',  peaks)

        self.arrays.finalize_array('all_peaks')
        #~ self._reset_waveform_and_features()
        self._reset_arrays(_reset_after_peak_arrays)
        self.on_new_cluster()
    
    


    

    
    def extract_some_waveforms(self, n_left=None, n_right=None,
                            wf_left_ms=None, wf_right_ms=None,
                            index=None, mode='rand', nb_max=10000,
                            align_waveform=False, subsample_ratio=20):
        """
        Extract waveform snippet for a subset of peaks (already detected).
        
        Note that this operation is slow.
        
        After this the attribute some_peaks_index will contain index in all_peaks that
        have waveforms.
        
        Parameters
        ----------
        n_left: int
            Left sweep in sample must be negative
        n_right: int
            Right sweep in sample
        wf_left_ms: 
            Left sweep in ms must be negative
        wf_right_ms: 
            Right sweep in ms must be negative
        index: None (by default) or numpy array of int
            If mode is None then the user can give a selection index of peak 
            to extract waveforms.
        mode: 'rand' (default) or 'all' or None
           'rand' select randomly some peak to extract waveform.
           If None then index must not be None.
        nb_max: int 
            When rand then is this the number of selected waveform.
        
        
        """
        if n_left is None or n_right is None:
            if 'waveform_extractor_params' in self.info:
                n_left = self.info['waveform_extractor_params']['n_left']
                n_right = self.info['waveform_extractor_params']['n_right']
            elif wf_left_ms is not None and wf_right_ms is not None:
                n_left = int(wf_left_ms / 1000. * self.dataio.sample_rate)
                n_right = int(wf_right_ms / 1000. * self.dataio.sample_rate)
            else:
                raise(ValueError('Must provide wf_left_ms/wf_right_ms'))
        
        peak_sign = self.info['peak_detector_params']['peak_sign']
        
        peak_width = n_right - n_left
        
        if index is not None:
            some_peaks_index = index.copy()
        else:
            if mode=='rand' and self.nb_peak>nb_max:
                some_peaks_index = np.random.choice(self.nb_peak, size=nb_max).astype('int64')
            elif mode=='rand' and self.nb_peak<=nb_max:
                some_peaks_index = np.arange(self.nb_peak, dtype='int64')
            elif mode=='all':
                some_peaks_index = np.arange(self.nb_peak, dtype='int64')
            else:
                raise(NotImplementedError, 'unknown mode')
        
        # this is important to not take 2 times the sames, this leads to bad mad/median
        some_peaks_index = np.unique(some_peaks_index)
        
        some_peak_mask = np.zeros(self.nb_peak, dtype='bool')
        some_peak_mask[some_peaks_index] = True
        
        nb = some_peaks_index.size
        
        # make it persitent
        #~ self.arrays.create_array('some_peaks_index', 'int64', (nb,), self.memory_mode)
        #~ self.some_peaks_index[:] = some_peaks_index
        self.arrays.add_array('some_peaks_index', some_peaks_index, self.memory_mode)
        
        shape=(nb, peak_width, self.nb_channel)
        self.arrays.create_array('some_waveforms', self.info['internal_dtype'], shape, self.memory_mode)

        seg_nums = np.unique(self.all_peaks['segment'])
        n = 0
        for seg_num in seg_nums:
            insegment_peaks  = self.all_peaks[some_peak_mask & (self.all_peaks['segment']==seg_num)]
            for peak in insegment_peaks:
                i_start = peak['index']+n_left
                i_stop = i_start+peak_width
                if align_waveform:
                    ratio = subsample_ratio
                    wf = self.dataio.get_signals_chunk(seg_num=seg_num, chan_grp=self.chan_grp, i_start=i_start-peak_width, i_stop=i_stop+peak_width, signal_type='processed')
                    wf2 = scipy.signal.resample(wf, wf.shape[0]*ratio, axis=0)
                    wf2_around_peak = wf2[(peak_width-n_left-2)*ratio:(peak_width-n_left+3)*ratio, :]
                    #~ print(wf2_around_peak.shape)
                    if peak_sign=='+':
                        #~ ind_chan_max = np.argmax(np.max(wf2_around_peak, axis=0))
                        ind_chan_max = np.argmax(wf2_around_peak[ratio, :])
                        ind_max = np.argmax(wf2_around_peak[:, ind_chan_max])
                    elif peak_sign=='-':
                        #~ ind_chan_max_old = np.argmin(np.min(wf2_around_peak, axis=0))
                        ind_chan_max = np.argmin(wf2_around_peak[ratio, :])
                        #~ print(ind_chan_max_old, ind_chan_max)
                        ind_max = np.argmin(wf2_around_peak[:, ind_chan_max])
                    shift = ind_max - ratio*2
                    #~ print('ind_chan_max', ind_chan_max, 'ind_max', ind_max, 'shift', shift)
                    
                    #~ i1=peak_width*ratio+ind_max
                    #~ i1_old = (peak_width-n_left-1)*ratio + ind_max + n_left*ratio 
                    #~ i1 = peak_width*ratio + shift
                    i1 = peak_width*ratio + shift
                    #~ print('i1_old', i1_old, 'i1', i1)
                    i2 = i1+peak_width*ratio
                    wf_short = wf2[i1:i2:ratio, :]
                    self.some_waveforms[n, :, :] = wf_short
                    
                    #DEBUG
                    #~ wf_short = self.dataio.get_signals_chunk(seg_num=seg_num, chan_grp=self.chan_grp, i_start=i_start, i_stop=i_stop, signal_type='processed')
                    #~ import matplotlib.pyplot as plt
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(wf2[:, ind_chan_max])
                    #~ x = (peak_width-n_left-2)*ratio + ind_max
                    #~ print(x, n_left, n_right, peak_width, ratio)
                    #~ y = wf2[(peak_width-n_left-2)*ratio+ind_max, ind_chan_max]
                    #~ y_av = wf2[(peak_width-n_left-2)*ratio+ind_max-ratio, ind_chan_max]
                    #~ y_af = wf2[(peak_width-n_left-2)*ratio+ind_max+ratio, ind_chan_max]
                    #~ ax.plot([x-ratio, x, x+ratio], [y_av, y, y_af], marker='o', markersize=10)
                    #~ ax.axvline((peak_width-n_left-2)*ratio)
                    #~ ax.axvline((peak_width-n_left+3)*ratio)
                    #~ ax.plot(np.arange(wf_short.shape[0])*ratio+peak_width*ratio, wf_short[:, ind_chan_max], ls='--')
                    #~ plt.show()
                    #END DEBUG
                    
                else:
                    wf = self.dataio.get_signals_chunk(seg_num=seg_num, chan_grp=self.chan_grp, i_start=i_start, i_stop=i_stop, signal_type='processed')
                    self.some_waveforms[n, :, :] = wf
                n +=1
        
        self.info['waveform_extractor_params'] = dict(n_left=n_left, n_right=n_right, 
                                                                nb_max=nb_max, align_waveform=align_waveform,
                                                                subsample_ratio=subsample_ratio)
        self.flush_info()
        
        self.projector = None
        self._reset_arrays(_reset_after_waveforms_arrays)

        self.all_peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
        self.all_peaks['cluster_label'][self.some_peaks_index] = 0
        
        self.on_new_cluster()
        self.compute_all_centroid()
        
    
    def clean_waveforms(self, alien_value_threshold=100.):
        """
        Detect bad waveform (artefact, ...) and tag them with allien 
        label (-9)
        
        """
        if alien_value_threshold is not None:
            over = np.any(np.abs(self.some_waveforms)>alien_value_threshold, axis=(1,2))
            index_over = self.some_peaks_index[over]
            index_ok = self.some_peaks_index[~over]
            self.all_peaks['cluster_label'][index_over] = labelcodes.LABEL_ALIEN
            self.all_peaks['cluster_label'][index_ok] = 0

        self.info['clean_waveforms_params'] = dict(alien_value_threshold=alien_value_threshold)
        self.flush_info()

        self.on_new_cluster()
        self.compute_all_centroid()

    
    def find_good_limits(self, mad_threshold = 1.1, channel_percent=0.3, extract=True, min_left=-5, max_right=5):
        """
        Find goods limits for the waveform.
        Where the MAD is above noise level (=1.)
        
        The technics constists in finding continuous samples
        above 10% of backgroud noise for at least 30% of channels
        
        **Parameters**
        
        mad_threshold: (default 1.1) threshold noise
        channel_percent:  (default 0.3) percent of channel above this noise.
        """
        
        old_n_left = self.info['waveform_extractor_params']['n_left']
        old_n_right = self.info['waveform_extractor_params']['n_right']

        median, mad = median_mad(self.some_waveforms, axis = 0)
        # any channel above MAD mad_threshold
        nb_above = np.sum(mad>=mad_threshold, axis=1)
        #~ print('nb_above', nb_above)
        #~ print('self.nb_channel*channel_percent', self.nb_channel*channel_percent)
        above = nb_above>=self.nb_channel*channel_percent
        #find max consequitive point that are True
        #~ print('above', above)
        
        
        up, = np.where(np.diff(above.astype(int))==1)
        down, = np.where(np.diff(above.astype(int))==-1)
        
        
        if len(up)==0 or len(down)==0:
            return None, None
        else:
            up = up[up<max(down)]
            down = down[down>min(up)]
            if len(up)==0 or len(down)==0:
                return None, None
            else:
                best = np.argmax(down-up)
                n_left = int(self.info['waveform_extractor_params']['n_left'] + up[best])
                n_right = int(self.info['waveform_extractor_params']['n_left'] + down[best]+1)
                #~ print(old_n_left, old_n_right)
                #~ print(n_left, n_right)
                
                n_left = min(n_left, min_left)
                n_right = max(n_right, max_right)
                #~ print(n_left, n_right)
                
                if extract:
                    self.projector = None
                    self.extract_some_waveforms(n_left=n_left, n_right=n_right,
                                            index=self.some_peaks_index.copy(), # copy is to avoid reference loop
                                            align_waveform=self.info['waveform_extractor_params']['align_waveform'])
                
                return n_left, n_right


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
            #~ length = self.dataio.get_segment_length(seg_num) #This is wrong
            length = min(self.info['processed_length'], self.dataio.get_segment_length(seg_num))
            
            possibles = np.ones(length, dtype='bool')
            possibles[:peak_width] = False
            possibles[-peak_width:] = False
            peaks = self.all_peaks[self.all_peaks['segment']==seg_num]
            for peak in peaks:
                possibles[peak['index']+n_left-n_right:peak['index']+n_right-n_left]
            possible_indexes, = np.nonzero(possibles)
            noise_index = np.zeros(n_by_seg, dtype=_dtype_peak)
            noise_index['index'] = possible_indexes[np.sort(np.random.choice(possible_indexes.size, size=n_by_seg))]
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

    def extract_some_features(self, method='global_pca', selection=None, **params): #n_components=5, 
        """
        Extract feature from waveforms.
        """
        
        
        if selection is None:
            #by default selection is valid label >=0
            selection = self.all_peaks['cluster_label'][self.some_peaks_index]>=0
        
        #~ wf = self.some_waveforms.reshape(self.some_waveforms.shape[0], -1)
        #~ params['n_components'] = n_components
        features, channel_to_features, self.projector = decomposition.project_waveforms(self.some_waveforms, method=method, selection=None,
                    catalogueconstructor=self, **params)
        
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
    project = extract_some_features
    
    def apply_projection(self):
        assert self.projector is not None
        features = self.projector.transform(self.some_waveforms)
        
        #trick to make it persistant
        #~ self.arrays.create_array('some_features', self.info['internal_dtype'], features.shape, self.memory_mode)
        #~ self.some_features[:] = features
        self.arrays.add_array('some_features', some_features.astype(self.info['internal_dtype']), self.memory_mode)
        
    
    
    def find_clusters(self, method='kmeans', selection=None, **kargs):
        """
        Find cluster for peaks that have a waveform and feature.
        
        """
        #done in a separate module cluster.py
        
        if selection is not None:
            old_labels = np.unique(self.all_peaks['cluster_label'][selection])
            #~ print(old_labels)
        
        
        labels = cluster.find_clusters(self, method=method, selection=selection, **kargs)
        
        if selection is None:
            self.on_new_cluster()
            self.compute_all_centroid()
            #maybe remove this but this is a good practice
            self.order_clusters(by='waveforms_rms')
        else:
            new_labels = np.unique(labels)
            for new_label in new_labels:
                if new_label not in self.clusters['cluster_label'] and new_label>0:
                    self.add_one_cluster(new_label)
                if new_label>0:
                    self.compute_one_centroid(new_label)
            for old_label in old_labels:
                ind = self.index_of_label(old_label)
                nb_peak = np.sum(self.all_peaks['cluster_label']==old_label)
                if nb_peak == 0:
                    self.pop_labels_from_cluster([old_label])
                else:
                    self.clusters['nb_peak'][ind] = nb_peak
                    self.compute_one_centroid(old_label)

    def on_new_cluster(self):
        #~ print('cc.on_new_cluster')
        if self.all_peaks is None:
            return
        cluster_labels = np.unique(self.all_peaks['cluster_label'])
        clusters = np.zeros(cluster_labels.shape, dtype=_dtype_cluster)
        clusters['cluster_label'][:] = cluster_labels
        clusters['cell_label'][:] = cluster_labels
        clusters['max_on_channel'][:] = -1
        clusters['max_peak_amplitude'][:] = np.nan
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
        clusters['max_on_channel'][pos_insert] = -1
        clusters['max_peak_amplitude'][pos_insert] = np.nan
        clusters['waveform_rms'][pos_insert] = np.nan
        clusters['nb_peak'][pos_insert] = np.sum(self.all_peaks['cluster_label']==label)
        
        self.arrays.add_array('clusters', clusters, self.memory_mode)
        
        for name in _centroids_arrays:
            arr = getattr(self, name).copy()
            new_arr = np.zeros((arr.shape[0]+1, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            if label>=0:
                new_arr[:-1, :, :] = arr
            else:
                new_arr[1:, :, :] = arr
            self.arrays.add_array(name, new_arr, self.memory_mode)
        
        #TODO set one color
        self.refresh_colors(reset=False)
        
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
            new_arr = getattr(self, name)[keep, :, :].copy()
            self.arrays.add_array(name, new_arr, self.memory_mode)
    
    def move_cluster_to_trash(self, labels):
        if isinstance(labels, int):
            labels = [labels]
        
        mask = np.zeros(self.all_peaks.size, dtype='bool')
        for k in labels:
            mask |= self.all_peaks['cluster_label']== k
        self.change_spike_label(mask, labelcodes.LABEL_TRASH)
    
    def compute_one_centroid(self, k, flush=True):
        #~ t1 = time.perf_counter()
        ind = self.index_of_label(k)
        
        n_left = int(self.info['waveform_extractor_params']['n_left'])
        
        wf = self.some_waveforms[self.all_peaks['cluster_label'][self.some_peaks_index]==k]
        median, mad = median_mad(wf, axis = 0)
        mean, std = np.mean(wf, axis=0), np.std(wf, axis=0)
        max_on_channel = np.argmax(np.abs(median[-n_left,:]), axis=0)
        
        # to persistant arrays
        self.centroids_median[ind, :, :] = median
        self.centroids_mad[ind, :, :] = mad
        self.centroids_mean[ind, :, :] = mean
        self.centroids_std[ind, :, :] = std
        
        self.clusters['max_on_channel'][ind] = max_on_channel
        self.clusters['max_peak_amplitude'][ind] = median[-n_left, max_on_channel]
        self.clusters['waveform_rms'][ind] = np.sqrt(np.mean(median**2))

        if flush:
            for name in ('clusters',) + _centroids_arrays:
                self.arrays.flush_array(name)
        
        #~ t2 = time.perf_counter()
        #~ print('compute_one_centroid',k, t2-t1)

    def compute_all_centroid(self):
        
        t1 = time.perf_counter()
        if self.some_waveforms is None:
            for name in _centroids_arrays:
                self.arrays.detach_array(name)
                setattr(self, name, None)
            return
        
        
        n_left = int(self.info['waveform_extractor_params']['n_left'])
        n_right = int(self.info['waveform_extractor_params']['n_right'])
        
        for name in _centroids_arrays:
            empty = np.zeros((self.cluster_labels.size, n_right - n_left, self.nb_channel), dtype=self.info['internal_dtype'])
            self.arrays.add_array(name, empty, self.memory_mode)
        
        #~ t1 = time.perf_counter()
        for k in self.cluster_labels:
            if k <0: continue
            
            self.compute_one_centroid(k, flush=False)

        for name in ('clusters',) + _centroids_arrays:
            self.arrays.flush_array(name)
        
        #~ self.arrays.flush_array('clusters')
        
        #~ t2 = time.perf_counter()
        #~ print('compute_all_centroid', t2-t1)
    
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
    
    def trash_small_cluster(self, n=10):
        to_remove = []
        for k in list(self.cluster_labels):
            mask = self.all_peaks['cluster_label']==k
            if np.sum(mask)<=n:
                self.all_peaks['cluster_label'][mask] = -1
                to_remove.append(k)
        
        self.pop_labels_from_cluster(to_remove)

    def compute_spike_waveforms_similarity(self, method='cosine_similarity', size_max = 1e7):
        """This compute the similarity spike by spike.
        """
        #~ t1 = time.perf_counter()
        spike_waveforms_similarity = None
        if self.some_waveforms is not None:
            wf = self.some_waveforms
            wf = wf.reshape(wf.shape[0], -1)
            if wf.size<size_max:
                spike_waveforms_similarity = metrics.compute_similarity(wf, method)
        
        if spike_waveforms_similarity is None:
            self.arrays.detach_array('spike_waveforms_similarity')
            self.spike_waveforms_similarity = None
        else:
            self.arrays.add_array('spike_waveforms_similarity', spike_waveforms_similarity.astype('float32'), self.memory_mode)

        #~ t2 = time.perf_counter()
        #~ print('compute_spike_waveforms_similarity', t2-t1)
        
        return self.spike_waveforms_similarity

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
            print('auto_merge', k1, 'with', k2)
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
            #~ chan = self.centroids[k]['max_on_channel']
            chan = self.clusters['max_on_channel'][ind]
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
    
    
    def clean_cluster(self, too_small=10):
        self.trash_small_cluster(n=too_small)
    
    def compute_spike_silhouette(self, size_max=1e7):
        #~ t1 = time.perf_counter()
        
        spike_silhouette = None
        wf = self.some_waveforms
        if wf is not None:
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
        elif by=='max_peak_amplitude':
            order = np.argsort(np.abs(pos_clusters['max_peak_amplitude']))[::-1]
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
            arr_pos = arr[mask_pos, :, :].copy()
            arr[mask_pos, :, :] = arr_pos[order, :, :]
        
        self.refresh_colors(reset=True)
        
        #~ self._reset_metrics()
        self._reset_arrays(_persistent_metrics)

    def make_catalogue(self):
        #TODO: offer possibility to resample some waveforms or choose the number
        
        #~ t1 = time.perf_counter()
        self.catalogue = {}
        
        self.catalogue = {}
        self.catalogue['chan_grp'] = self.chan_grp
        n_left = self.catalogue['n_left'] = int(self.info['waveform_extractor_params']['n_left'] +2)
        self.catalogue['n_right'] = int(self.info['waveform_extractor_params']['n_right'] -2)
        self.catalogue['peak_width'] = self.catalogue['n_right'] - self.catalogue['n_left']
        
        #for colors
        self.refresh_colors(reset=False)
        
        keep = self.cluster_labels>=0
        cluster_labels = np.array(self.cluster_labels[keep], copy=True)
        # TODO this is redundant with clusters, this shoudl be removed but imply some change in peeler.
        self.catalogue['cluster_labels'] = cluster_labels 
        self.catalogue['clusters'] = self.clusters[keep].copy()
        
        
        n, full_width, nchan = self.some_waveforms.shape
        centers0 = np.zeros((len(cluster_labels), full_width - 4, nchan), dtype=self.info['internal_dtype'])
        centers1 = np.zeros_like(centers0)
        centers2 = np.zeros_like(centers0)
        self.catalogue['centers0'] = centers0 # median of wavforms
        self.catalogue['centers1'] = centers1 # median of first derivative of wavforms
        self.catalogue['centers2'] = centers2 # median of second derivative of wavforms
        
        subsample = np.arange(1.5, full_width-2.5, 1/20.)
        self.catalogue['subsample_ratio'] = 20
        interp_centers0 = np.zeros((len(cluster_labels), subsample.size, nchan), dtype=self.info['internal_dtype'])
        self.catalogue['interp_centers0'] = interp_centers0
        
        #~ print('peak_width', self.catalogue['peak_width'])
        
        self.catalogue['label_to_index'] = {}
        for i, k in enumerate(cluster_labels):
            self.catalogue['label_to_index'][k] = i
            
            #print('construct_catalogue', k)
            # take peak of this cluster
            # and reshaape (nb_peak, nb_channel, nb_csample)
            #~ wf = self.some_waveforms[self.all_peaks['cluster_label']==k]
            wf0 = self.some_waveforms[self.all_peaks['cluster_label'][self.some_peaks_index]==k]
            #~ wf0 = wf0.copy()
            #~ print(wf0.shape, wf0.size)
            
            # compute first and second derivative on dim=1 (time)
            # Note this was the old implementation but too slow
            #~ kernel = np.array([1,0,-1])/2.
            #~ kernel = kernel[None, :, None]
            #~ wf1 =  scipy.signal.fftconvolve(wf0,kernel,'same') # first derivative
            #~ wf2 =  scipy.signal.fftconvolve(wf1,kernel,'same') # second derivative
            # this is teh new one
            wf1 = np.zeros_like(wf0)
            wf1[:, 1:-1, :] = (wf0[:, 2:,: ] - wf0[:, :-2,: ])/2.
            wf2 = np.zeros_like(wf1)
            wf2[:, 1:-1, :] = (wf1[:, 2:,: ] - wf1[:, :-2,: ])/2.
            
            
            #median and
            #eliminate margin because of border effect of derivative and reshape
            center0 = np.median(wf0, axis=0)
            centers0[i,:,:] = center0[2:-2, :]
            centers1[i,:,:] = np.median(wf1, axis=0)[2:-2, :]
            centers2[i,:,:] = np.median(wf2, axis=0)[2:-2, :]
            #~ center0 = np.mean(wf0, axis=0)
            #~ centers0[i,:,:] = center0[2:-2, :]
            #~ centers1[i,:,:] = np.mean(wf1, axis=0)[2:-2, :]
            #~ centers2[i,:,:] = np.mean(wf2, axis=0)[2:-2, :]

            #interpolate centers0 for reconstruction inbetween bsample when jitter is estimated
            f = scipy.interpolate.interp1d(np.arange(full_width), center0, axis=0, kind='cubic')
            oversampled_center = f(subsample)
            interp_centers0[i, :, :] = oversampled_center
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(np.arange(full_width-4), center0[2:-2, :], color='b', marker='o')
            #~ ax.plot(subsample-2.,oversampled_center, color='c')
            #~ plt.show()
            
        #find max  channel for each cluster for peak alignement
        self.catalogue['max_on_channel'] = np.zeros_like(self.catalogue['cluster_labels'])
        for i, k in enumerate(cluster_labels):
            center = self.catalogue['centers0'][i,:,:]
            self.catalogue['max_on_channel'][i] = np.argmax(np.abs(center[-n_left,:]), axis=0)
        
        #colors
        
        
        #~ self.catalogue['cluster_colors'] = {}
        #~ self.catalogue['cluster_colors'].update(self.colors)
        
        #params
        self.catalogue['signal_preprocessor_params'] = dict(self.info['signal_preprocessor_params'])
        self.catalogue['peak_detector_params'] = dict(self.info['peak_detector_params'])
        self.catalogue['clean_waveforms_params'] = dict(self.info['clean_waveforms_params'])
        self.catalogue['signals_medians'] = np.array(self.signals_medians, copy=True)
        self.catalogue['signals_mads'] = np.array(self.signals_mads, copy=True)
        
        
        #~ t2 = time.perf_counter()
        #~ print('make_catalogue', t2-t1)
        
        return self.catalogue
    
    def make_catalogue_for_peeler(self):
        """
        Make and save catalogue in the working dir for the Peeler.
        
        """
        self.make_catalogue()
        self.dataio.save_catalogue(self.catalogue, name='initial')
        
    def create_savepoint(self):
        """this create a copy of the entire catalogue_constructor subdir
        Usefull for the UI when the user wants to snapshot and try tricky merge/split.
        """
        
        copy_path = self.catalogue_path + '_SAVEPOINT_{:%Y-%m-%d_%Hh%Mm%S}'.format(datetime.datetime.now())
        
        if not os.path.exists(copy_path):
            shutil.copytree(self.catalogue_path, copy_path)
            
        return copy_path





