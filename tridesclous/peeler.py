"""

.. autoclass:: Peeler
   :members:

"""

import os
import json
from collections import OrderedDict, namedtuple
import time

import numpy as np
import scipy.signal


from . import signalpreprocessor
from .peakdetector import  detect_peaks_in_chunk

from .tools import make_color_dict

import matplotlib.pyplot as plt
#import seaborn as sns


from tqdm import tqdm

from . import pythran_tools
if hasattr(pythran_tools, '__pythran__'):
    HAVE_PYTHRAN = True
else:
    HAVE_PYTHRAN = False

try:
    import pyopencl
    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False


_dtype_spike = [('index', 'int64'), ('cluster_label', 'int64'), ('jitter', 'float64'),]

Spike = namedtuple('Spike', ('index', 'cluster_label', 'jitter'))


from .labelcodes import (LABEL_TRASH, LABEL_UNCLASSIFIED, LABEL_ALIEN)

LABEL_LEFT_LIMIT = -11
LABEL_RIGHT_LIMIT = -12
LABEL_MAXIMUM_SHIFT = -13
# good label are >=0


#~ maximum_jitter_shift = 10
maximum_jitter_shift = 4
#~ maximum_jitter_shift = 1

class Peeler:
    """
    The peeler is core of spike sorting itself.
    It basically do a *template matching* on a signals.
    
    This class nedd a *catalogue* constructed by :class:`CatalogueConstructor`.
    Then the compting is applied chunk chunk on the raw signal itself.
    
    So this class is the same for both offline/online computing.
    
    At each chunk, the algo is basically this one:
      1. apply the processing chain (filter, normamlize, ....)
      2. Detect peaks
      3. Try to classify peak and detect the *jitter*
      4. With labeled peak create a prediction for the chunk
      5. Substract the prediction from the processed signals.
      6. Go back to **2** until there is no peak or only peaks that can't be labeled.
      7. return labeld spikes from this or previous chunk and the processed signals (for display or recoding)
    
    The main difficulty in the implemtation is to deal with edge because spikes 
    waveforms can spread out in between 2 chunk.
    
    Note that the global latency depend on this é paramters:
      * lostfront_chunksize
      * chunksize

    
    """
    def __init__(self, dataio):
        #for online dataio is None
        self.dataio = dataio

    def __repr__(self):
        t = "Peeler <id: {}> \n  workdir: {}\n".format(id(self), self.dataio.dirname)
        
        return t

    def change_params(self, catalogue=None, chunksize=1024, 
                                        internal_dtype='float32', 
                                        use_sparse_template=False,
                                        sparse_threshold_mad=1.5,
                                        use_opencl_with_sparse=False,
                                        use_pythran_with_sparse=False,
                                        ):
        """
        Set parameters for the Peeler.
        
        
        Parameters
        ----------
        catalogue: the catalogue (a dict)
            The catalogue made by CatalogueConstructor.
        chunksize: int (1024 by default)
            the size of chunk for processing.
        internal_dtype: 'float32' or 'float64'
            dtype of internal processing. float32 is OK. float64 is totally useless.
        use_sparse_template: bool (dafult False)
            For very high channel count, centroids from catalogue can be sparcifyed.
            The speedup a lot the process but the sparse_threshold_mad must be
            set carrefully and compared with use_sparse_template=False.
            For low channel count this is useless.
        sparse_threshold_mad: float (1.5 by default)
            The threshold level.
            Under this value if all sample on one channel for one centroid
            is considred as NaN
        use_opencl_with_sparse: bool
            When use_sparse_template is True, you can use this to accelerate
            the labelling of each spike. Usefull for high channel count.
        use_pythran_with_sparse: bool
            experimental same as use_opencl_with_sparse but with pythran
        """
        assert catalogue is not None
        self.catalogue = catalogue
        self.chunksize = chunksize
        self.internal_dtype= internal_dtype
        self.use_sparse_template = use_sparse_template
        self.sparse_threshold_mad = sparse_threshold_mad
        self.use_opencl_with_sparse = use_opencl_with_sparse
        self.use_pythran_with_sparse = use_pythran_with_sparse
        
        # Some check
        if self.use_opencl_with_sparse or self.use_pythran_with_sparse:
            assert self.use_sparse_template, 'For that option you must use sparse template'
        if self.use_sparse_template:
            assert self.use_opencl_with_sparse or self.use_pythran_with_sparse, 'For that option you must use OpenCL or Pytran'
        if self.use_opencl_with_sparse:
            assert HAVE_PYOPENCL, 'OpenCL is not available'
        if self.use_pythran_with_sparse:
            assert HAVE_PYTHRAN, 'Pythran is not available'
        
        self.colors = make_color_dict(self.catalogue['clusters'])
        
        # precompute some value for jitter estimation
        n = self.catalogue['cluster_labels'].size
        self.catalogue['wf1_norm2'] = np.zeros(n)
        self.catalogue['wf2_norm2'] = np.zeros(n)
        self.catalogue['wf1_dot_wf2'] = np.zeros(n)
        for i, k in enumerate(self.catalogue['cluster_labels']):
            chan = self.catalogue['max_on_channel'][i]
            wf0 = self.catalogue['centers0'][i,: , chan]
            wf1 = self.catalogue['centers1'][i,: , chan]
            wf2 = self.catalogue['centers2'][i,: , chan]

            self.catalogue['wf1_norm2'][i] = wf1.dot(wf1)
            self.catalogue['wf2_norm2'][i] = wf2.dot(wf2)
            self.catalogue['wf1_dot_wf2'][i] = wf1.dot(wf2)
        
        if self.use_sparse_template:
            centers = wf0 = self.catalogue['centers0']
            #~ print(centers.shape)
            mask = np.any(np.abs(centers)>sparse_threshold_mad, axis=1)
            #~ print(mask.shape)
            #~ print(mask)
            print('average sparseness for templates', np.sum(mask)/mask.size)
            self.catalogue['sparse_mask'] = mask
            
            #~ for i in range(centers.shape[0]):
                #~ fig, ax = plt.subplots()
                #~ center = centers[i,:,:].copy()
                #~ center_sparse = center.copy()
                #~ center_sparse[:, ~mask[i, :]] = 0.
                #~ ax.plot(center.T.flatten(), color='g')
                #~ ax.plot(center_sparse.T.flatten(), color='r', ls='--')
                #~ ax.axhline(sparse_threshold_mad)
                #~ ax.axhline(-sparse_threshold_mad)
                #~ plt.show()
        
            if self.use_opencl_with_sparse:
                self.ctx = pyopencl.create_some_context(interactive=False)
                
                self.queue = pyopencl.CommandQueue(self.ctx)
                
                centers = self.catalogue['centers0']
                nb_channel = centers.shape[2]
                peak_width = centers.shape[1]
                nb_cluster = centers.shape[0]
                kernel = kernel_opencl%{'nb_channel': nb_channel,'peak_width':peak_width,
                                                        'total':peak_width*nb_channel,'nb_cluster' : nb_cluster}
                prg = pyopencl.Program(self.ctx, kernel)
                opencl_prg = prg.build(options='-cl-mad-enable')
                self.kern_waveform_distance = getattr(opencl_prg, 'waveform_distance')

                wf_shape = centers.shape[1:]
                one_waveform = np.zeros(wf_shape, dtype='float32')
                self.one_waveform_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=one_waveform)

                self.catalogue_center_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=centers)

                self.waveform_distance = np.zeros((nb_cluster), dtype='float32')
                self.waveform_distance_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.waveform_distance)

                #~ mask[:] = 0
                self.mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=mask.astype('u1'))

                rms_waveform_channel = np.zeros(nb_channel, dtype='float32')
                self.rms_waveform_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=rms_waveform_channel)
                
                self.cl_global_size = (centers.shape[0], centers.shape[2])
                #~ self.cl_local_size = None
                self.cl_local_size = (centers.shape[0], 1) # faster a GPU because of memory access
                #~ self.cl_local_size = (1, centers.shape[2])
                

    
    def process_one_chunk(self,  pos, sigs_chunk):
        #~ print('*'*5)
        #~ print('chunksize', self.chunksize, '=', self.chunksize/self.sample_rate*1000, 'ms')
        
        #~ t1 = time.perf_counter()
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        #~ t2 = time.perf_counter()
        #~ print('process_data', (t2-t1)*1000)
        
        
        #note abs_head_index is smaller than pos because prepcorcessed chunk
        # is late because of local filfilt in signalpreprocessor
        
        #shift rsiruals buffer and put the new one on right side
        #~ t1 = time.perf_counter()
        fifo_roll_size = self.fifo_residuals.shape[0]-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_residuals.shape[0]:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk
        #~ t2 = time.perf_counter()
        #~ print('fifo move', (t2-t1)*1000.)

        
        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        # TODO remove from peak the very begining of the signal because of border filtering effects
        
        #~ t1 = time.perf_counter()
        good_spikes = []
        all_ready_tested = []
        while True:
            #detect peaks
            t3 = time.perf_counter()
            local_peaks = detect_peaks_in_chunk(self.fifo_residuals, self.n_span, self.relative_threshold, self.peak_sign)
            t4 = time.perf_counter()
            #~ print('self.fifo_residuals median', np.median(self.fifo_residuals, axis=0))
            #~ print('  detect_peaks_in_chunk', (t4-t3)*1000.)
            
            if len(all_ready_tested)>0:
                local_peaks_to_check = local_peaks[~np.in1d(local_peaks, all_ready_tested)]
            else:
                local_peaks_to_check = local_peaks
            
            n_ok = 0
            for i, local_peak in enumerate(local_peaks_to_check):
                #~ print('    local_peak', local_peak, 'i', i)
                #~ t3 = time.perf_counter()
                spike = self.classify_and_align_one_spike(local_peak, self.fifo_residuals, self.catalogue)
                #~ t4 = time.perf_counter()
                #~ print('    classify_and_align_one_spike', (t4-t3)*1000.)
                
                if spike.cluster_label>=0:
                    #~ t3 = time.perf_counter()
                    #~ print('     >>spike.index', spike.index, spike.cluster_label, 'abs index', spike.index+shift)
                    spikes = np.array([spike], dtype=_dtype_spike)
                    prediction = make_prediction_signals(spikes, self.fifo_residuals.dtype, self.fifo_residuals.shape, self.catalogue, safe=False)
                    self.fifo_residuals -= prediction
                    spikes['index'] += shift
                    good_spikes.append(spikes)
                    n_ok += 1
                    #~ t4 = time.perf_counter()
                    #~ print('    make_prediction_signals and sub', (t4-t3)*1000.)
                    
                    #~ print('    all_ready_tested before', all_ready_tested)
                    all_ready_tested = [ind for ind in all_ready_tested if np.abs(spike.index-ind)>self.peak_width]
                    #~ print('    all_ready_tested new deal', all_ready_tested)
                else:
                    all_ready_tested.append(local_peak)
            
            if n_ok==0:
                # no peak can be labeled
                # reserve bad spikes on the right limit for next time
                local_peaks = local_peaks[local_peaks<(self.chunksize+self.n_span)]
                bad_spikes = np.zeros(local_peaks.shape[0], dtype=_dtype_spike)
                bad_spikes['index'] = local_peaks + shift
                bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                break
        
        #~ t2 = time.perf_counter()
        #~ print('LOOP classify_and_align_one_spike', (t2-t1)*1000)
        
        
        #concatenate, sort and count
        # here the trick is to keep spikes at the right border
        # and keep then until the next loop this avoid unordered spike
        if len(good_spikes)>0:
            good_spikes = np.concatenate(good_spikes)
            near_border = (good_spikes['index'] - shift)>=(self.chunksize+self.n_span)
            near_border_good_spikes = good_spikes[near_border].copy()
            good_spikes = good_spikes[~near_border]

            all_spikes = np.concatenate([good_spikes] + [bad_spikes] + self.near_border_good_spikes)
            self.near_border_good_spikes = [near_border_good_spikes] # for next chunk
        else:
            all_spikes = np.concatenate([bad_spikes] + self.near_border_good_spikes)
            self.near_border_good_spikes = []
        
        # all_spikes = all_spikes[np.argsort(all_spikes['index'])]
        all_spikes = all_spikes.take(np.argsort(all_spikes['index']))
        self.total_spike += all_spikes.size
        
        return abs_head_index, preprocessed_chunk, self.total_spike, all_spikes
            
    
    
    def _initialize_before_each_segment(self, sample_rate=None, nb_channel=None, source_dtype=None):
        
        self.nb_channel = nb_channel
        self.sample_rate = sample_rate
        self.source_dtype = source_dtype
        
        self.signalpreprocessor_engine = self.catalogue['signal_preprocessor_params']['signalpreprocessor_engine']
        #~ print('self.signalpreprocessor_engine', self.signalpreprocessor_engine)
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[self.signalpreprocessor_engine]
        #~ SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines['numpy']
        self.signalpreprocessor = SignalPreprocessor_class(sample_rate, nb_channel, self.chunksize, source_dtype)
        
        p = dict(self.catalogue['signal_preprocessor_params'])
        p.pop('signalpreprocessor_engine')
        p['normalize'] = True
        p['signals_medians'] = self.catalogue['signals_medians']
        p['signals_mads'] = self.catalogue['signals_mads']
        self.signalpreprocessor.change_params(**p)
        
        assert self.chunksize>self.signalpreprocessor.lostfront_chunksize
        
        self.internal_dtype = self.signalpreprocessor.output_dtype

        self.peak_sign = self.catalogue['peak_detector_params']['peak_sign']
        self.relative_threshold = self.catalogue['peak_detector_params']['relative_threshold']
        peak_span = self.catalogue['peak_detector_params']['peak_span']
        self.n_span = int(sample_rate*peak_span)//2
        self.n_span = max(1, self.n_span)
        self.peak_width = self.catalogue['peak_width']
        self.n_side = self.catalogue['peak_width'] + maximum_jitter_shift + self.n_span + 1
        
        assert self.chunksize > (self.n_side+1), 'chunksize is too small because of n_size'
        
        self.alien_value_threshold = self.catalogue['clean_waveforms_params']['alien_value_threshold']
        
        self.total_spike = 0
        
        self.near_border_good_spikes = []
        
        self.fifo_residuals = np.zeros((self.n_side+self.chunksize, nb_channel), 
                                                                dtype=self.internal_dtype)
    
    
    def initialize_online_loop(self, sample_rate=None, nb_channel=None, source_dtype=None):
        self._initialize_before_each_segment(sample_rate=sample_rate, nb_channel=nb_channel, source_dtype=source_dtype)
    
    def run_offline_loop_one_segment(self, seg_num=0, duration=None, progressbar=True):
        chan_grp = self.catalogue['chan_grp']
        
        kargs = {}
        kargs['sample_rate'] = self.dataio.sample_rate
        kargs['nb_channel'] = self.dataio.nb_channel(chan_grp)
        kargs['source_dtype'] = self.dataio.source_dtype
        self._initialize_before_each_segment(**kargs)
        
        if duration is not None:
            length = int(duration*self.dataio.sample_rate)
        else:
            length = self.dataio.get_segment_length(seg_num)
        #~ length -= length%self.chunksize
        
        #initialize engines
        self.dataio.reset_processed_signals(seg_num=seg_num, chan_grp=chan_grp, dtype=self.internal_dtype)
        self.dataio.reset_spikes(seg_num=seg_num, chan_grp=chan_grp, dtype=_dtype_spike)

        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=chan_grp, chunksize=self.chunksize, 
                                                    i_stop=length, signal_type='initial')
        if progressbar:
            iterator = tqdm(iterable=iterator, total=length//self.chunksize)
        for pos, sigs_chunk in iterator:
            
            sig_index, preprocessed_chunk, total_spike, spikes = self.process_one_chunk(pos, sigs_chunk)
            
            if sig_index<=0:
                continue
            
            # save preprocessed_chunk to file
            self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num,chan_grp=chan_grp,
                        i_start=sig_index-preprocessed_chunk.shape[0], i_stop=sig_index,
                        signal_type='processed')
            
            if spikes is not None and spikes.size>0:
                self.dataio.append_spikes(seg_num=seg_num, chan_grp=chan_grp, spikes=spikes)
        
        if len(self.near_border_good_spikes)>0:
            # deal with extra remaining spikes
            extra_spikes = self.near_border_good_spikes[0]
            extra_spikes = extra_spikes.take(np.argsort(extra_spikes['index']))
            self.total_spike += extra_spikes.size
            if extra_spikes.size>0:
                self.dataio.append_spikes(seg_num=seg_num, chan_grp=chan_grp, spikes=extra_spikes)
        
        self.dataio.flush_processed_signals(seg_num=seg_num, chan_grp=chan_grp)
        self.dataio.flush_spikes(seg_num=seg_num, chan_grp=chan_grp)

    def run_offline_all_segment(self, **kargs):
        #TODO remove chan_grp here because it is redundant from catalogue['chan_grp']
        assert hasattr(self, 'catalogue'), 'So peeler.change_params first'
        
        
        #~ print('run_offline_all_segment', chan_grp)
        for seg_num in range(self.dataio.nb_segment):
            self.run_offline_loop_one_segment(seg_num=seg_num, **kargs)
    
    run = run_offline_all_segment

    def classify_and_align_one_spike(self, local_index, residual, catalogue):
        # local_index is index of peaks inside residual and not
        # the absolute peak_pos. So time scaling must be done outside.
        
        width = catalogue['peak_width']
        n_left = catalogue['n_left']
        alien_value_threshold = catalogue['clean_waveforms_params']['alien_value_threshold']
        
        
        #ind is the windows border!!!!!
        ind = local_index + n_left

        if ind+width+maximum_jitter_shift+1>=residual.shape[0]:
            # too near right limits no label
            label = LABEL_RIGHT_LIMIT
            jitter = 0
        elif ind<=maximum_jitter_shift:
            # too near left limits no label
            #~ print('     LABEL_LEFT_LIMIT', ind)
            label = LABEL_LEFT_LIMIT
            jitter = 0
        elif catalogue['centers0'].shape[0]==0:
            # empty catalogue
            label  = LABEL_UNCLASSIFIED
            jitter = 0
        else:
            waveform = residual[ind:ind+width,:]
            
            if alien_value_threshold is not None and \
                    np.any(np.abs(waveform)>alien_value_threshold) :
                label  = LABEL_ALIEN
                jitter = 0
            else:
                
                #~ t1 = time.perf_counter()
                label, jitter = self.estimate_one_jitter(waveform)
                #~ t2 = time.perf_counter()
                #~ print('  estimate_one_jitter', (t2-t1)*1000.)

                #~ jitter = -jitter
                #TODO debug jitter sign is positive on right and negative to left
                
                #~ print('label, jitter', label, jitter)
                
                # if more than one sample of jitter
                # then we try a peak shift
                # take it if better
                #TODO debug peak shift
                if np.abs(jitter) > 0.5 and label >=0:
                    prev_ind, prev_label, prev_jitter =ind, label, jitter
                    
                    shift = -int(np.round(jitter))
                    #~ print('classify and align shift', shift)
                    
                    if np.abs(shift) >maximum_jitter_shift:
                        #~ print('     LABEL_MAXIMUM_SHIFT avec shift')
                        label = LABEL_MAXIMUM_SHIFT
                    else:
                        ind = ind + shift
                        if ind+width>=residual.shape[0]:
                            #~ print('     LABEL_RIGHT_LIMIT avec shift')
                            label = LABEL_RIGHT_LIMIT
                        elif ind<0:
                            #~ print('     LABEL_LEFT_LIMIT avec shift')
                            label = LABEL_LEFT_LIMIT
                            #TODO: force to label anyway the spike if spike is at the left of FIFO
                        else:
                            waveform = residual[ind:ind+width,:]
                            new_label, new_jitter = self.estimate_one_jitter(waveform)
                            if np.abs(new_jitter)<np.abs(prev_jitter):
                                #~ print('keep shift')
                                label, jitter = new_label, new_jitter
                                local_index += shift
                            else:
                                #~ print('no keep shift worst jitter')
                                pass

        #security if with jitter the index is out
        if label>=0:
            local_pos = local_index - np.round(jitter).astype('int64') + n_left
            if local_pos<0:
                label = LABEL_LEFT_LIMIT
            elif (local_pos+width) >=residual.shape[0]:
                label = LABEL_RIGHT_LIMIT
        
        return Spike(local_index, label, jitter)
    
    
    def estimate_one_jitter(self, waveform):
        """
        Estimate the jitter for one peak given its waveform
        
        Method proposed by Christophe Pouzat see:
        https://hal.archives-ouvertes.fr/hal-01111654v1
        http://christophe-pouzat.github.io/LASCON2016/SpikeSortingTheElementaryWay.html
        
        for best reading (at least for me SG):
          * wf = the wafeform of the peak
          * k = cluster label of the peak
          * wf0, wf1, wf2 : center of catalogue[k] + first + second derivative
          * jitter0 : jitter estimation at order 0
          * jitter1 : jitter estimation at order 1
          * h0_norm2: error at order0
          * h1_norm2: error at order1
          * h2_norm2: error at order2
        """
        
        # This line is the slower part !!!!!!
        # cluster_idx = np.argmin(np.sum(np.sum((catalogue['centers0']-waveform)**2, axis = 1), axis = 1))
        
        catalogue = self.catalogue
        
        if self.use_opencl_with_sparse:
            rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
            
            pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)
            pyopencl.enqueue_copy(self.queue,  self.rms_waveform_channel_cl, rms_waveform_channel)
            event = self.kern_waveform_distance(self.queue,  self.cl_global_size, self.cl_local_size,
                        self.one_waveform_cl, self.catalogue_center_cl, self.mask_cl, 
                        self.rms_waveform_channel_cl, self.waveform_distance_cl)
            pyopencl.enqueue_copy(self.queue,  self.waveform_distance, self.waveform_distance_cl)
            cluster_idx = np.argmin(self.waveform_distance)

        elif self.use_pythran_with_sparse:
            s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                catalogue['centers0'],  catalogue['sparse_mask'])
            cluster_idx = np.argmin(s)
        else:
            # replace by this (indentique but faster, a but)
            
            #~ t1 = time.perf_counter()
            d = catalogue['centers0']-waveform[None, :, :]
            d *= d
            #s = d.sum(axis=1).sum(axis=1)  # intuitive
            #s = d.reshape(d.shape[0], -1).sum(axis=1) # a bit faster
            s = np.einsum('ijk->i', d) # a bit faster
            cluster_idx = np.argmin(s)
            #~ t2 = time.perf_counter()
            #~ print('    np.argmin V2', (t2-t1)*1000., cluster_idx)
        

        k = catalogue['cluster_labels'][cluster_idx]
        chan = catalogue['max_on_channel'][cluster_idx]
        #~ print('cluster_idx', cluster_idx, 'k', k, 'chan', chan)

        
        #~ return k, 0.

        wf0 = catalogue['centers0'][cluster_idx,: , chan]
        wf1 = catalogue['centers1'][cluster_idx,: , chan]
        wf2 = catalogue['centers2'][cluster_idx,: , chan]
        wf = waveform[:, chan]
        #~ print()
        #~ print(wf0.shape, wf.shape)
        
        
        #it is  precompute that at init speedup 10%!!! yeah
        #~ wf1_norm2 = wf1.dot(wf1)
        #~ wf2_norm2 = wf2.dot(wf2)
        #~ wf1_dot_wf2 = wf1.dot(wf2)
        wf1_norm2= catalogue['wf1_norm2'][cluster_idx]
        wf2_norm2 = catalogue['wf2_norm2'][cluster_idx]
        wf1_dot_wf2 = catalogue['wf1_dot_wf2'][cluster_idx]
        
        
        h = wf - wf0
        h0_norm2 = h.dot(h)
        h_dot_wf1 = h.dot(wf1)
        jitter0 = h_dot_wf1/wf1_norm2
        h1_norm2 = np.sum((h-jitter0*wf1)**2)
        #~ print(h0_norm2, h1_norm2)
        #~ print(h0_norm2 > h1_norm2)
        
        
        
        if h0_norm2 > h1_norm2:
            #order 1 is better than order 0
            h_dot_wf2 = np.dot(h,wf2)
            rss_first = -2*h_dot_wf1 + 2*jitter0*(wf1_norm2 - h_dot_wf2) + 3*jitter0**2*wf1_dot_wf2 + jitter0**3*wf2_norm2
            rss_second = 2*(wf1_norm2 - h_dot_wf2) + 6*jitter0*wf1_dot_wf2 + 3*jitter0**2*wf2_norm2
            jitter1 = jitter0 - rss_first/rss_second
            #~ h2_norm2 = np.sum((h-jitter1*wf1-jitter1**2/2*wf2)**2)
            #~ if h1_norm2 <= h2_norm2:
                #when order 2 is worse than order 1
                #~ jitter1 = jitter0
        else:
            jitter1 = 0.
        #~ print('jitter1', jitter1)
        #~ return k, 0.
        
        #~ print(np.sum(wf**2), np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2))
        #~ print(np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2))
        #~ return k, jitter1

        
        if np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2):
            #prediction should be smaller than original (which have noise)
            return k, jitter1
        else:
            #otherwise the prediction is bad
            #~ print('bad prediction')
            return LABEL_UNCLASSIFIED, 0.


def make_prediction_signals(spikes, dtype, shape, catalogue, safe=True):
    #~ n_left, peak_width, 
    
    prediction = np.zeros(shape, dtype=dtype)
    for i in range(spikes.size):
        k = spikes[i]['cluster_label']
        if k<0: continue
        
        #~ cluster_idx = np.nonzero(catalogue['cluster_labels']==k)[0][0]
        cluster_idx = catalogue['label_to_index'][k]
        
        #~ print('make_prediction_signals', 'k', k, 'cluster_idx', cluster_idx)
        
        # prediction with no interpolation
        #~ wf0 = catalogue['centers0'][cluster_idx,:,:]
        #~ pred = wf0
        
        # predict with tailor approximate with derivative
        #~ wf1 = catalogue['centers1'][cluster_idx,:,:]
        #~ wf2 = catalogue['centers2'][cluster_idx]
        #~ pred = wf0 +jitter*wf1 + jitter**2/2*wf2
        
        #predict with with precilputed splin
        r = catalogue['subsample_ratio']
        pos = spikes[i]['index'] + catalogue['n_left']
        jitter = spikes[i]['jitter']
        #TODO debug that sign
        shift = -int(np.round(jitter))
        pos = pos + shift
        
        #~ if np.abs(jitter)>=0.5:
            #~ print('strange jitter', jitter)
        
        #TODO debug that sign
        #~ if shift >=1:
            #~ print('jitter', jitter, 'jitter+shift', jitter+shift, 'shift', shift)
        #~ int_jitter = int((jitter+shift)*r) + r//2
        int_jitter = int((jitter+shift)*r) + r//2
        #~ int_jitter = -int((jitter+shift)*r) + r//2
        
        #~ assert int_jitter>=0
        #~ assert int_jitter<r
        #TODO this is wrong we should move index first
        #~ int_jitter = max(int_jitter, 0)
        #~ int_jitter = min(int_jitter, r-1)
        
        pred = catalogue['interp_centers0'][cluster_idx, int_jitter::r, :]
        #~ print(pred.shape)
        #~ print(int_jitter, spikes[i]['jitter'])
        
        
        #~ print(prediction[pos:pos+catalogue['peak_width'], :].shape)
        
        
        if pos>=0 and  pos+catalogue['peak_width']<shape[0]:
            prediction[pos:pos+catalogue['peak_width'], :] += pred
        else:
            if not safe:
                print(spikes)
                n_left = catalogue['n_left']
                width = catalogue['peak_width']
                local_pos = spikes['index'] - np.round(spikes['jitter']).astype('int64') + n_left
                print(local_pos)
                #~ spikes['LABEL_LEFT_LIMIT'][(local_pos<0)] = LABEL_LEFT_LIMIT
                print('LEFT', (local_pos<0))
                #~ spikes['cluster_label'][(local_pos+width)>=shape[0]] = LABEL_RIGHT_LIMIT
                print('LABEL_RIGHT_LIMIT', (local_pos+width)>=shape[0])
                
                print('i', i)
                print(dtype, shape, catalogue['n_left'], catalogue['peak_width'], pred.shape)
                raise(ValueError('Border error {} {} {} {} {}'.format(pos, catalogue['peak_width'], shape, jitter, spikes[i])))
                
        
    return prediction


    
    
    
    

kernel_opencl = """

#define nb_channel %(nb_channel)d
#define peak_width %(peak_width)d
#define nb_cluster %(nb_cluster)d
#define total %(total)d
    
inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__kernel void waveform_distance(__global  float *one_waveform,
                                        __global  float *catalogue_center,
                                        __global  uchar  *mask,
                                        __global  float *rms_waveform_channel,
                                        __global  float *waveform_distance){
    
    int cluster_idx = get_global_id(0);
    int c = get_global_id(1);
    
    
    // initialize sum by cluster
    if (c==0){
        waveform_distance[cluster_idx] = 0;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    
    
    float sum = 0;
    float d;
    
    
    if (mask[nb_channel*cluster_idx+c]>0){
        for (int s=0; s<peak_width; ++s){
            d = one_waveform[nb_channel*s+c] - catalogue_center[total*cluster_idx+nb_channel*s+c];
            sum += d*d;
        }
    }
    else{
        sum = rms_waveform_channel[c];
    }
    
    AtomicAdd(&waveform_distance[cluster_idx], sum);
    
}


"""



    
