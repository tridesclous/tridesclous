import time
import numpy as np


from .peeler_tools import *
from .peeler_tools import _dtype_spike
from .tools import make_color_dict
from .signalpreprocessor import signalpreprocessor_engines
from .peakdetector import peakdetector_engines  #detect_peaks_in_chunk


from . import pythran_tools
if hasattr(pythran_tools, '__pythran__'):
    HAVE_PYTHRAN = True
else:
    HAVE_PYTHRAN = False

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_loop_sparse_dist
except ImportError:
    HAVE_NUMBA = False




from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags


# this should be an attribute
maximum_jitter_shift = 4
LABEL_NO_MORE_PEAK = -20

class PeelerEngineClassic(OpenCL_Helper):

    def change_params(self, catalogue=None, chunksize=1024, 
                                        internal_dtype='float32', 
                                        use_sparse_template=False,
                                        sparse_threshold_mad=1.5,
                                        argmin_method='numpy',
                                        
                                        cl_platform_index=None,
                                        cl_device_index=None,
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
        argmin_method: 'numpy', 'opencl', 'pythran' or 'numba'
            Method use to compute teh minial distance to template.
        """
        assert catalogue is not None
        self.catalogue = catalogue
        self.chunksize = chunksize
        self.internal_dtype= internal_dtype
        self.use_sparse_template = use_sparse_template
        self.sparse_threshold_mad = sparse_threshold_mad
        
        self.argmin_method = argmin_method
        
        # Some check
        if self.use_sparse_template:
            assert self.argmin_method != 'numpy', 'numpy methdo do not do sparse template acceleration'

            if self.argmin_method == 'opencl':
                assert HAVE_PYOPENCL, 'OpenCL is not available'
            elif self.argmin_method == 'pythran':
                assert HAVE_PYTHRAN, 'Pythran is not available'
            elif self.argmin_method == 'numba':
                assert HAVE_NUMBA, 'Numba is not available'
            
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
        
        
        #~ print('self.use_sparse_template', self.use_sparse_template)
        
        centers = self.catalogue['centers0']
        #~ print(centers.shape)
        if self.use_sparse_template:
            #~ print(centers.shape)
            # TODO use less memory
            self.sparse_mask = np.any(np.abs(centers)>sparse_threshold_mad, axis=1)
        else:
            self.sparse_mask = np.ones((centers.shape[0], centers.shape[2]), dtype='bool')
        
        #~ print('self.sparse_mask.shape', self.sparse_mask.shape)
        self.weight_per_template = {}
        for i, k in enumerate(self.catalogue['cluster_labels']):
            mask = self.sparse_mask[i, :]
            wf = centers[i, :, :][:, mask]
            self.weight_per_template[k] = np.sum(wf**2, axis=0)
            #~ print(wf.shape, self.weight_per_template[k].shape)

        #~ print(mask.shape)
        #~ print(mask)
        #~ print('average sparseness for templates', np.sum(mask)/mask.size)
        self.catalogue['sparse_mask'] = self.sparse_mask
        self.catalogue['weight_per_template'] = self.weight_per_template

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
        
        
        
        
        if self.use_sparse_template:
            
            if self.argmin_method == 'opencl'  and self.catalogue['centers0'].size>0:
            #~ if self.use_opencl_with_sparse and self.catalogue['centers0'].size>0:
                OpenCL_Helper.initialize_opencl(self, cl_platform_index=cl_platform_index, cl_device_index=cl_device_index)
                
                #~ self.ctx = pyopencl.create_some_context(interactive=False)
                #~ self.queue = pyopencl.CommandQueue(self.ctx)
                
                centers = self.catalogue['centers0']
                nb_channel = centers.shape[2]
                peak_width = centers.shape[1]
                nb_cluster = centers.shape[0]
                kernel = kernel_opencl%{'nb_channel': nb_channel,'peak_width':peak_width,
                                                        'wf_size':peak_width*nb_channel,'nb_cluster' : nb_cluster}
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
                self.sparse_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask.astype('u1'))

                rms_waveform_channel = np.zeros(nb_channel, dtype='float32')
                self.rms_waveform_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=rms_waveform_channel)
                
                self.cl_global_size = (centers.shape[0], centers.shape[2])
                #~ self.cl_local_size = None
                self.cl_local_size = (centers.shape[0], 1) # faster a GPU because of memory access
                #~ self.cl_local_size = (1, centers.shape[2])

    def initialize_before_each_segment(self, sample_rate=None, nb_channel=None, source_dtype=None, geometry=None):
        
        self.nb_channel = nb_channel
        self.sample_rate = sample_rate
        self.source_dtype = source_dtype
        self.geometry = geometry
        
        # signal processor class
        self.signalpreprocessor_engine = self.catalogue['signal_preprocessor_params']['signalpreprocessor_engine']
        SignalPreprocessor_class = signalpreprocessor_engines[self.signalpreprocessor_engine]
        self.signalpreprocessor = SignalPreprocessor_class(sample_rate, nb_channel, self.chunksize, source_dtype)
        p = dict(self.catalogue['signal_preprocessor_params'])
        p.pop('signalpreprocessor_engine')
        p['normalize'] = True
        p['signals_medians'] = self.catalogue['signals_medians']
        p['signals_mads'] = self.catalogue['signals_mads']
        self.signalpreprocessor.change_params(**p)
        
        self.internal_dtype = self.signalpreprocessor.output_dtype
        
        assert self.chunksize>self.signalpreprocessor.lostfront_chunksize, 'lostfront_chunksize ({}) is greater than chunksize ({})!'.format(self.signalpreprocessor.lostfront_chunksize, self.chunksize)

        # peak detecetor class
        p = dict(self.catalogue['peak_detector_params'])
        peakdetector_engine = p.pop('peakdetector_engine', 'numpy') # TODO put engine in info json back
        PeakDetector_class = peakdetector_engines[peakdetector_engine]
        self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        self.chunksize, self.internal_dtype, self.geometry)
        self.peakdetector.change_params(**p)

        self.peak_sign = self.catalogue['peak_detector_params']['peak_sign']
        self.relative_threshold = self.catalogue['peak_detector_params']['relative_threshold']
        peak_span_ms = self.catalogue['peak_detector_params']['peak_span_ms']
        self.n_span = int(sample_rate * peak_span_ms / 1000.)//2
        self.n_span = max(1, self.n_span)
        self.peak_width = self.catalogue['peak_width']
        self.n_side = self.catalogue['peak_width'] + maximum_jitter_shift + self.n_span + 1
        self.n_right = self.catalogue['n_right']
        self.n_left = self.catalogue['n_left']
        
        assert self.chunksize > (self.n_side+1), 'chunksize is too small because of n_size'
        
        self.alien_value_threshold = self.catalogue['clean_waveforms_params']['alien_value_threshold']
        
        self.total_spike = 0
        
        self.near_border_good_spikes = []
        
        self.fifo_residuals = np.zeros((self.n_side+self.chunksize, nb_channel), 
                                                                dtype=self.internal_dtype)
        
        self.mask_already_tested = np.ones(self.fifo_residuals.shape[0] - 2 * self.n_span, dtype='bool')

    #~ def NEW_process_one_chunk(self,  pos, sigs_chunk):
    def process_one_chunk(self,  pos, sigs_chunk):
        #~ print('*'*10)
        t1 = time.perf_counter()
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        #~ t2 = time.perf_counter()
        #~ print('process_data', (t2-t1)*1000)
        
        
        #shift rsiruals buffer and put the new one on right side
        t1 = time.perf_counter()
        fifo_roll_size = self.fifo_residuals.shape[0]-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_residuals.shape[0]:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk
        #~ t2 = time.perf_counter()
        #~ print('fifo move', (t2-t1)*1000.)

        
        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        
        
        # negative mask 1: not tested 0: already tested
        self.mask_already_tested[:] = True
        
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        
        
        good_spikes = []
        
        n_loop = 0
        t3 = time.perf_counter()
        while True:
            #~ print('peeler level +1')
            nb_good_spike = 0
            local_ind = self.select_next_peak()
            #~ print('start inner loop')
            while local_ind != LABEL_NO_MORE_PEAK:
            
                #~ print('  local_ind', local_ind)
                #~ t2 = time.perf_counter()
                #~ print('  select_next_peak', (t2-t1)*1000)
                
                #~ if local_ind == LABEL_NO_MORE_PEAK:
                    #~ print('break inner loop 1')
                    #~ break
                
                t1 = time.perf_counter()
                spike = self.classify_and_align_next_spike(local_ind)
                #~ t2 = time.perf_counter()
                #~ print('  classify_and_align_next_spike', (t2-t1)*1000)
                #~ print(spike.cluster_label)

                #~ print('spike', spike)
                
                
                
                if spike.cluster_label == LABEL_NO_MORE_PEAK:
                    #~ print('break inner loop 1')
                    break
                
                if (spike.cluster_label >=0):
                    #~ good_spikes.append(np.array([spike], dtype=_dtype_spike))
                    good_spikes.append(spike)
                    nb_good_spike+=1
                
                local_ind = self.select_next_peak()
                
                #~ # debug
                n_loop +=1 
                
                
                #~ import matplotlib.pyplot as plt
                #~ from .peakdetector import make_sum_rectified
                #~ # print('spike', spike)
                #~ fig, ax = plt.subplots()
                #~ ax.plot(self.fifo_residuals)
                #~ ax.plot(np.arange(self.mask_already_tested.size) + self.n_span, self.mask_already_tested.astype(float)*10, color='k')
                #~ local_peaks,  = np.nonzero(self.local_peaks_mask & self.mask_already_tested)
                #~ local_peaks += self.n_span
                #~ sum_rectified = make_sum_rectified(self.fifo_residuals, self.peakdetector.relative_threshold, self.peakdetector.peak_sign, self.peakdetector.spatial_matrix)
                #~ ax.scatter(local_peaks, np.min(self.fifo_residuals[local_peaks, :], axis=1), color='k')
                #~ # ax.plot(sum_rectified, color='k', lw=1.5)
                #~ # ax.scatter(local_peaks, sum_rectified[local_peaks], color='k')
                #~ for p in local_peaks:
                    #~ ax.axvline(p, color='k', ls='--')
                #~ ax.axvline(local_ind, color='r', ls='-')
                #~ ax.set_ylim(-300, 100)
                #~ plt.show()
            
            if nb_good_spike == 0:
                #~ print('break main loop')
                break
            else:
                
                t1 = time.perf_counter()
                for spike in good_spikes[-nb_good_spike:]:
                    local_ind = spike.index
                    sl1 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1 + self.n_span)
                    sl2 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1- self.n_span)
                    self.local_peaks_mask[sl2] = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals[sl1, :])
                    
                    # set neighboor untested
                    self.mask_already_tested[local_ind - self.peak_width - self.n_span:local_ind + self.peak_width - self.n_span] = True
                #~ t2 = time.perf_counter()
                #~ print('  update mask', (t2-t1)*1000)


        #~ t4 = time.perf_counter()
        #~ print('LOOP classify_and_align_one_spike', (t4-t3)*1000)
        #~ print('nb_good_spike', len(good_spikes), 'n_loop', n_loop)
        
        nolabel_indexes, = np.nonzero(~self.mask_already_tested)
        nolabel_indexes += self.n_span
        nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
        bad_spikes['index'] = nolabel_indexes + shift
        bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        
        if len(good_spikes)>0:
            # TODO remove from peak the very begining of the signal because of border filtering effects
            
            good_spikes = np.array(good_spikes, dtype=_dtype_spike)
            good_spikes['index'] += shift
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
    
    def select_next_peak(self):
        # TODO find faster
        local_peaks_indexes,  = np.nonzero(self.local_peaks_mask & self.mask_already_tested)
        #~ print('select_next_peak')
        #~ print(local_peaks_indexes + self.n_span )
        if local_peaks_indexes.size>0:
            local_peaks_indexes += self.n_span
            amplitudes = np.max(np.abs(self.fifo_residuals[local_peaks_indexes, :]), axis=1)
            #~ print(self.fifo_residuals[local_peaks_indexes, :])
            #~ print(amplitudes)
            #~ print(amplitudes.shape)
            ind = np.argmax(amplitudes)
            #~ print(ind)
            return local_peaks_indexes[ind]
            #~ return local_peaks_indexes[0] + self.n_span
        else:
            return LABEL_NO_MORE_PEAK
            
    
    def classify_and_align_next_spike(self, local_ind):
        #ind is the windows border!!!!!
        left_ind = local_ind + self.n_left

        if left_ind+self.peak_width+maximum_jitter_shift+1>=self.fifo_residuals.shape[0]:
            # too near right limits no label
            label = LABEL_RIGHT_LIMIT
            jitter = 0
        elif left_ind<=maximum_jitter_shift:
            # too near left limits no label
            #~ print('     LABEL_LEFT_LIMIT', left_ind)
            label = LABEL_LEFT_LIMIT
            jitter = 0
        elif self.catalogue['centers0'].shape[0]==0:
            # empty catalogue
            label  = LABEL_UNCLASSIFIED
            jitter = 0
        else:
            waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            
            if self.alien_value_threshold is not None and \
                    np.any((waveform>self.alien_value_threshold) | (waveform<-self.alien_value_threshold)) :
                label  = LABEL_ALIEN
                jitter = 0
            else:
                
                #~ t1 = time.perf_counter()
                cluster_idx = self.get_best_template(left_ind)
                #~ t2 = time.perf_counter()
                #~ print('    get_best_template', (t2-t1)*1000)
                
                #~ t1 = time.perf_counter()
                jitter = self.estimate_jitter(left_ind, cluster_idx)
                #~ t2 = time.perf_counter()
                #~ print('    estimate_jitter', (t2-t1)*1000)
                
                #~ t1 = time.perf_counter()
                ok = self.accept_tempate(left_ind, cluster_idx, jitter)
                #~ t2 = time.perf_counter()
                #~ print('    accept_tempate', (t2-t1)*1000)

                if  not ok:
                    label  = LABEL_UNCLASSIFIED
                    jitter = 0
                else:
                    #~ print('cluster_idx', cluster_idx, 'jitter', jitter)
                    shift = -int(np.round(jitter))
                    if (np.abs(jitter) > 0.5) and \
                            (left_ind+shift+self.peak_width<self.fifo_residuals.shape[0]) and\
                            ((left_ind + shift) >= 0):
                        shift = -int(np.round(jitter))
                        new_jitter = self.estimate_jitter(left_ind + shift, cluster_idx)
                        ok = self.accept_tempate(left_ind+shift, cluster_idx, jitter)
                        if ok and np.abs(new_jitter)<np.abs(jitter):
                            jitter = new_jitter
                            left_ind += shift
                            shift = -int(np.round(jitter))
                    
                    # security to not be outside the fifo
                    if np.abs(shift) >maximum_jitter_shift:
                        label = LABEL_MAXIMUM_SHIFT
                    elif (left_ind+shift+self.peak_width)>=self.fifo_residuals.shape[0]:
                        # normally this should be resolve in the next chunk
                        label = LABEL_RIGHT_LIMIT
                    elif (left_ind + shift) < 0:
                        # TODO assign the previous label ???
                        label = LABEL_LEFT_LIMIT
                    else:
                        label = self.catalogue['cluster_labels'][cluster_idx]

                        pos, pred = make_prediction_one_spike(left_ind - self.n_left, label, jitter, self.fifo_residuals.dtype, self.catalogue)
                        #~ print(pos, self.fifo_residuals.shape, pred.shape, 'left_ind', left_ind, self.peak_width, 'jitter', jitter, 'shift', -int(np.round(jitter)), 'label', label)
                        #~ print()
                        self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
                    
        local_ind = left_ind - self.n_left
        #security if with jitter the index is out
        if label>=0:
            local_pos = local_ind - np.round(jitter).astype('int64') + self.n_left
            if local_pos<0:
                label = LABEL_LEFT_LIMIT
            elif (local_pos+self.peak_width) >=self.fifo_residuals.shape[0]:
                label = LABEL_RIGHT_LIMIT
        
        if label < 0:
            # set peak tested to not test it again
            self.mask_already_tested[local_ind - self.n_span] = False

        #~ self.update_peak_mask(local_ind, label)
        #~ t2 = time.perf_counter()
        #~ print('    update_peak_mask', (t2-t1)*1000)
        
        
        return Spike(local_ind, label, jitter)





    def get_best_template(self, left_ind):
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        
        if self.argmin_method == 'opencl':
            rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
            
            pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)
            pyopencl.enqueue_copy(self.queue,  self.rms_waveform_channel_cl, rms_waveform_channel)
            event = self.kern_waveform_distance(self.queue,  self.cl_global_size, self.cl_local_size,
                        self.one_waveform_cl, self.catalogue_center_cl, self.sparse_mask_cl, 
                        self.rms_waveform_channel_cl, self.waveform_distance_cl)
            pyopencl.enqueue_copy(self.queue,  self.waveform_distance, self.waveform_distance_cl)
            cluster_idx = np.argmin(self.waveform_distance)
        
        elif self.argmin_method == 'pythran':
            s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                self.catalogue['centers0'],  self.catalogue['sparse_mask'])
            cluster_idx = np.argmin(s)
        
        elif self.argmin_method == 'numba':
            s = numba_loop_sparse_dist(waveform, self.catalogue['centers0'],  self.catalogue['sparse_mask'])
            cluster_idx = np.argmin(s)
        
        elif self.argmin_method == 'numpy':
            # replace by this (indentique but faster, a but)
            d = self.catalogue['centers0']-waveform[None, :, :]
            d *= d
            #s = d.sum(axis=1).sum(axis=1)  # intuitive
            #s = d.reshape(d.shape[0], -1).sum(axis=1) # a bit faster
            s = np.einsum('ijk->i', d) # a bit faster
            cluster_idx = np.argmin(s)
        else:
            raise(NotImplementedError())
        
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        return cluster_idx

    
    
    def estimate_jitter(self, left_ind, cluster_idx):
        
        chan_max = self.catalogue['max_on_channel'][cluster_idx]
        
        wf0 = self.catalogue['centers0'][cluster_idx,: , chan_max]
        wf1 = self.catalogue['centers1'][cluster_idx,: , chan_max]
        wf2 = self.catalogue['centers2'][cluster_idx,: , chan_max]

        wf = self.fifo_residuals[left_ind:left_ind+self.peak_width,chan_max]
        
        
        #it is  precompute that at init for speedup
        wf1_norm2= self.catalogue['wf1_norm2'][cluster_idx]
        wf2_norm2 = self.catalogue['wf2_norm2'][cluster_idx]
        wf1_dot_wf2 = self.catalogue['wf1_dot_wf2'][cluster_idx]
        
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
        
        return jitter1

    def accept_tempate(self, left_ind, cluster_idx, jitter):
        # criteria mono channel = old implementation
        #~ keep_template = np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2)
        
        # criteria multi channel
        mask = self.catalogue['sparse_mask'][cluster_idx]
        full_wf0 = self.catalogue['centers0'][cluster_idx,: , :][:, mask]
        full_wf1 = self.catalogue['centers1'][cluster_idx,: , :][:, mask]
        full_wf2 = self.catalogue['centers2'][cluster_idx,: , :][:, mask]
        
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        full_wf = waveform[:, :][:, mask]
        label = self.catalogue['cluster_labels'][cluster_idx]
        weight = self.weight_per_template[label]
        wf_nrj = np.sum(full_wf**2, axis=0)
        res_nrj = np.sum((full_wf-(full_wf0+jitter*full_wf1+jitter**2/2*full_wf2))**2, axis=0)
        # criteria per channel
        crietria_weighted = (wf_nrj>res_nrj).astype('float') * weight
        accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        
        return accept_template


    def get_remaining_spikes(self):
        if len(self.near_border_good_spikes)>0:
            # deal with extra remaining spikes
            extra_spikes = self.near_border_good_spikes[0]
            extra_spikes = extra_spikes.take(np.argsort(extra_spikes['index']))
            self.total_spike += extra_spikes.size
            return extra_spikes

#########################################

    def OLD_process_one_chunk(self,  pos, sigs_chunk):
    #~ def process_one_chunk(self,  pos, sigs_chunk):
    
        #~ print('*'*5)
        #~ print('chunksize', self.chunksize, '=', self.chunksize/self.sample_rate*1000, 'ms')
        
        t1 = time.perf_counter()
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        #~ t2 = time.perf_counter()
        #~ print('process_data', (t2-t1)*1000)
        
        
        #note abs_head_index is smaller than pos because prepcorcessed chunk
        # is late because of local filfilt in signalpreprocessor
        
        #shift rsiruals buffer and put the new one on right side
        t1 = time.perf_counter()
        fifo_roll_size = self.fifo_residuals.shape[0]-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_residuals.shape[0]:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk
        #~ t2 = time.perf_counter()
        #~ print('fifo move', (t2-t1)*1000.)

        
        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        # TODO remove from peak the very begining of the signal because of border filtering effects
        
     
        good_spikes = []
        #~ already_tested = []
        
        # negative mask 1: not tested 0: already tested
        mask_already_tested = np.ones(self.fifo_residuals.shape[0] - 2 * self.n_span, dtype='bool')
        
        local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        #~ print('sum(local_peaks_mask)', np.sum(local_peaks_mask))
        
        n_loop = 0
        t3 = time.perf_counter()
        while True:
            #detect peaks
            #~ t3 = time.perf_counter()
            
            #~ local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
            
            local_peaks_indexes,  = np.nonzero(local_peaks_mask & mask_already_tested)
            #~ print(local_peaks_indexes)
            local_peaks_indexes += self.n_span
            #~ exit()
            
            #~ t4 = time.perf_counter()
            #~ print('  detect_peaks_in_chunk', (t4-t3)*1000.)pythran_loop_sparse_dist
            
            #~ if len(already_tested)>0:
                #~ local_peaks_to_check = local_peaks_indexes[~np.in1d(local_peaks_indexes, already_tested)]
            #~ else:
                #~ local_peaks_to_check = local_peaks_indexes
            
            n_ok = 0
            for local_ind in local_peaks_indexes:
                #~ print('    local_peak', local_peak, 'i', i)
                t1 = time.perf_counter()
                spike = self.classify_and_align_one_spike(local_ind, self.fifo_residuals, self.catalogue)
                t2 = time.perf_counter()
                #~ print('    classify_and_align_one_spike', (t2-t1)*1000., spike.cluster_label)
                
                if spike.cluster_label>=0:
                    #~ print('     >>spike.index', spike.index, spike.cluster_label, 'abs index', spike.index+shift)
                    
                    
                    #~ spikes = np.array([spike], dtype=_dtype_spike)
                    #~ prediction = make_prediction_signals(spikes, self.fifo_residuals.dtype, self.fifo_residuals.shape, self.catalogue, safe=False)
                    #~ self.fifo_residuals -= prediction
                    #~ spikes['index'] += shift
                    #~ good_spikes.append(spikes)
                    #~ n_ok += 1
                    
                    # substract one spike
                    pos, pred = make_prediction_one_spike(spike.index, spike.cluster_label, spike.jitter, self.fifo_residuals.dtype, self.catalogue)
                    self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
                    
                    # append
                    spikes = np.array([spike], dtype=_dtype_spike)
                    spikes['index'] += shift
                    good_spikes.append(spikes)
                    n_ok += 1
                                    
                    # recompute peak in neiborhood
                    # here indexing is tricky 
                    # sl1 : we need n_pan more in each side
                    # sl2: we need a shift of n_span because smaler shape
                    sl1 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1 + self.n_span)
                    sl2 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1- self.n_span)
                    local_peaks_mask[sl2] = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals[sl1, :])
                    
                    
                    #~ print('    already_tested before', already_tested)
                    #~ already_tested = [ind for ind in already_tested if np.abs(spike.index-ind)>self.peak_width]
                    
                    # set neighboor untested
                    mask_already_tested[local_ind - self.peak_width - self.n_span:local_ind + self.peak_width - self.n_span] = True

                    
                    #~ print('    already_tested new deal', already_tested)
                else:
                    # set peak tested
                    #~ print(mask_already_tested.shape)
                    #~ print(self.fifo_residuals.shape)
                    #~ print(self.n_span)
                    mask_already_tested[local_ind - self.n_span] = False
                    #~ print('already tested', local_ind)
                    #~ already_tested.append(local_peak)
                n_loop += 1
            
            if n_ok==0:
                # no peak can be labeled
                # reserve bad spikes on the right limit for next time

                #~ local_peaks_indexes = local_peaks_indexes[local_peaks_indexes<(self.chunksize+self.n_span)]
                #~ bad_spikes = np.zeros(local_peaks_indexes.shape[0], dtype=_dtype_spike)
                #~ bad_spikes['index'] = local_peaks_indexes + shift
                #~ bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                
                nolabel_indexes, = np.nonzero(~mask_already_tested)
                nolabel_indexes += self.n_span
                nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
                bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
                bad_spikes['index'] = nolabel_indexes + shift
                bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                
                break
        
        #~ t4 = time.perf_counter()
        #~ print('LOOP classify_and_align_one_spike', (t4-t3)*1000)
        #~ print('n_good', len(good_spikes), 'n_loop', n_loop)
        
        
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
    
    
        
    
    

    
    def classify_and_align_one_spike(self, local_index, residual, catalogue):
        # local_index is index of peaks inside residual and not
        # the absolute peak_pos. So time scaling must be done outside.
        
        peak_width = catalogue['peak_width']
        n_left = catalogue['n_left']
        #~ alien_value_threshold = catalogue['clean_waveforms_params']['alien_value_threshold']
        
        
        #ind is the windows border!!!!!
        left_ind = local_index + n_left

        if left_ind+peak_width+maximum_jitter_shift+1>=residual.shape[0]:
            # too near right limits no label
            label = LABEL_RIGHT_LIMIT
            jitter = 0
        elif left_ind<=maximum_jitter_shift:
            # too near left limits no label
            #~ print('     LABEL_LEFT_LIMIT', left_ind)
            label = LABEL_LEFT_LIMIT
            jitter = 0
        elif catalogue['centers0'].shape[0]==0:
            # empty catalogue
            label  = LABEL_UNCLASSIFIED
            jitter = 0
        else:
            waveform = residual[left_ind:left_ind+peak_width,:]
            
            if self.alien_value_threshold is not None and \
                    np.any((waveform>self.alien_value_threshold) | (waveform<-self.alien_value_threshold)) :
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
                    prev_ind, prev_label, prev_jitter =left_ind, label, jitter
                    
                    shift = -int(np.round(jitter))
                    #~ print('classify and align shift', shift)
                    
                    if np.abs(shift) >maximum_jitter_shift:
                        #~ print('     LABEL_MAXIMUM_SHIFT avec shift')
                        label = LABEL_MAXIMUM_SHIFT
                    else:
                        left_ind = left_ind + shift
                        if left_ind+peak_width>=residual.shape[0]:
                            #~ print('     LABEL_RIGHT_LIMIT avec shift')
                            label = LABEL_RIGHT_LIMIT
                        elif left_ind < 0:
                            #~ print('     LABEL_LEFT_LIMIT avec shift')
                            label = LABEL_LEFT_LIMIT
                            #TODO: force to label anyway the spike if spike is at the left of FIFO
                        else:
                            waveform = residual[left_ind:left_ind+peak_width,:]
                            #~ print('    second estimate jitter')
                            new_label, new_jitter = self.estimate_one_jitter(waveform, label=label)
                            #~ new_label, new_jitter = self.estimate_one_jitter(waveform, label=None)
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
            elif (local_pos+peak_width) >=residual.shape[0]:
                label = LABEL_RIGHT_LIMIT
        
        return Spike(local_index, label, jitter)
    
    
    def estimate_one_jitter(self, waveform, label=None):
        """
        Estimate the jitter for one peak given its waveform
        
        label=None general case
        label not None when estimate_one_jitter is the second call
        frequently happen when abs(jitter)>0.5
        
        
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
        
        if label is None:
            #~ if self.use_opencl_with_sparse:
            if self.argmin_method == 'opencl':
                t1 = time.perf_counter()
                rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
                
                pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)
                pyopencl.enqueue_copy(self.queue,  self.rms_waveform_channel_cl, rms_waveform_channel)
                event = self.kern_waveform_distance(self.queue,  self.cl_global_size, self.cl_local_size,
                            self.one_waveform_cl, self.catalogue_center_cl, self.sparse_mask_cl, 
                            self.rms_waveform_channel_cl, self.waveform_distance_cl)
                pyopencl.enqueue_copy(self.queue,  self.waveform_distance, self.waveform_distance_cl)
                cluster_idx = np.argmin(self.waveform_distance)
                t2 = time.perf_counter()
                #~ print('       np.argmin opencl_with_sparse', (t2-t1)*1000., cluster_idx)


            #~ elif self.use_pythran_with_sparse:
            elif self.argmin_method == 'pythran':
                s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                    catalogue['centers0'],  catalogue['sparse_mask'])
                cluster_idx = np.argmin(s)
            elif self.argmin_method == 'numba':
                s = numba_loop_sparse_dist(waveform, catalogue['centers0'],  catalogue['sparse_mask'])
                cluster_idx = np.argmin(s)
            
            elif self.argmin_method == 'numpy':
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
            else:
                raise(NotImplementedError())
            
            k = catalogue['cluster_labels'][cluster_idx]
        else:
            t1 = time.perf_counter()
            cluster_idx = np.nonzero(catalogue['cluster_labels']==label)[0][0]
            k = label
            t2 = time.perf_counter()
            #~ print('       second argmin', (t2-t1)*1000., cluster_idx)
        
        
        chan_max = catalogue['max_on_channel'][cluster_idx]
        #~ print('cluster_idx', cluster_idx, 'k', k, 'chan', chan)

        
        #~ return k, 0.

        wf0 = catalogue['centers0'][cluster_idx,: , chan_max]
        wf1 = catalogue['centers1'][cluster_idx,: , chan_max]
        wf2 = catalogue['centers2'][cluster_idx,: , chan_max]
        wf = waveform[:, chan_max]
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
        
        
        
        # criteria mono channel = old implementation
        #~ keep_template = np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2)
        
        # criteria multi channel
        mask = catalogue['sparse_mask'][cluster_idx]
        full_wf0 = catalogue['centers0'][cluster_idx,: , :][:, mask]
        full_wf1 = catalogue['centers1'][cluster_idx,: , :][:, mask]
        full_wf2 = catalogue['centers2'][cluster_idx,: , :][:, mask]
        full_wf = waveform[:, :][:, mask]
        weight = self.weight_per_template[k]
        wf_nrj = np.sum(full_wf**2, axis=0)
        res_nrj = np.sum((full_wf-(full_wf0+jitter1*full_wf1+jitter1**2/2*full_wf2))**2, axis=0)
        # criteria per channel
        crietria_weighted = (wf_nrj>res_nrj).astype('float') * weight
        accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        
        
        if accept_template:
            # keep prediction
            return k, jitter1
        else:
            #otherwise the prediction is bad
            return LABEL_UNCLASSIFIED, 0.


kernel_opencl = """

#define nb_channel %(nb_channel)d
#define peak_width %(peak_width)d
#define nb_cluster %(nb_cluster)d
#define wf_size %(wf_size)d
    
inline void atomic_add_float(volatile __global float *source, const float operand) {
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
                                        __global  uchar  *sparse_mask,
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
    
    
    if (sparse_mask[nb_channel*cluster_idx+c]>0){
        for (int s=0; s<peak_width; ++s){
            d = one_waveform[nb_channel*s+c] - catalogue_center[wf_size*cluster_idx+nb_channel*s+c];
            sum += d*d;
        }
    }
    else{
        sum = rms_waveform_channel[c];
    }
    
    atomic_add_float(&waveform_distance[cluster_idx], sum);
    
}


"""


