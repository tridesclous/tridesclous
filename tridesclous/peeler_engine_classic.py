import time
import numpy as np


from .peeler_tools import *
from .peeler_tools import _dtype_spike
from .peeler_engine_base import PeelerEngineGeneric

from .peakdetector import get_peak_detector_class


import matplotlib.pyplot as plt


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





class PeelerEngineClassic(PeelerEngineGeneric):
    
    def change_params(self, **kargs):
        PeelerEngineGeneric.change_params(self, **kargs)

        if self.argmin_method == 'opencl'  and self.catalogue['centers0'].size>0:
        #~ if self.use_opencl_with_sparse and self.catalogue['centers0'].size>0:
            OpenCL_Helper.initialize_opencl(self, cl_platform_index=self.cl_platform_index, cl_device_index=self.cl_device_index)
            
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


    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        
        # force engine to global
        p = dict(self.catalogue['peak_detector_params'])
        p.pop('engine')
        p.pop('method')
        
        self.peakdetector_method = 'global'
        self.peakdetector_engine = 'numpy'
        PeakDetector_class = get_peak_detector_class(self.peakdetector_method, self.peakdetector_engine)
        
        chunksize = self.fifo_size-2*self.n_span # not the real chunksize here
        self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        chunksize, self.internal_dtype, self.geometry)
        self.peakdetector.change_params(**p)
        
        self.mask_not_already_tested = np.ones(self.fifo_size - 2 * self.n_span, dtype='bool')
        

    def detect_local_peaks_before_peeling_loop(self):
        # negative mask 1: not tested 0: already tested
        self.mask_not_already_tested[:] = True
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        
        
        
        #~ peak_inds,  =  np.nonzero(self.local_peaks_mask )
        #~ peak_chans =  np.argmin(self.fifo_residuals[peak_inds, :], axis=1)
        #~ peak_inds = peak_inds + self.n_span
        #~ fig, ax = plt.subplots()
        #~ plot_sigs = self.fifo_residuals.copy()
        #~ for c in range(self.nb_channel):
            #~ plot_sigs[:, c] += c*30
        #~ ax.plot(plot_sigs, color='k')
        #~ ampl = plot_sigs[peak_inds, peak_chans]
        #~ ax.scatter(peak_inds, ampl, color='r')
        #~ for peak_ind in peak_inds:
            #~ ax.axvline(peak_ind)
        
        #~ plt.show()        


    
    def select_next_peak(self):
        # TODO find faster
        
        local_peaks_indexes,  = np.nonzero(self.local_peaks_mask & self.mask_not_already_tested)
        if self._plot_debug:
            print('select_next_peak', local_peaks_indexes + self.n_span)
        
        #~ print(local_peaks_indexes.size)
        #~ print('select_next_peak')
        #~ print(local_peaks_indexes + self.n_span )
        if local_peaks_indexes.size>0:
            local_peaks_indexes += self.n_span
            #~ if self._plot_debug:
                #~ print('select_next_peak', local_peaks_indexes)
            amplitudes = np.max(np.abs(self.fifo_residuals[local_peaks_indexes, :]), axis=1)
            ind = np.argmax(amplitudes)
            return local_peaks_indexes[ind], None
            #~ return local_peaks_indexes[0]
        else:
            return LABEL_NO_MORE_PEAK, None
    




    def on_accepted_spike(self, spike):
        # remove spike prediction from fifo residuals
        left_ind = spike.index + self.n_left
        cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
        pos, pred = make_prediction_one_spike(spike.index, cluster_idx, spike.jitter, self.fifo_residuals.dtype, self.catalogue)
        self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
        
        # this prevent search peaks in the zone until next "reset_to_not_tested"
        self.set_already_tested_spike_zone(spike.index, cluster_idx)
    
    
    def set_already_tested(self, peak_ind, peak_chan):
        self.mask_not_already_tested[peak_ind - self.n_span] = False
    
    def set_already_tested_spike_zone(self, peak_ind, cluster_idx):
        self.mask_not_already_tested[peak_ind + self.n_left - self.n_span:peak_ind + self.n_right- self.n_span] = False
        
    def reset_to_not_tested(self, good_spikes):
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        #~ self.mask_not_already_tested[:] = True
        for spike in good_spikes:
            peak_ind = spike.index
            self.mask_not_already_tested[peak_ind + self.n_left - self.n_span:peak_ind + self.n_right- self.n_span] = True
            
        #~ for spike in good_spikes:
            #~ peak_ind = spike.index
            #~ # TODO here make enlarge a bit with maximum_jitter_shift
            #~ sl1 = slice(peak_ind + self.n_left - 1 - self.n_span, peak_ind + self.n_right + 1 + self.n_span)
            #~ sl2 = slice(peak_ind + self.n_left - 1 - self.n_span, peak_ind + self.n_right + 1- self.n_span)
            #~ self.local_peaks_mask[sl2] = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals[sl1, :])
            
            #~ # set neighboor untested
            #~ self.mask_not_already_tested[peak_ind - self.peak_width - self.n_span:peak_ind + self.peak_width - self.n_span] = True
    
    def get_no_label_peaks(self):
        # nolabel_indexes, = np.nonzero(~self.mask_not_already_tested)
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        nolabel_indexes, = np.nonzero(mask)
        nolabel_indexes += self.n_span
        nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
        bad_spikes['index'] = nolabel_indexes
        bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        return bad_spikes
        


    def get_best_template(self, left_ind, peak_chan):
        
        assert peak_chan is None
        
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
            shift = None
        
        elif self.argmin_method == 'pythran':
            s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                self.catalogue['centers0'],  self.sparse_mask)
            cluster_idx = np.argmin(s)
            shift = None
        
        elif self.argmin_method == 'numba':
            #~ s = numba_loop_sparse_dist(waveform, self.catalogue['centers0'],  self.sparse_mask)
            #~ cluster_idx = np.argmin(s)
            #~ shift = None
            
            shifts = list(range(-self.maximum_jitter_shift, self.maximum_jitter_shift+1))
            all_s = []
            for shift in shifts:
                waveform = self.fifo_residuals[left_ind+shift:left_ind+self.peak_width+shift,:]
                s = numba_loop_sparse_dist(waveform, self.catalogue['centers0'],  self.sparse_mask)
                all_s.append(s)
            all_s = np.array(all_s)
            shift_ind, cluster_idx = np.unravel_index(np.argmin(all_s, axis=None), all_s.shape)
            shift = shifts[shift_ind]
            #~ print(shift, cluster_idx)
            
            
            #~ if self._plot_debug:
                #~ fig, ax = plt.subplots()
                #~ ax.plot(shifts, all_s, marker='o')
                #~ ax.set_title(f'{left_ind-self.n_left} {shift}')
            

            #~ s = numba_loop_sparse_dist(waveform, self.catalogue['centers0'],  self.sparse_mask)
            #~ cluster_idx = np.argmin(s)
        
        elif self.argmin_method == 'numpy':
            # replace by this (indentique but faster, a but)
            d = self.catalogue['centers0']-waveform[None, :, :]
            d *= d
            #s = d.sum(axis=1).sum(axis=1)  # intuitive
            #s = d.reshape(d.shape[0], -1).sum(axis=1) # a bit faster
            s = np.einsum('ijk->i', d) # a bit faster
            cluster_idx = np.argmin(s)
            shift = None
        else:
            raise(NotImplementedError())
        
        #~ print('get_best_template', left_ind-self.n_left)
        #~ if 16000 < (left_ind-self.n_left) <16400:
        
        #~ if self._plot_debug:
            #~ fig, ax = plt.subplots()
            #~ chan_order = np.argsort(self.distances[0, :])
            #~ channels = self.channels_adjacency[chan_ind]
            #~ channels = chan_order
            #~ wf = waveform[:, channels]
            #~ wf0 = self.catalogue['centers0'][cluster_idx, :, :][:, channels]
            #~ wf = waveform
            #~ wf0 = self.catalogue['centers0'][cluster_idx, :, :]
            #~ wf= waveform
            
            #~ ax.plot(wf.T.flatten(), color='k')
            #~ ax.plot(wf0.T.flatten(), color='m')
            
            #~ plot_chan = channels.tolist().index(chan_ind)
            #~ plot_chan = chan_ind
            #~ ax.axvline(plot_chan * self.peak_width - self.n_left)
            #~ ax.set_title(f'cluster_idx {cluster_idx}')
            
            #~ plt.show()
            
        
        
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        return cluster_idx, shift

    
    


    def accept_tempate(self, left_ind, cluster_idx, jitter):
        #~ self._debug_nb_accept_tempate += 1
        #~ import matplotlib.pyplot as plt
        # criteria mono channel = old implementation
        #~ keep_template = np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2)
        
        if jitter is None:
            # this must have a jitter
            jitter = 0
        
        if np.abs(jitter) > (self.maximum_jitter_shift - 0.5):
            return False
        
        # criteria multi channel
        mask = self.sparse_mask[cluster_idx]
        full_wf0 = self.catalogue['centers0'][cluster_idx,: , :][:, mask]
        full_wf1 = self.catalogue['centers1'][cluster_idx,: , :][:, mask]
        full_wf2 = self.catalogue['centers2'][cluster_idx,: , :][:, mask]
        
        # waveform L2 on mask
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        full_wf = waveform[:, :][:, mask]
        wf_nrj = np.sum(full_wf**2, axis=0)
        
        # prediction L2 on mask
        label = self.catalogue['cluster_labels'][cluster_idx]
        weight = self.weight_per_template[label]
        pred_wf = (full_wf0+jitter*full_wf1+jitter**2/2*full_wf2)
        
        residual_nrj = np.sum((full_wf-pred_wf)**2, axis=0)
        
        # criteria per channel
        #~ thresh = 0.9
        thresh = 0.7
        #~ thresh = 0.5
        
        crietria_weighted = (wf_nrj>residual_nrj).astype('float') * weight
        accept_template = np.sum(crietria_weighted) >= thresh * np.sum(weight)

        label = self.catalogue['clusters'][cluster_idx]['cluster_label']
        #~ if True:
        #~ if np.random.rand()<0.05:
        #~ if label == 151:
        #~ if self._plot_debug:
            #~ print('label == 151', 'cluster_idx', cluster_idx)
            
            #~ max_chan_ind = self.catalogue['clusters'][cluster_idx]['extremum_channel']
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.fifo_residuals[:, max_chan_ind])
            
            #~ ax.scatter([left_ind-self.n_left], [self.fifo_residuals[left_ind-self.n_left, max_chan_ind]], color='r')
            
            #~ fig, axs = plt.subplots(nrows=2, sharex=True)
            #~ axs[0].plot(full_wf.T.flatten(), color='b')
            #~ if accept_template:
                #~ axs[0].plot(pred_wf.T.flatten(), color='g')
            #~ else:
                #~ axs[0].plot(pred_wf.T.flatten(), color='r')
            
            #~ axs[0].plot((full_wf-pred_wf).T.flatten(), color='m')
            
            #~ plt.show()

            



        
        #DEBUG
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        #~ if label in (10, ):
            
            #~ print('accept_tempate',accept_template, 'label', label)
            #~ print(wf_nrj>res_nrj)
            #~ print(weight)
            #~ print(crietria_weighted)
            #~ print(np.sum(crietria_weighted), np.sum(weight), np.sum(crietria_weighted)/np.sum(weight))
            
            #~ print()
        #~ #ENDDEBUG
        
        
        return accept_template
    
    
    def _plot_before_peeling_loop(self):
        fig, ax = plt.subplots()
        plot_sigs = self.fifo_residuals.copy()
        
        self._plot_sigs_before = plot_sigs
        #~ chan_order = np.argsort(self.distances[0, :])
        
        for c in range(self.nb_channel):
        #~ for c in chan_order:
            plot_sigs[:, c] += c*30
        
        ax.plot(plot_sigs, color='k')

        ax.axvline(self.fifo_size - self.n_right, color='r')
        ax.axvline(-self.n_left, color='r')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        nolabel_indexes, = np.nonzero(mask)
        nolabel_indexes += self.n_span
        
        for ind in nolabel_indexes:
            ax.axvline(ind, ls='--')
        
        #~ plt.show()
    
    def _plot_label_unclassified(self, left_ind, peak_chan, cluster_idx, jitter):
        fig, ax = plt.subplots()
        
        wf = self.fifo_residuals[left_ind:left_ind+self.peak_width, :]
        wf0 = self.catalogue['centers0'][cluster_idx, :, :]
        
        ax.plot(wf.T.flatten(), color='b')
        ax.plot(wf0.T.flatten(), color='g')
        
        ax.set_title(f'label_unclassified {left_ind-self.n_left} {cluster_idx}')

    def _plot_after_peeling_loop(self, good_spikes):
        fig, ax = plt.subplots()
        plot_sigs = self.fifo_residuals.copy()
        
        
        #~ chan_order = np.argsort(self.distances[0, :])
        
        for c in range(self.nb_channel):
        #~ for c in chan_order:
            plot_sigs[:, c] += c*30
        ax.plot(plot_sigs, color='k')
        
        ax.plot(self._plot_sigs_before, color='b')
        
        ax.axvline(self.fifo_size - self.n_right, color='r')
        ax.axvline(-self.n_left, color='r')

        for ind in np.nonzero(~self.mask_not_already_tested)[0] + self.n_span:
            ax.axvline(ind, ls='-', color='g')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        nolabel_indexes, = np.nonzero(mask)
        nolabel_indexes += self.n_span
        
        for ind in nolabel_indexes:
            ax.axvline(ind, ls='--')
        
        
        
        #~ ax.scatter(nolabel_indexes, plot_sigs[nolabel_indexes, chan_indexes], color='r')
        
        good_spikes = np.array(good_spikes, dtype=_dtype_spike)
        pred = make_prediction_signals(good_spikes, self.internal_dtype, plot_sigs.shape, self.catalogue, safe=True)
        plot_pred = pred.copy()
        for c in range(self.nb_channel):
        #~ for c in chan_order:
            plot_pred[:, c] += c*30
        
        ax.plot(plot_pred, color='m')
        
        plt.show()


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


