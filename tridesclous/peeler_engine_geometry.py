"""
Here implementation that tale in account the geometry
of the probe to speed up template matching.

"""

import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor



from .peeler_engine_base import PeelerEngineGeneric

from .peeler_tools import *
from .peeler_tools import _dtype_spike

import sklearn.metrics.pairwise

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags

from .peakdetector import PeakDetectorSpatiotemporal, PeakDetectorSpatiotemporal_OpenCL

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_loop_sparse_dist_with_geometry
except ImportError:
    HAVE_NUMBA = False




import matplotlib.pyplot as plt


class PeelerEngineGeometry(PeelerEngineGeneric):
    def change_params(self, adjacency_radius_um=200, **kargs):
        PeelerEngineGeneric.change_params(self, **kargs)
        
        self.adjacency_radius_um = adjacency_radius_um
        
        assert self.use_sparse_template

    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)

        p = dict(self.catalogue['peak_detector_params'])
        _ = p.pop('peakdetector_engine', 'numpy')
        #~ self.peakdetector = PeakDetectorSpatiotemporal(self.sample_rate, self.nb_channel,
                        #~ self.chunksize+self.n_side, self.internal_dtype, self.geometry)
        
        # TODO size fido
        self.peakdetector = PeakDetectorSpatiotemporal_OpenCL(self.sample_rate, self.nb_channel,
                                                        self.fifo_residuals.shape[0]-2*self.n_span, self.internal_dtype, self.geometry)
        self.peakdetector.change_params(**p)

        
        self.mask_not_already_tested = np.ones((self.fifo_residuals.shape[0] - 2 * self.n_span,self.nb_channel),  dtype='bool')

        d = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
        self.channels_adjacency = {}
        for c in range(self.nb_channel):
            nearest, = np.nonzero(d[c, :]<self.adjacency_radius_um)
            self.channels_adjacency[c] = nearest
            #~ print(c, nearest)

    def select_next_peak(self):
        #~ print('select_next_peak')
        # TODO find faster
        local_peaks_indexes, chan_indexes  = np.nonzero(self.local_peaks_mask & self.mask_not_already_tested)
        print(local_peaks_indexes.size)
        #~ print('select_next_peak')
        #~ print(local_peaks_indexes + self.n_span )
        if local_peaks_indexes.size>0:
            local_peaks_indexes += self.n_span
            amplitudes = np.abs(self.fifo_residuals[local_peaks_indexes, chan_indexes])
            ind = np.argmax(amplitudes)
            peak_ind = local_peaks_indexes[ind]
            chan_ind = chan_indexes[ind]
            return peak_ind, chan_ind
        else:
            return LABEL_NO_MORE_PEAK, None

    def on_accepted_spike(self, spike):
        # remove spike prediction from fifo residuals
        left_ind = spike.index + self.n_left
        cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
        pos, pred = make_prediction_one_spike(spike.index, cluster_idx, spike.jitter, self.fifo_residuals.dtype, self.catalogue)
        self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
        
        # this prevent search peaks in the zone until next "reset_to_not_tested"
        self.set_already_tested_spike_zone(spike.index, cluster_idx)


    def set_already_tested_spike_zone(self, peak_ind, cluster_idx):
        mask = self.sparse_mask[cluster_idx, :]
        for c in np.nonzero(mask)[0]:
            self.mask_not_already_tested[peak_ind + self.n_left - self.n_span:peak_ind + self.n_right- self.n_span, c] = False
    
    def set_already_tested(self, peak_ind, peak_chan):
        self.mask_not_already_tested[peak_ind - self.n_span, peak_chan] = False


    def reset_to_not_tested(self, good_spikes):
        #TODO : more efficient only local !!!!
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)

        for spike in good_spikes:
            peak_ind = spike.index
            cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
            mask = self.sparse_mask[cluster_idx, :]
            for c in np.nonzero(mask)[0]:
                self.mask_not_already_tested[peak_ind - self.peak_width - self.n_span:peak_ind + self.peak_width - self.n_span, c] = True
    
    def get_no_label_peaks(self):
        nolabel_indexes, chan_indexes = np.nonzero(~self.mask_not_already_tested)
        nolabel_indexes += self.n_span
        nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
        bad_spikes['index'] = nolabel_indexes
        bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        return bad_spikes

    def get_best_template(self, left_ind, chan_ind):
        
        assert self.argmin_method == 'numba'
        
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]

        #~ print('get_best_template')
        #~ print('chan_ind', chan_ind)
        
        # TODO 
        
        possibles_cluster_idx, = np.nonzero(self.sparse_mask[:, chan_ind])
        #~ print(possibles_cluster_idx)
        #~ print('possibles_cluster_idx', possibles_cluster_idx.size)
        
        #~ print(self.sparse_mask.shape)
        
        #~ fig, ax = plt.subplots()
        #~ ax.plot(self.fifo_residuals[:, chan_ind])
        #~ ax.scatter([left_ind-self.n_left], [self.fifo_residuals[left_ind-self.n_left, chan_ind]], color='r')
        #~ plt.show()
        
        
        #~ for cluster_idx in possibles_cluster_idx:
            #~ fig, ax = plt.subplots()
            #~ ax.plot(waveform.T.flatten(), color='k')
            #~ ax.plot(self.catalogue['centers0'][cluster_idx, :, :].T.flatten(), color='m')
            #~ ax.set_title(str(cluster_idx))
            #~ plt.show()
            
            
        
        
        
        
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
                                self.catalogue['centers0'],  self.sparse_mask)
            cluster_idx = np.argmin(s)
        
        elif self.argmin_method == 'numba':
            s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  self.sparse_mask, possibles_cluster_idx, self.channels_adjacency[chan_ind])
            cluster_idx = possibles_cluster_idx[np.argmin(s)]
        
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
        
        
        #~ print('cluster_idx', cluster_idx)
        #~ print('chan_ind', chan_ind)
        
        
        #~ fig, ax = plt.subplots()
        #~ channels = self.channels_adjacency[chan_ind]
        #~ wf = waveform[:, channels]
        #~ wf0 = self.catalogue['centers0'][cluster_idx, :, :][:, channels]
        #~ wf = waveform
        #~ wf0 = self.catalogue['centers0'][cluster_idx, :, :]
        
        #~ ax.plot(wf.T.flatten(), color='k')
        #~ ax.plot(wf0.T.flatten(), color='m')
        
        #~ plot_chan = channels.tolist().index(chan_ind)
        #~ plot_chan = chan_ind
        #~ ax.axvline(plot_chan * self.peak_width - self.n_left)
        
        #~ plt.show()

        
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        return cluster_idx


    def accept_tempate(self, left_ind, cluster_idx, jitter):
        # criteria mono channel = old implementation
        #~ keep_template = np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2)
        
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
        crietria_weighted = (wf_nrj>residual_nrj).astype('float') * weight
        accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        #~ accept_template = np.sum(crietria_weighted) >= 0.7 * np.sum(weight)

        #~ if True:
            
            #~ max_chan_ind = self.catalogue['clusters'][cluster_idx]['max_on_channel']
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




