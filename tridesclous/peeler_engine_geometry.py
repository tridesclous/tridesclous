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

from .peakdetector import get_peak_detector_class

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_loop_sparse_dist_with_geometry, numba_explore_shifts
except ImportError:
    HAVE_NUMBA = False




import matplotlib.pyplot as plt


class PeelerEngineGeometrical(PeelerEngineGeneric):
    def change_params(self, adjacency_radius_um=200, **kargs):
        PeelerEngineGeneric.change_params(self, **kargs)
        
        assert self.use_sparse_template
        
        self.adjacency_radius_um = adjacency_radius_um
        self.shifts = np.arange(-self.maximum_jitter_shift, self.maximum_jitter_shift+1)

        

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
                                                    'wf_size':peak_width*nb_channel,'nb_cluster' : nb_cluster, 
                                                    'maximum_jitter_shift': self.maximum_jitter_shift}
            #~ print(kernel)
            prg = pyopencl.Program(self.ctx, kernel)
            opencl_prg = prg.build(options='-cl-mad-enable')
            self.kern_waveform_distance = getattr(opencl_prg, 'waveform_distance')
            self.kern_explore_shifts = getattr(opencl_prg, 'explore_shifts')
            
            

            wf_shape = centers.shape[1:]
            one_waveform = np.zeros(wf_shape, dtype='float32')
            self.one_waveform_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=one_waveform)
            
            long_waveform = np.zeros((wf_shape[0]+self.shifts.size, wf_shape[1]) , dtype='float32')
            self.long_waveform_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=long_waveform)
            

            self.catalogue_center_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=centers)

            self.waveform_distance = np.zeros((nb_cluster), dtype='float32')
            self.waveform_distance_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.waveform_distance)

            #~ mask[:] = 0
            self.sparse_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask.astype('u1'))

            rms_waveform_channel = np.zeros(nb_channel, dtype='float32')
            self.rms_waveform_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=rms_waveform_channel)
            
            
            
            self.adjacency_radius_um_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=np.array([self.adjacency_radius_um], dtype='float32'))
            
            
            self.cl_global_size = (centers.shape[0], centers.shape[2])
            #~ self.cl_local_size = None
            self.cl_local_size = (centers.shape[0], 1) # faster a GPU because of memory access
            #~ self.cl_local_size = (1, centers.shape[2])

            self.cl_global_size2 = (len(self.shifts), centers.shape[2])
            #~ self.cl_local_size = None
            self.cl_local_size2 = (len(self.shifts), 1) # faster a GPU because of memory access
            #~ self.cl_local_size = (1, centers.shape[2])
            
            # to check if distance is valid is a coeff (because maxfloat on opencl)
            self.max_float32 = np.finfo('float32').max * 0.8



    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        
        p = dict(self.catalogue['peak_detector_params'])
        p.pop('engine')
        p.pop('method')
        
        self.peakdetector_method = 'geometrical'
        
        if HAVE_PYOPENCL:
            self.peakdetector_engine = 'opencl'
        elif HAVE_NUMBA:
            self.peakdetector_engine = 'numba'
        else:
            self.peakdetector_engine = 'numpy'
            print('WARNING peak detetcor will slow : install opencl')
        
        PeakDetector_class = get_peak_detector_class(self.peakdetector_method, self.peakdetector_engine)
        
        chunksize = self.fifo_size-2*self.n_span # not the real chunksize here
        self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        chunksize, self.internal_dtype, self.geometry)
        self.peakdetector.change_params(**p)
        
        # DEBUG
        #~ p['nb_neighbour'] = 4

        self.channel_distances = sklearn.metrics.pairwise.euclidean_distances(self.geometry).astype('float32')
        self.channels_adjacency = {}
        for c in range(self.nb_channel):
            nearest, = np.nonzero(self.channel_distances[c, :]<self.adjacency_radius_um)
            self.channels_adjacency[c] = nearest
        
        if self.argmin_method == 'opencl'  and self.catalogue['centers0'].size>0:
            self.channel_distances_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.channel_distances)
            
            
            self.all_distance = np.zeros((self.shifts.size, ), dtype='float32')
            self.all_distance_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.all_distance)
            
            
            
        
        #~ self.peakdetector = PeakDetectorGeometricalNumpy(self.sample_rate, self.nb_channel,
                        #~ self.fifo_size-2*self.n_span, self.internal_dtype, self.geometry)
        
        #~ # TODO size fido
        #~ self.peakdetector = PeakDetectorGeometricalOpenCL(self.sample_rate, self.nb_channel,
                                                        #~ self.fifo_size-2*self.n_span, self.internal_dtype, self.geometry)
        #~ self.peakdetector.change_params(**p)

        
        self.mask_not_already_tested = np.ones((self.fifo_size - 2 * self.n_span,self.nb_channel),  dtype='bool')


    def OLD_detect_local_peaks_before_peeling_loop(self):
    #~ def detect_local_peaks_before_peeling_loop(self):
        self.mask_not_already_tested[:] = True
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        
    #~ def NEW_detect_local_peaks_before_peeling_loop(self):
    def detect_local_peaks_before_peeling_loop(self):
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        local_peaks_indexes, chan_indexes  = np.nonzero(mask)
        local_peaks_indexes += self.n_span
        amplitudes = np.abs(self.fifo_residuals[local_peaks_indexes, chan_indexes])
        order = np.argsort(amplitudes)[::-1]
        self.pending_peaks = list(zip(local_peaks_indexes[order], chan_indexes[order]))
        self.already_tested = []
    
    def OLD_select_next_peak(self):
    #~ def select_next_peak(self):
        local_peaks_indexes, chan_indexes  = np.nonzero(self.local_peaks_mask & self.mask_not_already_tested)
        
        if local_peaks_indexes.size>0:
            local_peaks_indexes += self.n_span
            amplitudes = np.abs(self.fifo_residuals[local_peaks_indexes, chan_indexes])
            ind = np.argmax(amplitudes)
            peak_ind = local_peaks_indexes[ind]
            chan_ind = chan_indexes[ind]
            return peak_ind, chan_ind
        else:
            return LABEL_NO_MORE_PEAK, None

    #~ def NEW_select_next_peak(self):
    def select_next_peak(self):
        #~ print(len(self.pending_peaks))
        if len(self.pending_peaks)>0:
            peak_ind, chan_ind = self.pending_peaks[0]
            self.pending_peaks = self.pending_peaks[1:]
            return peak_ind, chan_ind
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

    def OLD_set_already_tested_spike_zone(self, peak_ind, cluster_idx):
    #~ def set_already_tested_spike_zone(self, peak_ind, cluster_idx):
        mask = self.sparse_mask[cluster_idx, :]
        for c in np.nonzero(mask)[0]:
            self.mask_not_already_tested[peak_ind + self.n_left - self.n_span:peak_ind + self.n_right- self.n_span, c] = False

    #~ def NEW_set_already_tested_spike_zone(self, peak_ind, cluster_idx):
    def set_already_tested_spike_zone(self, peak_ind, cluster_idx):
        mask = self.sparse_mask[cluster_idx, :]
        #~ keep = [not((ind == peak_ind) and (mask[chan_ind])) for ind, chan_ind in self.pending_peaks]
        #~ keep = [(ind != peak_ind) and not(mask[chan_ind]) for ind, chan_ind in self.pending_peaks]
        
        pending_peaks_ = []
        #~ for p, ok in zip(self.pending_peaks, keep):
        for ind, chan_ind in self.pending_peaks:
            #~ ok = (ind != peak_ind) and not(mask[chan_ind])
            in_zone = mask[chan_ind] and (ind+self.n_left<peak_ind<ind+self.n_right)
            if in_zone:
                self.already_tested.append((ind, chan_ind))
            else:
                pending_peaks_.append((ind, chan_ind))
        self.pending_peaks = pending_peaks_
    
    def OLD_set_already_tested(self, peak_ind, peak_chan):
    #~ def set_already_tested(self, peak_ind, peak_chan):
        self.mask_not_already_tested[peak_ind - self.n_span, peak_chan] = False
    
    #~ def NEW_set_already_tested(self, peak_ind, peak_chan):
    def set_already_tested(self, peak_ind, peak_chan):
        #~ self.mask_not_already_tested[peak_ind - self.n_span, peak_chan] = False
        #~ self.pending_peaks = [p for p in self.pending_peaks if (p[0]!=peak_ind) and (peak_chan!=p[1])]
        self.already_tested.append((peak_ind, peak_chan))

    def OLDreset_to_not_tested(self, good_spikes):
    #~ def reset_to_not_tested(self, good_spikes):
        #TODO : more efficient only local !!!!
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)

        for spike in good_spikes:
            peak_ind = spike.index
            cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
            mask = self.sparse_mask[cluster_idx, :]
            for c in np.nonzero(mask)[0]:
                self.mask_not_already_tested[peak_ind + self.n_left - self.n_span:peak_ind + self.n_right- self.n_span, c] = True

    
    #~ def NEW_reset_to_not_tested(self, good_spikes):
    def reset_to_not_tested(self, good_spikes):
        #~ self.already_tested = []
        for spike in good_spikes:
            cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
            mask = self.sparse_mask[cluster_idx, :]
            self.already_tested = [ p for p in self.already_tested if not(((p[0]-spike.index)<self.peak_width)  and mask[p[1]] ) ]
        
        # BAD IDEA because all peak are tested again and again after loop
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        local_peaks_indexes, chan_indexes  = np.nonzero(mask)
        local_peaks_indexes += self.n_span
        amplitudes = np.abs(self.fifo_residuals[local_peaks_indexes, chan_indexes])
        order = np.argsort(amplitudes)[::-1]
        possible_pending_peaks = list(zip(local_peaks_indexes[order], chan_indexes[order]))
       
        self.pending_peaks = []
        for peak in possible_pending_peaks:
            #~ ok = all((peak[0] != p[0]) and (peak[1] != p[1]) for p in self.already_tested)
            if peak not in self.already_tested:
                self.pending_peaks.append(peak)

    def get_no_label_peaks(self):
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        nolabel_indexes, chan_indexes = np.nonzero(mask)
        #~ nolabel_indexes, chan_indexes = np.nonzero(~self.mask_not_already_tested)
        
        nolabel_indexes += self.n_span
        nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
        bad_spikes['index'] = nolabel_indexes
        bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        return bad_spikes

    def get_best_template(self, left_ind, chan_ind):
        assert self.argmin_method in ('numba', 'opencl')
        
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]

        
        if self.argmin_method == 'opencl':
            rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
            
            pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)
            pyopencl.enqueue_copy(self.queue,  self.rms_waveform_channel_cl, rms_waveform_channel)
            event = self.kern_waveform_distance(self.queue,  self.cl_global_size, self.cl_local_size,
                        self.one_waveform_cl, self.catalogue_center_cl, self.sparse_mask_cl, 
                        self.rms_waveform_channel_cl, self.waveform_distance_cl,  self.channel_distances_cl, 
                        self.adjacency_radius_um_cl, np.int32(chan_ind))
            pyopencl.enqueue_copy(self.queue,  self.waveform_distance, self.waveform_distance_cl)
            
            cluster_idx = np.argmin(self.waveform_distance)
            shift = None
            
            # TODO avoid double enqueue
            long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
            pyopencl.enqueue_copy(self.queue,  self.long_waveform_cl, long_waveform)
            event = self.kern_explore_shifts(
                                        self.queue,  self.cl_global_size2, self.cl_local_size2,
                                        self.long_waveform_cl,
                                        self.catalogue_center_cl,
                                        self.sparse_mask_cl, 
                                        self.all_distance_cl,
                                        np.int32(cluster_idx))
            pyopencl.enqueue_copy(self.queue,  self.all_distance, self.all_distance_cl)
            shift = self.shifts[np.argmin(self.all_distance)]

            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.shifts, self.all_distance, marker='o')
            #~ ax.set_title(f'{left_ind-self.n_left} {chan_ind} {shift}')
            #~ plt.show()            
            
            
        
        #~ elif self.argmin_method == 'pythran':
            #~ s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                #~ self.catalogue['centers0'],  self.sparse_mask)
            #~ cluster_idx = np.argmin(s)
            #~ shift = None
        
        elif self.argmin_method == 'numba':
            possibles_cluster_idx, = np.nonzero(self.sparse_mask[:, chan_ind])
            
            if possibles_cluster_idx.size ==0:
                cluster_idx = -1
                shift = None
            else:
            
                s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  self.sparse_mask, possibles_cluster_idx, self.channels_adjacency[chan_ind])
                cluster_idx = possibles_cluster_idx[np.argmin(s)]
                if s[cluster_idx] > self.max_float32:
                    # no match
                    cluster_idx = -1
                    shift = None
                else:
                    shift = None
                    # explore shift
                    long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
                    all_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][cluster_idx, : , :],  self.sparse_mask[cluster_idx, :], self.maximum_jitter_shift)
                    shift = self.shifts[np.argmin(all_dist)]
                #~ print('      shift', shift)
            
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.shifts, all_dist, marker='o')
            #~ ax.set_title(f'{left_ind-self.n_left} {chan_ind} {shift}')
            #~ plt.show()            
            
            #~ shifts = list(range(-self.maximum_jitter_shift, self.maximum_jitter_shift+1))
            #~ all_s = []
            #~ for shift in shifts:
                #~ waveform = self.fifo_residuals[left_ind+shift:left_ind+self.peak_width+shift,:]
                #~ s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  self.sparse_mask, possibles_cluster_idx, self.channels_adjacency[chan_ind])
                #~ all_s.append(s)
            #~ all_s = np.array(all_s)
            #~ shift_ind, cluster_idx = np.unravel_index(np.argmin(all_s, axis=None), all_s.shape)
            #~ cluster_idx = possibles_cluster_idx[cluster_idx]
            #~ shift = shifts[shift_ind]
            
            if self._plot_debug:
                fig, ax = plt.subplots()
                ax.plot(self.shifts, all_dist, marker='o')
                ax.set_title(f'{left_ind-self.n_left} {chan_ind} {shift}')
        
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
        
        
        #~ print('cluster_idx', cluster_idx)
        #~ print('chan_ind', chan_ind)
        
        
        #~ fig, ax = plt.subplots()
        #~ chan_order = np.argsort(self.channel_distances[0, :])
        #~ channels = self.channels_adjacency[chan_ind]
        #~ channels = chan_order
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
        return cluster_idx, shift
    
    #~ def estimate_jitter(self, left_ind, cluster_idx):
        #~ return 0.

    def accept_tempate(self, left_ind, cluster_idx, jitter):
        #~ self._debug_nb_accept_tempate += 1
        
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
        crietria_weighted = (wf_nrj>residual_nrj).astype('float') * weight
        #~ accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        accept_template = np.sum(crietria_weighted) >= 0.7 * np.sum(weight)

        #~ if True:
            
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
        #~ chan_order = np.argsort(self.channel_distances[0, :])
        
        for c in range(self.nb_channel):
        #~ for c in chan_order:
            plot_sigs[:, c] += c*30
        
        ax.plot(plot_sigs, color='k')

        ax.axvline(self.fifo_size - self.n_right, color='r')
        ax.axvline(-self.n_left, color='r')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        peak_inds, chan_inds= np.nonzero(mask)
        peak_inds += self.n_span
        
        ax.scatter(peak_inds, plot_sigs[peak_inds, chan_inds], color='r')
        
    
    def _plot_label_unclassified(self, left_ind, peak_chan, cluster_idx, jitter):
        print('LABEL UNCLASSIFIED', left_ind, cluster_idx)
        fig, ax = plt.subplots()
        
        wf = self.fifo_residuals[left_ind:left_ind+self.peak_width, :]
        wf0 = self.catalogue['centers0'][cluster_idx, :, :]
        
        ax.plot(wf.T.flatten(), color='b')
        ax.plot(wf0.T.flatten(), color='g')
        
        ax.set_title(f'label_unclassified {left_ind-self.n_left} {cluster_idx}')

    def _plot_after_peeling_loop(self, good_spikes):
        fig, ax = plt.subplots()
        plot_sigs = self.fifo_residuals.copy()
        
        
        #~ chan_order = np.argsort(self.channel_distances[0, :])
        
        for c in range(self.nb_channel):
        #~ for c in chan_order:
            plot_sigs[:, c] += c*30
        ax.plot(plot_sigs, color='k')
        
        ax.plot(self._plot_sigs_before, color='b')
        
        ax.axvline(self.fifo_size - self.n_right, color='r')
        ax.axvline(-self.n_left, color='r')

        #~ for ind in np.nonzero(~self.mask_not_already_tested)[0] + self.n_span:
            #~ ax.axvline(ind, ls='-', color='g')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        peak_inds, chan_inds= np.nonzero(mask)
        peak_inds += self.n_span
        ax.scatter(peak_inds, plot_sigs[peak_inds, chan_inds], color='r')
        
        
        
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
#define maximum_jitter_shift %(maximum_jitter_shift)d

    
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
                                        __global  float *waveform_distance,
                                        __global  float *channel_distances,
                                        __global float * adjacency_radius_um,
                                        int chan_ind
                                        ){
    
    int cluster_idx = get_global_id(0);
    int c = get_global_id(1);
    
    
    // the chan_ind do not overlap spatialy this cluster
    if (sparse_mask[nb_channel*cluster_idx+chan_ind] == 0){
        if (c==0){
            waveform_distance[cluster_idx] = FLT_MAX;
        }
    }
    else {
        // candidate initialize sum by cluster
        if (c==0){
            waveform_distance[cluster_idx] = 0;
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (sparse_mask[nb_channel*cluster_idx+chan_ind] == 0){
        return;
    }
    
    
    float sum = 0;
    float d;
    
    if (channel_distances[c * nb_channel + chan_ind] < *adjacency_radius_um){
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
}


__kernel void explore_shifts(__global  float *long_waveform,
                                        __global  float *catalogue_center,
                                        __global  uchar  *sparse_mask,
                                        __global  float *all_distance,
                                        int cluster_idx){
    
    int shift = get_global_id(0);
    int c = get_global_id(1);

    if (c==0){
        all_distance[shift] = 0;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    float sum = 0;
    float d;

    if (sparse_mask[nb_channel*cluster_idx+c]>0){
        for (int s=0; s<peak_width; ++s){
            d = long_waveform[nb_channel*(s+shift)+c] - catalogue_center[wf_size*cluster_idx+nb_channel*s+c];
            sum += d*d;
        }
        atomic_add_float(&all_distance[shift], sum);
    }
    
    

}


"""

    


