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
    def change_params(self, 
                argmin_method='numba',
                adjacency_radius_um=100,
                **kargs):
        PeelerEngineGeneric.change_params(self, **kargs)
        self.argmin_method = argmin_method
        self.adjacency_radius_um = adjacency_radius_um # for waveform distance
        

     #~ # Some check
        #~ if self.use_sparse_template:
            #~ assert self.argmin_method != 'numpy', 'numpy methdo do not do sparse template acceleration'
            #~ if self.argmin_method == 'opencl':
                #~ assert HAVE_PYOPENCL, 'OpenCL is not available'
            #~ elif self.argmin_method == 'pythran':
                #~ assert HAVE_PYTHRAN, 'Pythran is not available'
            #~ elif self.argmin_method == 'numba':
                #~ assert HAVE_NUMBA, 'Numba is not available'
        
        #~ self.strict_template = True
        #~ self.strict_template = False
        
        if self.use_sparse_template:
            assert self.argmin_method in ('numba', 'opencl')
        
        if self.argmin_method == 'numpy':
            assert not self.use_sparse_template

    def initialize(self, **kargs):
        if self.argmin_method == 'opencl':
            OpenCL_Helper.initialize_opencl(self, cl_platform_index=self.cl_platform_index, cl_device_index=self.cl_device_index)
        
        PeelerEngineGeneric.initialize(self, **kargs)
        
        # create peak detector
        p = dict(self.catalogue['peak_detector_params'])

        #~ p.pop('engine')
        #~ p.pop('method')
        #~ self.peakdetector_method = 'geometrical'
        #~ if HAVE_PYOPENCL:
            #~ self.peakdetector_engine = 'opencl'
        #~ elif HAVE_NUMBA:
            #~ self.peakdetector_engine = 'numba'
        #~ else:
            #~ self.peakdetector_engine = 'numpy'
            #~ print('WARNING peak detetcor will slow : install opencl')

        self.peakdetector_engine = p.pop('engine')
        self.peakdetector_method = p.pop('method')
        
        PeakDetector_class = get_peak_detector_class(self.peakdetector_method, self.peakdetector_engine)
        
        chunksize = self.fifo_size-2*self.n_span # not the real chunksize here
        self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        chunksize, self.internal_dtype, self.geometry)
        self.peakdetector.change_params(**p)
        
        # some attrs
        self.shifts = np.arange(-self.maximum_jitter_shift, self.maximum_jitter_shift+1)
        self.nb_shift = self.shifts.size
        
        self.channel_distances = sklearn.metrics.pairwise.euclidean_distances(self.geometry).astype('float32')
        self.channels_adjacency = {}
        for c in range(self.nb_channel):
            if self.use_sparse_template:
                nearest, = np.nonzero(self.channel_distances[c, :]<self.adjacency_radius_um)
                self.channels_adjacency[c] = nearest
            else:
                self.channels_adjacency[c] = np.arange(self.nb_channel, dtype='int64')
        
        
        if self.argmin_method == 'opencl'  and self.catalogue['centers0'].size>0:
            
            # make kernels
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
            self.kern_explore_templates = getattr(opencl_prg, 'explore_templates')
            self.kern_explore_shifts = getattr(opencl_prg, 'explore_shifts')
            
            # create CL buffers

            wf_shape = centers.shape[1:]
            one_waveform = np.zeros(wf_shape, dtype='float32')
            self.one_waveform_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=one_waveform)
            
            long_waveform = np.zeros((wf_shape[0]+self.shifts.size, wf_shape[1]) , dtype='float32')
            self.long_waveform_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=long_waveform)
            

            self.catalogue_center_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=centers)

            self.distance_templates = np.zeros((nb_cluster), dtype='float32')
            self.distance_templates_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.distance_templates)

            self.distance_shifts = np.zeros((self.shifts.size, ), dtype='float32')
            self.distance_shifts_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.distance_shifts)


            self.sparse_mask_level1_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask_level1.astype('u1'))
            self.sparse_mask_level2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask_level2.astype('u1'))
            self.sparse_mask_level3_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask_level3.astype('u1'))
            
            rms_waveform_channel = np.zeros(nb_channel, dtype='float32')
            self.rms_waveform_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=rms_waveform_channel)
            
            self.adjacency_radius_um_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=np.array([self.adjacency_radius_um], dtype='float32'))
            
            self.cl_global_size = (centers.shape[0], centers.shape[2])
            self.cl_local_size = (centers.shape[0], 1) # faster a GPU because of memory access

            self.cl_global_size2 = (len(self.shifts), centers.shape[2])
            self.cl_local_size2 = (len(self.shifts), 1) # faster a GPU because of memory access
            
            self.channel_distances_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.channel_distances)
            
            
            self.kern_explore_templates.set_args(
                        self.one_waveform_cl, self.catalogue_center_cl,
                        self.sparse_mask_level3_cl,
                        self.rms_waveform_channel_cl, self.distance_templates_cl,  self.channel_distances_cl, 
                        self.adjacency_radius_um_cl, np.int32(0))
            
            self.kern_explore_shifts.set_args(
                                        self.long_waveform_cl,
                                        self.catalogue_center_cl,
                                        self.sparse_mask_level2_cl, 
                                        self.distance_shifts_cl,
                                        np.int32(0))
            
            
        #~ self.mask_not_already_tested = np.ones((self.fifo_size - 2 * self.n_span,self.nb_channel),  dtype='bool')
        

    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        self.peakdetector.reset_fifo_index()
        #~ self.mask_not_already_tested[:] = 1


    def detect_local_peaks_before_peeling_loop(self):
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        if mask.ndim ==1:
            local_peaks_indexes,   = np.nonzero(mask)
            chan_indexes = np.zeros(local_peaks_indexes.size, dtype='int64')
        else:
            local_peaks_indexes, chan_indexes  = np.nonzero(mask)
        local_peaks_indexes += self.n_span
        amplitudes = np.abs(self.fifo_residuals[local_peaks_indexes, chan_indexes])
        order = np.argsort(amplitudes)[::-1]
        self.pending_peaks = list(zip(local_peaks_indexes[order], chan_indexes[order]))
        self.already_tested = []
    
    def select_next_peak(self):
        #~ print(len(self.pending_peaks))
        if len(self.pending_peaks)>0:
            peak_ind, chan_ind = self.pending_peaks[0]
            self.pending_peaks = self.pending_peaks[1:]
            return peak_ind, chan_ind
        else:
            return LABEL_NO_MORE_PEAK, None

    #~ def on_accepted_spike(self, spike):
    def on_accepted_spike(self, peak_ind, cluster_idx, jitter):
        # remove spike prediction from fifo residuals
        #~ left_ind = spike.index + self.n_left
        #~ cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
        #~ peak_index = spike.index
        #~ pos, pred = make_prediction_one_spike(spike.index, cluster_idx, spike.jitter, self.fifo_residuals.dtype, self.catalogue)
        pos, pred = make_prediction_one_spike(peak_ind, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
        
        # this prevent search peaks in the zone until next "reset_to_not_tested"
        #~ self.set_already_tested_spike_zone(spike.index, cluster_idx)
        self.set_already_tested_spike_zone(peak_ind, cluster_idx)

    def set_already_tested_spike_zone(self, peak_ind, cluster_idx):
        # TODO test with sparse_mask_level3s!!!!!
        mask = self.sparse_mask_level1[cluster_idx, :]
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
    
    def set_already_tested(self, peak_ind, peak_chan):
        #~ self.mask_not_already_tested[peak_ind - self.n_span, peak_chan] = False
        #~ self.pending_peaks = [p for p in self.pending_peaks if (p[0]!=peak_ind) and (peak_chan!=p[1])]
        self.already_tested.append((peak_ind, peak_chan))

    def reset_to_not_tested(self, good_spikes):
        #~ print('reset_to_not_tested', len(good_spikes))
        #~ self.already_tested = []
        #~ print('self.already_tested', len(self.already_tested))
        for spike in good_spikes:
            # each good spike can remove from
            cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
            mask = self.sparse_mask_level1[cluster_idx, :]
            self.already_tested = [ p for p in self.already_tested if not((np.abs(p[0]-spike.index)<self.peak_width)  and mask[p[1]] ) ]
        #~ print('self.already_tested reduced', len(self.already_tested))
        # 
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)

        if mask.ndim ==1:
            local_peaks_indexes,   = np.nonzero(mask)
            chan_indexes = np.zeros(local_peaks_indexes.size, dtype='int64')
        else:
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

        #~ print('self.pending_peaks', len(self.pending_peaks))

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
        
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        
        if not np.any(self.sparse_mask_level3[:, chan_ind]):
            # no template near that channel
            cluster_idx = -1
            shift = None
            return cluster_idx, shift, None
        
        if self.argmin_method == 'opencl':
            #TODO remove this rms_waveform_channel no more usefull
            
            pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)

            self.distance_templates[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.distance_templates_cl, self.distance_templates)
            self.distance_shifts[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.distance_shifts_cl, self.distance_shifts)
            
            #~ rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
            #~ pyopencl.enqueue_copy(self.queue,  self.rms_waveform_channel_cl, rms_waveform_channel)
            
            
            #~ event = self.kern_explore_templates(self.queue,  self.cl_global_size, self.cl_local_size,
                        #~ self.one_waveform_cl, self.catalogue_center_cl,
                        #~ self.sparse_mask_level3_cl,
                        #~ self.rms_waveform_channel_cl, self.distance_templates_cl,  self.channel_distances_cl, 
                        #~ self.adjacency_radius_um_cl, np.int32(chan_ind))
            self.kern_explore_templates.set_arg(7, np.int32(chan_ind))
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_explore_templates, self.cl_global_size, self.cl_local_size,)
            pyopencl.enqueue_copy(self.queue,  self.distance_templates, self.distance_templates_cl)
            
            cluster_idx = np.argmin(self.distance_templates)
            shift = None
            
            # TODO avoid double enqueue
            long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
            pyopencl.enqueue_copy(self.queue,  self.long_waveform_cl, long_waveform)
            #~ event = self.kern_explore_shifts(
                                        #~ self.queue,  self.cl_global_size2, self.cl_local_size2,
                                        #~ self.long_waveform_cl,
                                        #~ self.catalogue_center_cl,
                                        #~ self.sparse_mask_level2_cl, 
                                        #~ self.distance_shifts_cl,
                                        #~ np.int32(cluster_idx))
            self.kern_explore_shifts.set_arg(4, np.int32(cluster_idx))
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_explore_shifts, self.cl_global_size2, self.cl_local_size2,)
            pyopencl.enqueue_copy(self.queue,  self.distance_shifts, self.distance_shifts_cl)
            #~ shift = self.shifts[np.argmin(self.distance_shifts)]

            ind_min = np.argmin(self.distance_shifts)
            shift = self.shifts[ind_min]
            distance = self.distance_shifts[ind_min]
            

            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.shifts, self.distance_shifts, marker='o')
            #~ ax.set_title(f'{left_ind-self.n_left} {chan_ind} {shift}')
            #~ plt.show()            
            
            
        
        #~ elif self.argmin_method == 'pythran':
            #~ s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                #~ self.catalogue['centers0'],  self.sparse_mask_level1)
            #~ cluster_idx = np.argmin(s)
            #~ shift = None
        
        elif self.argmin_method == 'numba':
            #TODO remove this rms_waveform_channel no more usefull
            rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
            
            #~ possibles_cluster_idx, = np.nonzero(self.sparse_mask_level1[:, chan_ind])
            possibles_cluster_idx, = np.nonzero(self.sparse_mask_level3[:, chan_ind])
            
            if possibles_cluster_idx.size ==0:
                cluster_idx = -1
                shift = None
            else:
                channel_adjacency = self.channels_adjacency[chan_ind]
                s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  
                                                        possibles_cluster_idx, rms_waveform_channel,channel_adjacency)
                cluster_idx = possibles_cluster_idx[np.argmin(s)]
                shift = None
                # explore shift
                long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
                all_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][cluster_idx, : , :],  self.sparse_mask_level2[cluster_idx, :], self.maximum_jitter_shift)
                ind_min = np.argmin(all_dist)
                shift = self.shifts[ind_min]
                distance = all_dist[ind_min]
                
                
                # experimental explore all shift for all templates!!!!!
                #~ shifts = list(range(-self.maximum_jitter_shift, self.maximum_jitter_shift+1))
                #~ channel_adjacency = self.channels_adjacency[chan_ind]
                #~ all_s = []
                #~ for shift in shifts:
                    #~ waveform = self.fifo_residuals[left_ind+shift:left_ind+self.peak_width+shift,:]
                    #~ s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  
                                                        #~ possibles_cluster_idx, rms_waveform_channel,channel_adjacency)                    
                    #~ all_s.append(s)
                #~ all_s = np.array(all_s)
                #~ shift_ind, ind_clus = np.unravel_index(np.argmin(all_s, axis=None), all_s.shape)
                #~ cluster_idx = possibles_cluster_idx[ind_clus]
                #~ shift = shifts[shift_ind]
                #~ distance = all_s[shift_ind][ind_clus]
                
                
                
            
                #~ print('      shift', shift)
            
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.shifts, all_dist, marker='o')
            #~ ax.set_title(f'{left_ind-self.n_left} {chan_ind} {shift}')
            #~ plt.show()            
            
            #~ shifts = list(range(-self.maximum_jitter_shift, self.maximum_jitter_shift+1))
            #~ all_s = []
            #~ for shift in shifts:
                #~ waveform = self.fifo_residuals[left_ind+shift:left_ind+self.peak_width+shift,:]
                #~ s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  self.sparse_mask_level1, possibles_cluster_idx, self.channels_adjacency[chan_ind])
                #~ all_s.append(s)
            #~ all_s = np.array(all_s)
            #~ shift_ind, cluster_idx = np.unravel_index(np.argmin(all_s, axis=None), all_s.shape)
            #~ cluster_idx = possibles_cluster_idx[cluster_idx]
            #~ shift = shifts[shift_ind]
            
            #~ if self._plot_debug:
                #~ fig, ax = plt.subplots()
                #~ ax.plot(self.shifts, all_dist, marker='o')
                #~ ax.set_title(f'{left_ind-self.n_left} {chan_ind} {shift}')
        
        elif self.argmin_method == 'numpy':
            assert not self.use_sparse_template
            # replace by this (indentique but faster, a but)
            d = self.catalogue['centers0']-waveform[None, :, :]
            d *= d
            #s = d.sum(axis=1).sum(axis=1)  # intuitive
            #s = d.reshape(d.shape[0], -1).sum(axis=1) # a bit faster
            distance_templates = np.einsum('ijk->i', d) # a bit faster
            cluster_idx = np.argmin(distance_templates)
            # TODO implement shift when numpy
            shift = 0
            distance = distance_templates[cluster_idx]
            
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
        return cluster_idx, shift, distance
    
    #~ def estimate_jitter(self, left_ind, cluster_idx):
        #~ return 0.

    def accept_tempate(self, left_ind, cluster_idx, jitter, distance):
        if jitter is None:
            # this must have a jitter
            jitter = 0
        
        if np.abs(jitter) > (self.maximum_jitter_shift - 0.5):
            return False
        
        #~ if self.strict_template:
            #~ return slef.accept_template_sctrict(left_ind, cluster_idx, jitter)
        
        #~ print('distance', distance)
        
        if distance < self.distance_limit[cluster_idx]:
        #~ if False:
            accept_template = True
            debug_d = True
        else:
            # criteria multi channel
            #~ mask = self.sparse_mask_level1[cluster_idx]
            mask = self.sparse_mask_level2[cluster_idx]
            #~ full_wf0 = self.catalogue['centers0'][cluster_idx,: , :][:, mask]
            #~ full_wf1 = self.catalogue['centers1'][cluster_idx,: , :][:, mask]
            #~ full_wf2 = self.catalogue['centers2'][cluster_idx,: , :][:, mask]
            
            # waveform L2 on mask
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            wf = full_waveform[:, mask]
            #~ wf_nrj = np.sum(full_wf**2, axis=0)
            
            # prediction L2 on mask
            #~ pred_wf = (full_wf0+jitter*full_wf1+jitter**2/2*full_wf2)
            
            # prediction with interpolation
            _, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
            pred_wf = pred_wf[:, mask]
        
        
            dist = (pred_wf - wf) ** 2
        
            debug_d = False
            residual_nrj_by_chan = np.sum(dist, axis=0)
            
            wf_nrj = np.sum(wf**2, axis=0)
            
            # criteria per channel
            weight = self.weight_per_template_dict[cluster_idx]
            crietria_weighted = (wf_nrj>residual_nrj_by_chan).astype('float') * weight
            #~ accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
            accept_template = np.sum(crietria_weighted) >= 0.7 * np.sum(weight)
        
        
        label = self.catalogue['cluster_labels'][cluster_idx]
        #~ if not accept_template and label in []:
        #~ if not accept_template:
        #~ if True:
        if False:
        #~ if label == 6 and not accept_template:
        #~ nears = np.array([ 5813767,  5813767, 11200038, 11322540, 14989650, 14989673, 14989692, 14989710, 15119220, 15830377, 16138346, 16216666, 17078883])
        #~ print(np.abs((left_ind - self.n_left) - nears))
        #~ print(np.abs((left_ind - self.n_left) - nears) < 2)
        #~ if label == 5 and np.any(np.abs((left_ind - self.n_left) - nears) < 50):
            
            if debug_d:
                mask = self.sparse_mask_level2[cluster_idx]
                full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
                wf = full_waveform[:, mask]
                _, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
                pred_wf = pred_wf[:, mask]
    
            
            if accept_template:
                if debug_d:
                    color = 'g'
                else:
                    color = 'c'
            else:
                color = 'r'
            fig, ax = plt.subplots()
            ax.plot(wf.T.flatten(), color='k')
            ax.plot(pred_wf.T.flatten(), color=color)
            
            ax.plot( wf.T.flatten() - pred_wf.T.flatten(), color=color, ls='--')
            
            print()
            print(distance, self.distance_limit[cluster_idx])
            if distance >= self.distance_limit[cluster_idx]:
                print(crietria_weighted, weight)
                print(np.sum(crietria_weighted),  np.sum(weight))
            
            #~ ax.plot(full_wf0.T.flatten(), color='y')
            #~ ax.plot( full_wf.T.flatten() - full_wf0.T.flatten(), color='y')
            
            #~ ax.set_title('not accepted')
            plt.show()
        
        return accept_template
    
    
    def accept_template_sctrict(self, left_ind, cluster_idx, jitter):
        # experimental
        mask = self.sparse_mask_level2[cluster_idx]
        
        # waveform
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        full_wf = waveform[:, mask]
        
        # prediction with interpolation
        _, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        pred_wf= pred_wf[:, mask]

        if np.all(np.abs(pred_wf - full_wf)<self.threshold):
            accept_template = True
            #~ debug_d = True
        else:
            accept_template = False        
        return accept_template
    
    def _plot_after_inner_peeling_loop(self):
        pass

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
        
        #~ 
        
    
    def _plot_label_unclassified(self, left_ind, peak_chan, cluster_idx, jitter):
        return
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
        
        
        for c in range(self.nb_channel):
            plot_sigs[:, c] += c*30
        ax.plot(plot_sigs, color='k')
        
        ax.plot(self._plot_sigs_before, color='b')
        
        ax.axvline(self.fifo_size - self.n_right, color='r')
        ax.axvline(-self.n_left, color='r')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        peak_inds, chan_inds= np.nonzero(mask)
        peak_inds += self.n_span
        ax.scatter(peak_inds, plot_sigs[peak_inds, chan_inds], color='r')
        
        
        
        good_spikes = np.array(good_spikes, dtype=_dtype_spike)
        pred = make_prediction_signals(good_spikes, self.internal_dtype, plot_sigs.shape, self.catalogue, safe=True)
        plot_pred = pred.copy()
        for c in range(self.nb_channel):
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


__kernel void explore_templates(__global  float *one_waveform,
                                        __global  float *catalogue_center,
                                        __global  uchar  *sparse_mask_level3,
                                        __global  float *rms_waveform_channel,
                                        __global  float *distance_templates,
                                        __global  float *channel_distances,
                                        __global float * adjacency_radius_um,
                                        int chan_ind
                                        ){
    
    int cluster_idx = get_global_id(0);
    int c = get_global_id(1);
    
    
    // the chan_ind do not overlap spatialy this cluster
    if (sparse_mask_level3[nb_channel*cluster_idx+chan_ind] == 0){
        if (c==0){
            distance_templates[cluster_idx] = FLT_MAX;
        }
    }
    else {
        // candidate initialize sum by cluster
        if (c==0){
            distance_templates[cluster_idx] = 0.0f;
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (sparse_mask_level3[nb_channel*cluster_idx+chan_ind] == 0){
        return;
    }
    
    
    float sum = 0;
    float d;
    
    if (channel_distances[c * nb_channel + chan_ind] < *adjacency_radius_um){
//        if (sparse_mask[nb_channel*cluster_idx+c]>0){
//            for (int s=0; s<peak_width; ++s){
//                d = one_waveform[nb_channel*s+c] - catalogue_center[wf_size*cluster_idx+nb_channel*s+c];
//                sum += d*d;
//            }
//        }
//        else{
//            sum = rms_waveform_channel[c];
//        }
        for (int s=0; s<peak_width; ++s){
            d = one_waveform[nb_channel*s+c] - catalogue_center[wf_size*cluster_idx+nb_channel*s+c];
            sum += d*d;
        }
        atomic_add_float(&distance_templates[cluster_idx], sum);
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

    


