"""
Here implementation that tale in account the geometry
of the probe to speed up template matching.

"""

import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor
import itertools



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
    from .numba_tools import numba_loop_sparse_dist_with_geometry, numba_explore_shifts, numba_explore_best_template, numba_explore_best_shift
except ImportError:
    HAVE_NUMBA = False




import matplotlib.pyplot as plt


class PeelerEngineGeometrical(PeelerEngineGeneric):
    def change_params(self, 
                argmin_method='numba',
                #~ adjacency_radius_um=100,
                #~ adjacency_radius_um=200,
                adjacency_radius_um=400,
                
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
        
        # debug
        #~ self.already_tested =[]

        if mask.ndim ==1:
            local_peaks_indexes,   = np.nonzero(mask)
            chan_indexes = np.zeros(local_peaks_indexes.size, dtype='int64')
        else:
            local_peaks_indexes, chan_indexes  = np.nonzero(mask)

        local_peaks_indexes += self.n_span
        # TODO fix amplitudes when mask.ndim ==1
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
        
        
        
        #~ if not np.any(self.sparse_mask_level3[:, chan_ind]):
            #~ # no template near that channel
            #~ peak_val = self.fifo_residuals[left_ind-self.n_left,chan_ind]   # debug
            #~ if np.abs(peak_val) > 8:# debug
                #~ print('not np.any(self.sparse_mask_level3[:, chan_ind])', peak_val)# debug
                
            #~ cluster_idx = -1
            #~ shift = None
            #~ return cluster_idx, shift, None
        
        
        
        if self.argmin_method == 'opencl':
            #TODO remove this rms_waveform_channel no more usefull
            
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            
            pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, full_waveform)

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
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            
            centers0 = self.catalogue['centers0']
            inner_othogonal_projector = self.catalogue['inner_othogonal_projector']
            othogonal_bondaries = self.catalogue['othogonal_bondaries']
            low_boundary = othogonal_bondaries[:, 0]
            high_boundary = othogonal_bondaries[:, 1]
            
            
            n = centers0.shape[0]
            flat_waveform = full_waveform.flatten()
            flat_centers0 = centers0.reshape(n, -1)
            
            #~ scalar_products = np.zeros(n, dtype='float32')
            #~ for i in range(n):
                #~ sp = np.sum((flat_waveform - flat_centers0[i, :]) * inner_othogonal_projector[i, :])
                #~ scalar_products[i] = sp
            scalar_products = np.sum((flat_waveform[np.newaxis, :] - flat_centers0[:, :]) * inner_othogonal_projector[:, :], axis=1)
            #~ print(scalar_products)
            
            possible_idx, = np.nonzero((scalar_products < high_boundary) & (scalar_products > low_boundary))
            
            if len(possible_idx) == 1:
                #~ cluster_idx = possible_idx[0]
                extra_idx = None
                #~ candidates_idx = [cluster_idx] # for shift
                candidates_idx =possible_idx
            elif len(possible_idx) == 0:
                extra_idx, = np.nonzero((np.abs(scalar_products) < 0.5))
                print('extra_idx', extra_idx)
                if len(extra_idx) ==0:
                    #~ cluster_idx = None
                    candidates_idx = []
                    #~ shift = None
                else:
                    #~ cluster_idx = extra_idx[np.argmin(scalar_products[extra_idx])]
                    candidates_idx = extra_idx
            elif len(possible_idx) > 1 :
                #~ cluster_idx = np.argmin(scalar_products)
                extra_idx = None
                candidates_idx = possible_idx
            
            debug_plot_change = False
            if len(candidates_idx) > 0:
                candidates_idx = np.array(candidates_idx, dtype='int64')
                shift_scalar_product, shift_distance = numba_explore_best_shift(self.fifo_residuals, left_ind, self.catalogue['centers0'],
                                self.catalogue['inner_othogonal_projector'], candidates_idx, self.maximum_jitter_shift) 
                #~ i0, i1 = np.unravel_index(np.argmin(np.abs(shift_scalar_product), axis=None), shift_scalar_product.shape)
                i0, i1 = np.unravel_index(np.argmin(shift_distance, axis=None), shift_distance.shape)
                #~ best_idx = candidates_idx[i0]
                shift = self.shifts[i1]
                cluster_idx = candidates_idx[i0]
                
                
                #~ if best_idx != cluster_idx:
                    #~ print('*'*50)
                    #~ print('best_idx != cluster_idx', best_idx, cluster_idx)
                    #~ print('*'*50)
                    #~ cluster_idx = best_idx
                    #~ debug_plot_change = True
            else:
                cluster_idx = None
                shift = None
                #~ fig, ax = plt.subplots()
                #~ ax.plot(self.shifts, shift_scalar_product.T)
                #~ plt.show()
            
            
            #~ if cluster_idx in (3,6):
            if False:
            #~ if True:
            #~ if len(possible_idx) != 1:
            #~ if len(possible_idx) > 1:
            #~ len(possible_idx) > 1
            
            #~ if 7 in possible_idx or  cluster_idx == 7:
            #~ if cluster_idx not in possible_idx and len(possible_idx) > 0:
            #~ if debug_plot_change:
            
                print()
                print('best cluster_idx', cluster_idx)
                print('possible_idx', possible_idx)
                print('extra_idx', extra_idx)
                print(scalar_products[possible_idx])
                print(high_boundary[possible_idx])
                
                
                fig, ax = plt.subplots()
                shift2 = 0 if shift is None else shift
                full_waveform2 = self.fifo_residuals[left_ind+shift2:left_ind+shift2+self.peak_width,:]
                
                ax.plot(full_waveform2.T.flatten(), color='k')
                if shift !=0 and shift is not None:
                    ax.plot(full_waveform.T.flatten(), color='grey', ls='--')
                
                for idx in candidates_idx:
                    ax.plot(self.catalogue['centers0'][idx, :].T.flatten(), color='m')
                if cluster_idx is not None:
                    ax.plot(self.catalogue['centers0'][cluster_idx, :].T.flatten(), color='c', ls='--')
                ax.set_title(f'best {cluster_idx} shift {shift} possible_idx {possible_idx}')
                
                if shift is not None:
                    fig, ax = plt.subplots()
                    ax.plot(self.shifts, np.abs(shift_scalar_product).T)

                    fig, ax = plt.subplots()
                    ax.plot(self.shifts, np.abs(shift_distance).T)
                
                plt.show()
            
            shift = 0
            
            return cluster_idx, shift, None

        elif self.argmin_method == 'numba_not_so_old':
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            
            
            mean_centroid = self.catalogue['mean_centroid']
            projector = self.catalogue['projector']
            feat_centroids = self.catalogue['feat_centroids']
            
            feat_waveform = (full_waveform.flatten() - mean_centroid) @ projector
            
            feat_distances = np.sum((feat_centroids - feat_waveform[np.newaxis])**2, axis=1)
            #~ print(len(feat_distances), ':', feat_distances)
            cluster_idx = np.argmin(feat_distances)
            #~ print(feat_waveform.shape)
            
            
            possible_idx, = np.nonzero(feat_distances < self.catalogue['feat_distance_boundaries'])
            #~ print()
            #~ print('best cluster_idx', cluster_idx)
            #~ print('possible_idx', possible_idx)
            
            
            
            
            #~ if len(possible_idx) != 1:
            #~ if len(possible_idx) > 1:
            if False:
            #~ if cluster_idx not in possible_idx and len(possible_idx) > 0:
                print()
                print('best cluster_idx', cluster_idx)
                print('possible_idx', possible_idx)
                print(feat_distances[possible_idx])
                print(self.catalogue['feat_distance_boundaries'][possible_idx])
                
                
                
                fig, ax = plt.subplots()
                ax.plot(feat_centroids.T, alpha=0.5)
                ax.plot(feat_waveform, color='k')
                
                fig, ax = plt.subplots()
                ax.plot(full_waveform.T.flatten(), color='k')
                for idx in possible_idx:
                    ax.plot(self.catalogue['centers0'][idx, :].T.flatten(), color='m')
                ax.plot(self.catalogue['centers0'][cluster_idx, :].T.flatten(), color='c', ls='--')
                ax.set_title(f'best {cluster_idx} possible_idx {possible_idx}')
                
                plt.show()
            
            shift = 0
            
            return cluster_idx, shift, None
            
            #~ exit()
            
            
        
        elif self.argmin_method == 'numba_old2':
            
            # commmon mask
            intercept_channel = self.sparse_mask_level3[:, chan_ind]
            print('intercept_channel', np.sum(intercept_channel))
            common_mask = np.sum(self.sparse_mask_level3[intercept_channel, :], axis=0) > 0
            print('common_mask', np.sum(common_mask))
            
            
            scalar_products, weighted_distance = numba_explore_best_template(self.fifo_residuals, left_ind, chan_ind,
                                    self.catalogue['centers0'], self.catalogue['centers0_normed'], self.catalogue['template_weight'],
                                    common_mask,
                                    self.catalogue['sparse_mask_level1'], 
                                    self.catalogue['sparse_mask_level2'], 
                                    self.catalogue['sparse_mask_level3'], 
                                    self.catalogue['sparse_mask_level4'])
            
            
            
            if max(scalar_products)<0:
                cluster_idx = -1
                shift = None
                return cluster_idx, shift, None
            
            #~ possibles_idx1, = np.nonzero((scalar_products > 0) &
                            #~ (scalar_products > self.catalogue['sp_normed_limit'][:, 0]) &
                            #~ (scalar_products < self.catalogue['sp_normed_limit'][:, 1]))

            #~ factor = 0.3
            #~ possibles_idx1, = np.nonzero(scalar_products > (self.catalogue['sp_normed_limit'][:, 0] * 0.8))
            #~ possibles_idx1, = np.nonzero(scalar_products > (self.catalogue['sp_normed_limit'][:, 0] * 0.5))
            #~ possibles_idx1, = np.nonzero(scalar_products > (self.catalogue['sp_normed_limit'][:, 0] * 0.1))
            #~ possibles_idx1, = np.nonzero(scalar_products > 0)
            possibles_idx1 = np.array([], dtype='int64')

            #~ possibles_idx, = np.nonzero(scalar_products>0)

            
            #~ min_ = min(weighted_distance)
            #~ factor = 2.
            #~ possibles_idx2, = np.nonzero(weighted_distance<(min_ * factor))
            
            #~ factor = 2
            #~ factor = 3
            #~ factor = 8
            possibles_idx2, = np.nonzero(weighted_distance<(self.distance_limit))
            #~ print(weighted_distance)
            #~ print(self.distance_limit)
            #~ possibles_idx2 = np.array([], dtype='int64')
            
            #~ candidates_idx = np.union1d(possibles_idx1, possibles_idx2)
            candidates_idx = possibles_idx2
            
            
            print()
            #~ print(possibles_idx1, possibles_idx2)
            print(candidates_idx)
            if len(candidates_idx) == 0:
                print()
                
                peak_val = self.fifo_residuals[left_ind-self.n_left, chan_ind]   # debug
                print('No candidate', 'peak_val', peak_val, 'chan_ind', chan_ind)
                cluster_idx = -1
                shift = None
                
                #~ print('ici', self.catalogue['sparse_mask_level3'][:, chan_ind])
                #~ print(np.nonzero(self.catalogue['sparse_mask_level3'][:, chan_ind]))
                #~ print(scalar_products)
                #~ print(self.catalogue['sp_normed_limit'][:, 0])
                
                #~ if True:
                    #~ fig, ax = plt.subplots()
                    #~ full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
                    #~ ax.plot(full_waveform.T.flatten(), color='k')
                    #~ ax.set_title('No candidate')
                    #~ plt.show()                
                return cluster_idx, shift, None                
                
            
            
            common_mask = np.sum(self.sparse_mask_level2[candidates_idx, :], axis=0) > 0
            #~ common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
            if candidates_idx.size > 1:
                discriminant_weight = np.zeros((self.peak_width, self.nb_channel), dtype='float32')
                for clus1 in candidates_idx:
                    for clus2 in candidates_idx:
                        if clus1 < clus2:
                            discriminant_weight += np.abs(self.catalogue['centers0'][clus1, :, :] - self.catalogue['centers0'][clus2, :, :])
            else:
                discriminant_weight = np.ones((self.peak_width, self.nb_channel), dtype='float32')
            discriminant_weight[:, ~common_mask] = 0 
            discriminant_weight /= np.sum(discriminant_weight)
            #~ print(discriminant_weight)
            

            #~ shift_scalar_product, shift_distance = numba_explore_best_shift(self.fifo_residuals, left_ind, self.catalogue['centers0'], self.catalogue['centers0_normed'],
                            #~ candidates_idx, self.catalogue['template_weight'], common_mask, self.maximum_jitter_shift, self.catalogue['sparse_mask_level1'])
                            
            shift_scalar_product, shift_distance = numba_explore_best_shift(self.fifo_residuals, left_ind, self.catalogue['centers0'], self.catalogue['centers0_normed'],
                            candidates_idx, discriminant_weight, common_mask, self.maximum_jitter_shift, self.catalogue['sparse_mask_level1'])

            
            
            #~ print(shift_scalar_product)
            #~ print(shift_distance)
            i0, i1 = np.unravel_index(np.argmin(shift_distance, axis=None), shift_distance.shape)
            best_idx = candidates_idx[i0]
            shift = self.shifts[i1]
            
            if shift == self.maximum_jitter_shift or shift == -self.maximum_jitter_shift:
                #~ print('bad shift')
                cluster_idx = -1
                shift = None
                
                #~ if True:
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(self.shifts, shift_distance.T)
                    #~ ax.set_title('bad shift')

                    #~ common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
                    #~ full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
                    #~ wf_common = full_waveform[:, common_mask]
                    
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(wf_common.T.flatten(), color='k')
                    #~ if len(candidates_idx)>0:
                        #~ for idx in candidates_idx:
                            #~ center0 = self.catalogue['centers0'][idx, : , :][:, common_mask]
                            #~ ax.plot(center0.T.flatten(), label = str(idx))
                        #~ ax.legend()
                    #~ plt.show()      

                return cluster_idx, shift, None                
                
            
            distance = None
            
            cluster_idx = best_idx


            
            
            #~ print(shift_distance)
            #~ exit()
            
            
            #~ best_idx = np.argmin(weighted_distance)
            #~ cluster_idx = best_idx
            #~ long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
            #~ shifts_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][cluster_idx, : , :],
                                #~ self.sparse_mask_level4[cluster_idx, :], self.maximum_jitter_shift, self.catalogue['template_weight'])
            #~ ind_min = np.argmin(shifts_dist)
            #~ shift = self.shifts[ind_min]
            #~ distance = shifts_dist[ind_min]


            # debug
            mask = self.sparse_mask_level3[cluster_idx]
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            wf = full_waveform[:, mask]
            center0 = self.catalogue['centers0'][cluster_idx, :, :][:, mask]
            w = self.catalogue['template_weight'][:, mask]
            distance2 = np.sum((wf - center0)**2 * w)
            distance2 /= np.sum(w)
            # debug
            
            #~ if True:
            #~ print(distance2, self.distance_limit[cluster_idx])
            #~ if distance2 >self.distance_limit[cluster_idx] and cluster_idx in (7,9):
            if False:
            #~ if len(candidates_idx) > 1:
                print(scalar_products)
                print(weighted_distance)
                print('possibles_idx1', possibles_idx1)
                print('possibles_idx2', possibles_idx2)
                print('candidates_idx', candidates_idx)
                print('best_idx', best_idx)

                common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
                full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
                wf_common = full_waveform[:, common_mask]
                
                fig, ax = plt.subplots()
                ax.plot(wf_common.T.flatten(), color='k')
                if len(candidates_idx)>0:
                    for idx in candidates_idx:
                        center0 = self.catalogue['centers0'][idx, : , :][:, common_mask]
                        ax.plot(center0.T.flatten(), label = str(idx))
                    ax.legend()


                fig, ax = plt.subplots()
                ax.plot(self.shifts, shift_distance.T)

                fig, ax = plt.subplots()
                ax.plot(self.shifts, shift_scalar_product.T)
                
                
                
                
                #~ plt.show()
                
                plt.show()
            
            
            
        elif self.argmin_method == 'numba_old':
            
            
            
            #TODO remove this rms_waveform_channel no more usefull
            #~ rms_waveform_channel = np.sum(full_waveform**2, axis=0).astype('float32')
            
            #~ possibles_cluster_idx, = np.nonzero(self.sparse_mask_level1[:, chan_ind])
            possibles_cluster_idx, = np.nonzero(self.sparse_mask_level3[:, chan_ind])
            
            if possibles_cluster_idx.size ==0:
                cluster_idx = -1
                shift = None
            else:
                # option by radius
                #~ channel_considered = self.channels_adjacency[chan_ind]
                #~ print()
                #~ print('radius channel_considered', channel_considered)
                
                # option with template sparsity
                #~ print(self.sparse_mask_level3[possibles_cluster_idx, :])
                mask = np.sum(self.sparse_mask_level3[possibles_cluster_idx, :], axis=0)
                #~ np.apply_along_axis(np.logical_or, 0, self.sparse_mask_level3[possibles_cluster_idx, :])
                #~ print(mask)
                channel_considered,  = np.nonzero(mask)
                #~ print('mask channel_considered', channel_considered)
                #~ exit()
                
                
                
                
                #~ print('channel_considered', channel_considered)
                
                #~ mask = np.zeros(self.nb_channel, dtype='bool')
                #~ for clus_idx in possibles_cluster_idx:
                    #~ mask |= self.sparse_mask_level2[clus_idx, :] 
                #~ channel_considered2,  = np.nonzero(mask)
                #~ print('channel_considered2', channel_considered2)
                
                #~ s = numba_loop_sparse_dist_with_geometry(full_waveform, self.catalogue['centers0'],  
                                                        #~ possibles_cluster_idx, rms_waveform_channel,channel_considered)
                
                
                cluster_dist = numba_loop_sparse_dist_with_geometry(full_waveform, self.catalogue['centers0'],
                                                    possibles_cluster_idx, channel_considered, self.catalogue['template_weight'])
                best_cluster_idx1 = possibles_cluster_idx[np.argmin(cluster_dist)]
                shift = None
                
                print()
                print('possibles_cluster_idx', possibles_cluster_idx)
                print('possible labels', self.catalogue['clusters']['cluster_label'][possibles_cluster_idx])
                print('cluster_dist', cluster_dist)
                print('channel_considered', channel_considered)
                
                #~ print('limits', self.catalogue['distance_limit'][possibles_cluster_idx])
                


                #~ # explore shift
                #~ long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
                #~ shifts_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][cluster_idx, : , :],  self.sparse_mask_level2[cluster_idx, :], self.maximum_jitter_shift, self.catalogue['template_weight'])
                #~ ind_min = np.argmin(shifts_dist)
                #~ shift = self.shifts[ind_min]
                #~ distance = shifts_dist[ind_min]
                
                factor = 1.5
                candidates_idx = possibles_cluster_idx[np.nonzero(cluster_dist<(min(cluster_dist) * factor))[0]]
                if len(candidates_idx) == 1:
                #~ if True:
                    cluster_idx= candidates_idx[0]
                    long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
                    shifts_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][cluster_idx, : , :],  self.sparse_mask_level2[cluster_idx, :], self.maximum_jitter_shift, self.catalogue['template_weight'])
                    ind_min = np.argmin(shifts_dist)
                    shift = self.shifts[ind_min]
                    distance = shifts_dist[ind_min]
                    
                else:
                    common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
                    weight = np.std(self.catalogue['centers0'][candidates_idx, : , :], axis=0)
                    weight[:, common_mask] /= np.sum(weight[:, common_mask])
                    #~ print(weight)
                    
                    all_shifts = []
                    all_distances = []
                    #~ all_distance_limit_ratio = []
                    for idx in candidates_idx:
                        # explore shift
                        long_waveform = self.fifo_residuals[left_ind-self.maximum_jitter_shift:left_ind+self.peak_width+self.maximum_jitter_shift+1,:]
                        #~ shifts_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][idx, : , :],  self.sparse_mask_level2[idx, :], self.maximum_jitter_shift, self.catalogue['template_weight'])
                        shifts_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][idx, : , :],  common_mask, self.maximum_jitter_shift, weight)
                        ind_min = np.argmin(shifts_dist)
                        #~ shift = self.shifts[ind_min]
                        #~ distance = shifts_dist[ind_min]
                        all_shifts.append(self.shifts[ind_min])
                        all_distances.append(shifts_dist[ind_min])
                        limit = self.catalogue['distance_limit'][idx]
                        #~ all_distance_limit_ratio.append(shifts_dist[ind_min] / limit)
                    
                    #~ ind_min = np.argmin(all_distance_limit_ratio)
                    #~ best_cluster_idx2 = candidates_idx[ind_min]
                    ind_min = np.argmin(all_distances)
                    cluster_idx = candidates_idx[ind_min]

                    shift = all_shifts[ind_min]
                    distance = all_distances[ind_min] # TODO this is wrong
                    
                    #~ if best_cluster_idx1 != best_cluster_idx2:
                    
                        #~ print()
                        #~ print('possibles_cluster_idx', possibles_cluster_idx)
                        #~ print('s', s)
                        #~ print(min(s))
                        #~ print('candidates_idx', candidates_idx)
                        #~ print('all_shifts', all_shifts)
                        #~ print('all_distances', all_distances)
                        #~ print('best_cluster_idx1', best_cluster_idx1)
                        #~ print('best_cluster_idx2', best_cluster_idx2)
                        
                        #~ common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
                        #~ wf_common = full_waveform[:, common_mask]                    
                        #~ fig, ax = plt.subplots()
                        #~ ax.plot(wf_common.T.flatten(), color='k')
                        #~ for idx in candidates_idx:
                            #~ center0 = self.catalogue['centers0'][idx, : , :][:, common_mask]
                            #~ ax.plot(center0.T.flatten(), label = str(idx))
                        #~ ax.legend()
                        #~ ax.set_title(f'best_cluster_idx1 {best_cluster_idx1} best_cluster_idx2 {best_cluster_idx2}')
                        #~ plt.show()                        
                    
                    
                    #~ cluster_idx = best_cluster_idx2
                    #~ cluster_idx = best_cluster_idx1
                
                #~ if best_cluster_idx1 != best_cluster_idx2:
                            #~ fig, axs = plt.subplots(nrows=2, sharex=True)
                            #~ axs[0].plot(wf_common.T.flatten(), color='k')
                            #~ axs[0].plot(center0_extended.T.flatten(), color='g')
                            #~ axs[0].plot(center0_nearest.T.flatten(), color='m')
                            #~ axs[1].plot(wf_common.T.flatten()-center0_extended.T.flatten(), color='g', ls='--')
                            #~ axs[1].plot(wf_common.T.flatten()-center0_nearest.T.flatten(), color='m', ls='--')
                            #~ plt.show()                    
                    

                #~ if distance > self.catalogue['distance_limit'][cluster_idx]:
                    #~ # no direct accept check other template in nearby
                    #~ other_candidate = {}
                    #~ for cluster_idx_nearest in self.catalogue['nearest_templates'][cluster_idx]:
                        #~ common_mask = self.sparse_mask_level2[cluster_idx] | self.sparse_mask_level2[cluster_idx_nearest]
                        #~ wf_common = full_waveform[:, common_mask]
                        #~ center0_extended = self.catalogue['centers0'][cluster_idx, :, :][:, common_mask]
                        #~ center0_nearest = self.catalogue['centers0'][cluster_idx_nearest, :, :][:, common_mask]
                        
                        #~ weight = np.abs(center0_extended - center0_nearest)
                        #~ dist = np.sum((wf_common - center0_extended)**2 * weight)
                        #~ dist /= np.sum(weight)
                        
                        #~ dist_nearest = np.sum((wf_common - center0_nearest)**2 * weight)
                        #~ dist_nearest /= np.sum(weight)
                        
                        #~ if dist_nearest < dist:
                            #~ other_candidate[cluster_idx_nearest] = dist_nearest / dist
                    
                    #~ if len(other_candidate)>0:
                        #~ cluster_idx_nearest = min(other_candidate, key=lambda x: other_candidate[x])
                        
                        
                        #~ cluster_idx = cluster_idx_nearest
                        #~ all_dist = numba_explore_shifts(long_waveform, self.catalogue['centers0'][cluster_idx, : , :],  self.sparse_mask_level2[cluster_idx, :], self.maximum_jitter_shift, self.catalogue['template_weight'])
                        #~ ind_min = np.argmin(all_dist)
                        #~ shift = self.shifts[ind_min]
                        #~ distance = all_dist[ind_min]

                        #~ label = self.catalogue['clusters']['cluster_label'][cluster_idx]
                
                
                
                
                #~ # experimental explore all shift for all templates!!!!!
                #~ shifts = list(range(-self.maximum_jitter_shift, self.maximum_jitter_shift+1))
                #~ channel_considered = self.channels_adjacency[chan_ind]
                #~ all_s = []
                #~ for shift in shifts:
                    #~ waveform = self.fifo_residuals[left_ind+shift:left_ind+self.peak_width+shift,:]
                    #~ s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  
                                                        #~ possibles_cluster_idx, rms_waveform_channel,channel_considered)                    
                    #~ all_s.append(s)
                #~ all_s = np.array(all_s)
                #~ shift_ind, ind_clus = np.unravel_index(np.argmin(all_s, axis=None), all_s.shape)
                #~ cluster_idx = possibles_cluster_idx[ind_clus]
                #~ shift = shifts[shift_ind]
                #~ distance = all_s[shift_ind][ind_clus]
                
                
                #~ label = self.catalogue['clusters']['cluster_label'][cluster_idx]
                #~ if label == 7:
                    #~ fig, ax = plt.subplots()
                    #~ print(len(possibles_cluster_idx))
                    #~ ax.plot(waveform[:, channel_considered].T.flatten(), color='k')
                    #~ for clus_idx in possibles_cluster_idx:
                        #~ center0 = self.catalogue['centers0'][clus_idx, :, :]
                        #~ if clus_idx == cluster_idx:
                            #~ lw=2
                        #~ else:
                            #~ lw=1
                        
                        #~ k = self.catalogue['clusters']['cluster_label'][clus_idx]
                        #~ color = self.colors[k]
                        #~ print('k', k, color)
                        #~ ax.plot(center0[:, channel_considered].T.flatten(), lw=lw, color=color)
                    
                    #~ label = self.catalogue['clusters']['cluster_label'][cluster_idx]
                    #~ ax.set_title(f'label={label} n_possible{len(possibles_cluster_idx)}')
                    #~ ax.axvline(list(channel_considered).index(chan_ind)*self.peak_width - self.n_left, color='k')
                    
                    #~ fig, ax = plt.subplots()
                    #~ print(cluster_idx)
                    #~ print(possibles_cluster_idx)
                    #~ print(s)
                    #~ for i, clus_idx in enumerate(possibles_cluster_idx):
                        #~ k = self.catalogue['clusters']['cluster_label'][clus_idx]
                        #~ color = self.colors[k]
                        #~ ax.axvline(s[i], color=color)
                
                #~ plt.show()
                
                
                
            
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
        
        #~ print(left_ind, cluster_idx, jitter, distance)
        #~ if self.strict_template:
            #~ return slef.accept_template_sctrict(left_ind, cluster_idx, jitter)
        
        #~ print('distance', distance)
        # DEBUG
        #~ if not hasattr(self, 'nb_14'):
            #~ self.nb_14 = 0
        #~ if self.catalogue['cluster_labels'][cluster_idx] ==14:
            #~ self.nb_14 += 1
            #~ print(self.nb_14)

        #~ mask = self.sparse_mask_level3[cluster_idx]
        #~ mask = self.sparse_mask_level4[cluster_idx]
        #~ full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        #~ wf = full_waveform[:, mask]
        #~ center0 = self.catalogue['centers0'][cluster_idx, :, :][:, mask]
        #~ w = self.catalogue['template_weight'][:, mask]
        #~ distance2 = np.sum((wf - center0)**2 * w)
        #~ distance2 /= np.sum(w)
        
        
        full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        #~ mean_centroid = self.catalogue['mean_centroid']
        #~ projector = self.catalogue['projector']
        #~ feat_centroids = self.catalogue['feat_centroids']
        #~ feat_waveform = (full_waveform.flatten() - mean_centroid) @ projector
        #~ feat_distance = np.sum((feat_waveform - feat_centroids[cluster_idx,:])**2)

        centers0 = self.catalogue['centers0']
        othogonal_bondaries = self.catalogue['othogonal_bondaries']
        inner_othogonal_projector = self.catalogue['inner_othogonal_projector']
        high_boundary = othogonal_bondaries[:, 1]
        low_boundary = othogonal_bondaries[:, 0]
        
        flat_waveform = full_waveform.flatten()
        sp = np.sum((flat_waveform - centers0[cluster_idx, :].flatten()) * inner_othogonal_projector[cluster_idx, :])
        
        #~ high_boundary
        

        #~ print('sp',sp, 'boundary', high_boundary[cluster_idx])
        
        immediate_accept = low_boundary[cluster_idx] < sp < high_boundary[cluster_idx]
        
        #~ print('distance2', distance2, 'distance', distance)

        #~ if distance < self.distance_limit[cluster_idx]:
        #~ if distance2 < self.distance_limit[cluster_idx]:
        if immediate_accept:
            
        #~ if False:
            accept_template = True
            #~ immediate_accept = True
        else:
            
            # criteria multi channel
            #~ mask = self.sparse_mask_level1[cluster_idx]
            #~ mask = self.sparse_mask_level4[cluster_idx]
            mask = self.sparse_mask_level2[cluster_idx]
            #~ full_wf0 = self.catalogue['centers0'][cluster_idx,: , :][:, mask]
            #~ full_wf1 = self.catalogue['centers1'][cluster_idx,: , :][:, mask]
            #~ full_wf2 = self.catalogue['centers2'][cluster_idx,: , :][:, mask]
            if np.sum(mask) == 0:
                # normally not possible unless no clean step
                accept_template = False
            else:
                
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
                
                
                # criteria per channel
                #~ residual_nrj_by_chan = np.sum(dist, axis=0)
                #~ wf_nrj = np.sum(wf**2, axis=0)
                #~ weight = self.weight_per_template_dict[cluster_idx]
                #~ crietria_weighted = (wf_nrj>residual_nrj_by_chan).astype('float') * weight
                #~ accept_template = np.sum(crietria_weighted) >= 0.7 * np.sum(weight)
                
                # criteria per sample
                #~ dist * np.abs(pred_wf) < 
                #~ dist_w = dist / np.abs(pred_wf)
                gain = (dist < wf**2).astype('float') * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
                #~ gain = (wf / pred_wf - 1) * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
                #~ gain = (pred_wf**2 / wf**1 - 1) * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
                #~ accept_template = np.sum(gain) > 0.8
                #~ accept_template = np.sum(gain) > 0.7
                accept_template0 = np.sum(gain) > 0.6
                #~ accept_template = np.sum(gain) > 0.5
                
                # criteria max residual
                max_res = np.max(np.abs(pred_wf - wf))
                max_pred = np.max(np.abs(pred_wf))
                accept_template1 = max_pred > max_res
                
                accept_template = accept_template0 and accept_template1
                #~ accept_template = accept_template0
                #~ accept_template = accept_template1
                
            # DEBUG
            #~ accept_template = False
            
            
            
            
            #~ accept_template = False
            
            # debug
            #~ limit_sp =self.catalogue['sp_normed_limit'][cluster_idx, :]
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform * self.catalogue['template_weight'])
            #~ print('limit_sp', limit_sp, 'sp', sp)
            
            
            
            #~ accept_template = False
            #~ immediate_accept = False
            
            # DEBUG always refuse!!!!!
            #~ accept_template = False
        
        
        label = self.catalogue['cluster_labels'][cluster_idx]
        
        # debug
        #~ if label == 13:
            #~ if accept_template and not immediate_accept:
                #~ accept_template = False
        
        # debug
        #~ if label == 13:
            #~ if not hasattr(self, 'count_accept'):
                #~ self.count_accept = {}
                #~ self.count_accept[label] = {'accept_template':0, 'immediate_accept':0, 'not_accepted':0}
            
            #~ if accept_template:
                #~ self.count_accept[label]['accept_template'] += 1
                #~ if immediate_accept:
                    #~ self.count_accept[label]['immediate_accept'] += 1
            #~ else:
                #~ self.count_accept[label]['not_accepted'] += 1
            #~ print(self.count_accept)
            
        #~ if not accept_template and label in []:
        #~ if not accept_template:
        #~ if accept_template:
        #~ if True:
        if False:
        #~ if not immediate_accept:
        #~ if immediate_accept:
        #~ if immediate_accept:
        #~ if label == 7 and not accept_template:
        #~ if label == 7:
        #~ if label == 121:
        #~ if label == 5:
        
        #~ if label == 13 and accept_template and not immediate_accept:
        #~ if label == 13 and not accept_template:
            
        #~ if label in (7,9):
        #~ nears = np.array([ 5813767,  5813767, 11200038, 11322540, 14989650, 14989673, 14989692, 14989710, 15119220, 15830377, 16138346, 16216666, 17078883])
        #~ print(np.abs((left_ind - self.n_left) - nears))
        #~ print(np.abs((left_ind - self.n_left) - nears) < 2)
        #~ if label == 5 and np.any(np.abs((left_ind - self.n_left) - nears) < 50):
            
            #~ if immediate_accept:
            
            mask = self.sparse_mask_level2[cluster_idx]
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            wf = full_waveform[:, mask]
            _, pred_waveform = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
            pred_wf = pred_waveform[:, mask]
    
            
            if accept_template:
                if immediate_accept:
                    color = 'g'
                else:
                    color = 'c'
            else:
                color = 'r'
            
            #~ if not immediate_accept:
                #~ fig, ax = plt.subplots()
                #~ ax.plot(gain.T.flatten(), color=color)
                #~ ax.set_title('{}'.format(np.sum(gain)))

            #~ fig, ax = plt.subplots()
            #~ ax.plot(feat_centroids.T, alpha=0.5)
            #~ ax.plot(feat_waveform, color='k')

            fig, ax = plt.subplots()
            ax.plot(full_waveform.T.flatten(), color='k')
            ax.plot(pred_waveform.T.flatten(), color=color)
            
            l0, l1 = othogonal_bondaries[cluster_idx, :]
            title = f'{cluster_idx} {sp:0.3f} lim [{l0:0.3f} {l1:0.3f}]'
            ax.set_title(title)
                
            #~ fig, ax = plt.subplots()
            #~ ax.plot(wf.T.flatten(), color='k')
            #~ ax.plot(pred_wf.T.flatten(), color=color)
            
            #~ ax.plot( wf.T.flatten() - pred_wf.T.flatten(), color=color, ls='--')
            
            print()
            print('cluster_idx',cluster_idx, 'immediate_accept', immediate_accept, 'accept_template', accept_template)
            #~ print(distance, self.distance_limit[cluster_idx])
            #~ print('distance', distance, distance2, 'limit_distance', self.distance_limit[cluster_idx])

            #~ limit_sp =self.catalogue['sp_normed_limit'][cluster_idx, :]
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform * self.catalogue['template_weight'])
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform)
            #~ print('limit_sp', limit_sp, 'sp', sp)
            
            #~ if not immediate_accept:
                #~ print('np.sum(gain)', np.sum(gain))


            #~ fig, ax = plt.subplots()
            #~ res = wf - pred_wf
            #~ count, bins = np.histogram(res, bins=150, weights=np.abs(pred_wf))
            #~ ax.plot(bins[:-1], count)
            #~ plt.show()

            
            
            #~ if distance2 >= self.distance_limit[cluster_idx]:
                #~ print(crietria_weighted, weight)
                #~ print(np.sum(crietria_weighted),  np.sum(weight))
            
            #~ ax.plot(full_wf0.T.flatten(), color='y')
            #~ ax.plot( full_wf.T.flatten() - full_wf0.T.flatten(), color='y')
            
            #~ ax.set_title('not accepted')
            plt.show()
        
        return accept_template
    
    #~ def get_best_collision(self, left_ind, chan_ind):
        #~ if not np.any(self.sparse_mask_level3[:, chan_ind]):
            #~ # no template near that channel
            #~ cluster_idx = -1
            #~ shift = None
            #~ return cluster_idx, shift, None
        
        #~ if self.argmin_method == 'opencl':
            #~ raise NotImplementedError
        #~ elif self.argmin_method == 'numba':
            #~ channel_adjacency = self.channels_adjacency[chan_ind]
            
            #~ waveform_all_chans = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            #~ waveform = waveform_all_chans[:, channel_adjacency]
            #~ waveform_flat = waveform.T.flatten()
            #~ waveform_one_chan = waveform_all_chans[:, chan_ind]
            
            #~ possibles_cluster_idx, = np.nonzero(self.sparse_mask_level3[:, chan_ind])
            
            #~ for clus_idx_a, clus_idx_b in itertools.combinations(possibles_cluster_idx, 2):
                #~ print(clus_idx_a, clus_idx_b)
                #~ k_a = self.catalogue['clusters']['cluster_label'][clus_idx_a]
                #~ color_a = self.colors[k_a]
                #~ k_b = self.catalogue['clusters']['cluster_label'][clus_idx_b]
                #~ color_b = self.colors[k_b]
                
                
                #~ Fp = self.catalogue['centers1'][[clus_idx_a, clus_idx_b], :][:, :, chan_ind]
                #~ print(Fp.shape)
                
                #~ center0_a = self.catalogue['centers0'][clus_idx_a, :][:, channel_adjacency]
                #~ center0_b = self.catalogue['centers0'][clus_idx_b, :][:, channel_adjacency]
                
                
                #~ delta = np.linalg.inv(Fp.T.dot(Fp)).dot(Fp.T).T.dot(waveform_one_chan)
                #~ print(delta)
                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(waveform.T.flatten(), color='k')
                #~ ax.plot(center0_a.T.flatten(), color=color_a)
                #~ ax.plot(center0_b.T.flatten(), color=color_b)
                
                #~ ax.axvline(list(channel_adjacency).index(chan_ind)*self.peak_width - self.n_left, color='k')


                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(Fp[0,], color=color_a)
                #~ ax.plot(Fp[1,], color=color_b)
                
                #~ plt.show()
                
            #~ exit()
            
    
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
            #~ immediate_accept = True
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
        #~ print('LABEL UNCLASSIFIED', left_ind, cluster_idx)
        fig, ax = plt.subplots()
        
        wf = self.fifo_residuals[left_ind:left_ind+self.peak_width, :]
        wf0 = self.catalogue['centers0'][cluster_idx, :, :]
        
        ax.plot(wf.T.flatten(), color='b')
        #~ ax.plot(wf0.T.flatten(), color='g')
        
        ax.set_title(f'label_unclassified {left_ind-self.n_left} {cluster_idx} chan{peak_chan}')
        
        ax.axvline(peak_chan*self.peak_width-self.n_left)
        
        plt.show()

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

    


