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
    from .numba_tools import numba_explore_best_shift, numba_sparse_scalar_product
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
        
        
        self.mask_already_tested = np.zeros((self.fifo_size, self.nb_channel), dtype='bool')
        
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
        # reset tested mask
        self.mask_already_tested[:] = False
        # and detect peak
        self.re_detect_local_peak()
        
        #~ print('detect_local_peaks_before_peeling_loop', self.pending_peaks.size)

    def re_detect_local_peak(self):
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        if mask.ndim ==1:
            #~ mask &= ~self.mask_already_tested[self.n_span:-self.n_span, 0]
            sample_indexes,   = np.nonzero(mask)
            sample_indexes += self.n_span
            tested = self.mask_already_tested[sample_indexes, 0]
            sample_indexes = sample_indexes[~tested]
            chan_indexes = np.zeros(sample_indexes.size, dtype='int64')
        else:
            #~ mask &= ~self.mask_already_tested[self.n_span:-self.n_span, :]
            sample_indexes, chan_indexes  = np.nonzero(mask)
            sample_indexes += self.n_span
            tested = self.mask_already_tested[sample_indexes, chan_indexes]
            sample_indexes = sample_indexes[~tested]
            chan_indexes = chan_indexes[~tested]
        
        
        amplitudes = np.abs(self.fifo_residuals[sample_indexes, chan_indexes])
        order = np.argsort(amplitudes)[::-1]
        
        #~ self.pending_peaks = list(zip(sample_indexes[order], chan_indexes[order]))
        #~ self.already_tested = []
        
        dtype_peak = [('sample_index', 'int32'), ('chan_index', 'int32'), ('peak_value', 'float32')]
        self.pending_peaks = np.zeros(sample_indexes.size, dtype=dtype_peak)
        self.pending_peaks['sample_index'] = sample_indexes
        self.pending_peaks['chan_index'] = chan_indexes
        self.pending_peaks['peak_value'] = amplitudes
        self.pending_peaks = self.pending_peaks[order]
        #~ print('re_detect_local_peak', self.pending_peaks.size)

    
    def select_next_peak(self):
        #~ print(len(self.pending_peaks))
        if len(self.pending_peaks)>0:
            sample_ind, chan_ind, ampl = self.pending_peaks[0]
            self.pending_peaks = self.pending_peaks[1:]
            return sample_ind, chan_ind
        else:
            return LABEL_NO_MORE_PEAK, None

    #~ def on_accepted_spike(self, spike):
    def on_accepted_spike(self, sample_ind, cluster_idx, jitter):
        # remove spike prediction from fifo residuals
        #~ left_ind = spike.index + self.n_left
        #~ cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
        #~ peak_index = spike.index
        #~ pos, pred = make_prediction_one_spike(spike.index, cluster_idx, spike.jitter, self.fifo_residuals.dtype, self.catalogue)
        pos, pred = make_prediction_one_spike(sample_ind, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
        
        # this prevent search peaks in the zone until next "reset_to_not_tested"
        #~ self.clean_pending_peaks_zone(spike.index, cluster_idx)
        self.clean_pending_peaks_zone(sample_ind, cluster_idx)

    def clean_pending_peaks_zone(self, sample_ind, cluster_idx):
        # TODO test with sparse_mask_level3s!!!!!
        mask = self.sparse_mask_level1[cluster_idx, :]
        #~ keep = [not((ind == sample_ind) and (mask[chan_ind])) for ind, chan_ind in self.pending_peaks]
        #~ keep = [(ind != sample_ind) and not(mask[chan_ind]) for ind, chan_ind in self.pending_peaks]
        
        #~ pending_peaks_ = []
        keep = np.zeros(self.pending_peaks.size, dtype='bool')
        for i, peak in enumerate(self.pending_peaks):
            #~ ok = (ind != sample_ind) and not(mask[chan_ind])
            in_zone = mask[peak['chan_index']] and (peak['sample_index']+self.n_left<sample_ind<peak['sample_index']+self.n_right)
            if in_zone:
                #~ self.already_tested.append((ind, chan_ind)) # ERREUR!!!!!!!
                pass
                keep[i] = False
            else:
                keep[i] = True
        self.pending_peaks = self.pending_peaks[keep]
        
        #~ print('clean_pending_peaks_zone', self.pending_peaks.size)
    
    def set_already_tested(self, sample_ind, peak_chan):
        #~ self.mask_not_already_tested[sample_ind - self.n_span, peak_chan] = False
        #~ self.pending_peaks = [p for p in self.pending_peaks if (p[0]!=sample_ind) and (peak_chan!=p[1])]
        #~ self.already_tested.append((sample_ind, peak_chan))
        
        self.mask_already_tested[sample_ind, peak_chan] = True

    def reset_to_not_tested(self, good_spikes):
        #~ print('reset_to_not_tested', len(good_spikes))
        #~ self.already_tested = []
        #~ print('self.already_tested', len(self.already_tested))
        for spike in good_spikes:
            # each good spike can remove from
            cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
            chan_mask = self.sparse_mask_level1[cluster_idx, :]
            #~ self.already_tested = [ p for p in self.already_tested if not((np.abs(p[0]-spike.index)<self.peak_width)  and mask[p[1]] ) ]
            self.mask_already_tested[spike.index + self.n_left:spike.index + self.n_right][:, chan_mask] = False
        #~ print('self.already_tested reduced', len(self.already_tested))
        # 
        
        self.re_detect_local_peak()
        
        #~ mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        
        #~ if mask.ndim ==1:
            #~ sample_indexes,   = np.nonzero(mask)
            #~ chan_indexes = np.zeros(sample_indexes.size, dtype='int64')
        #~ else:
            #~ sample_indexes, chan_indexes  = np.nonzero(mask)

        #~ sample_indexes += self.n_span
        #~ amplitudes = np.abs(self.fifo_residuals[sample_indexes, chan_indexes])
        #~ order = np.argsort(amplitudes)[::-1]
        #~ possible_pending_peaks = list(zip(sample_indexes[order], chan_indexes[order]))
        
        #~ self.pending_peaks = []
        #~ for peak in possible_pending_peaks:
            #~ if peak not in self.already_tested:
                #~ self.pending_peaks.append(peak)

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
            projections = self.catalogue['projections']

            strict_low = self.catalogue['boundaries'][:, 0]
            strict_high = self.catalogue['boundaries'][:, 1]
            flexible_low = self.catalogue['boundaries'][:, 2]
            flexible_high = self.catalogue['boundaries'][:, 3]
            
            
            n = centers0.shape[0]
            flat_waveform = full_waveform.flatten()
            flat_centers0 = centers0.reshape(n, -1)
            
            #~ scalar_products = np.zeros(n, dtype='float32')
            #~ for i in range(n):
                #~ sp = np.sum((flat_waveform - flat_centers0[i, :]) * projections[i, :])
                #~ scalar_products[i] = sp
            #~ scalar_products = np.sum((flat_waveform[np.newaxis, :] - flat_centers0[:, :]) * projections[:, :], axis=1)
            #~ print(scalar_products)
            
            scalar_products = numba_sparse_scalar_product(self.fifo_residuals, left_ind, centers0, projections, chan_ind,
                        self.sparse_mask_level1, )
            #~ print(scalar_products)
            
            
            possible_idx, = np.nonzero((scalar_products < strict_high) & (scalar_products > strict_low))
            #~ possible_idx, = np.nonzero((scalar_products < flexible_high) & (scalar_products > flexible_low))
            
            
            #~ do_plot = False
            if len(possible_idx) == 1:
                extra_idx = None
                candidates_idx =possible_idx
            elif len(possible_idx) == 0:
                #~ extra_idx, = np.nonzero((np.abs(scalar_products) < 0.5))
                extra_idx, = np.nonzero((scalar_products < flexible_high) & (scalar_products > flexible_low))
                #~ if len(extra_idx) ==0:
                    # give a try to very far ones.
                    #~ extra_idx, = np.nonzero((np.abs(scalar_products) < 1.))
                    #~ print('extra_idx', extra_idx)
                #~ if len(extra_idx) ==0:
                    #~ candidates_idx = []
                #~ else:
                    #~ candidates_idx = extra_idx
                candidates_idx = extra_idx
                #~ candidates_idx =possible_idx
                #~ pass
            elif len(possible_idx) > 1 :
                extra_idx = None
                candidates_idx = possible_idx
            
            debug_plot_change = False
            if len(candidates_idx) > 0:
                candidates_idx = np.array(candidates_idx, dtype='int64')
                common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
                shift_scalar_product, shift_distance = numba_explore_best_shift(self.fifo_residuals, left_ind, self.catalogue['centers0'],
                                self.catalogue['projections'], candidates_idx, self.maximum_jitter_shift, common_mask, self.sparse_mask_level1)
                #~ i0, i1 = np.unravel_index(np.argmin(np.abs(shift_scalar_product), axis=None), shift_scalar_product.shape)
                i0, i1 = np.unravel_index(np.argmin(shift_distance, axis=None), shift_distance.shape)
                #~ best_idx = candidates_idx[i0]
                shift = self.shifts[i1]
                cluster_idx = candidates_idx[i0]
                if np.abs(shift) == self.maximum_jitter_shift:
                    cluster_idx = None
                    shift = None
                    #~ print('maximum_jitter_shift >> cluster_idx = None ')
                    #~ do_plot = True
                #~ i0_bis, i1_bis = np.unravel_index(np.argmin(np.abs(shift_scalar_product), axis=None), shift_scalar_product.shape)
                #~ if i0 != i0_bis:
                    
                    #~ debug_plot_change = True
                    #~ print('Warning')
                    #~ print(possible_idx)
                    #~ print(shift_scalar_product)
                    #~ print(shift_distance)
                
                
                
                
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
            #~ if do_plot:
            if False:
            #~ if True:
            #~ if len(possible_idx) != 1:
            #~ if len(possible_idx) > 1:
            
            #~ if 7 in possible_idx or  cluster_idx == 7:
            #~ if cluster_idx not in possible_idx and len(possible_idx) > 0:
            #~ if debug_plot_change:
            
                print()
                print('best cluster_idx', cluster_idx)
                print('possible_idx', possible_idx)
                print('extra_idx', extra_idx)
                print(scalar_products[possible_idx])
                print(strict_high[possible_idx])
                
                
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
        
        return cluster_idx, shift, distance

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
        projections = self.catalogue['projections']
        
        strict_low = self.catalogue['boundaries'][:, 0]
        strict_high = self.catalogue['boundaries'][:, 1]
        flexible_low = self.catalogue['boundaries'][:, 2]
        flexible_high = self.catalogue['boundaries'][:, 3]
        
        
        flat_waveform = full_waveform.flatten()
        sp = np.sum((flat_waveform - centers0[cluster_idx, :].flatten()) * projections[cluster_idx, :])
        
        #~ strict_high
        

        #~ print('sp',sp, 'boundary', strict_high[cluster_idx])
        
        #~ immediate_accept = strict_low[cluster_idx] < sp < strict_high[cluster_idx]
        immediate_accept = flexible_low[cluster_idx] < sp < flexible_high[cluster_idx]
        
        #~ print('distance2', distance2, 'distance', distance)

        #~ if distance < self.distance_limit[cluster_idx]:
        #~ if distance2 < self.distance_limit[cluster_idx]:
        if immediate_accept:
            
        #~ if False:
            accept_template = True
            #~ immediate_accept = True

        
            #~ mask = self.sparse_mask_level2[cluster_idx]
            #~ full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            #~ wf = full_waveform[:, mask]
            #~ _, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
            #~ pred_wf = pred_wf[:, mask]
            
            #~ max_res = np.max(np.abs(pred_wf - wf))
            #~ max_pred = np.max(np.abs(pred_wf))
            #~ accept_template1 = max_pred > (max_res * 0.5)
            
            #~ accept_template = immediate_accept and accept_template1
            
            
            
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
                
                accept_template2 = flexible_low[cluster_idx] < sp < flexible_high[cluster_idx]
                
                accept_template = accept_template0 and accept_template1
                #~ accept_template = accept_template0 and accept_template1 and accept_template2
                #~ accept_template = accept_template0
                #~ accept_template = accept_template1
                
                #~ # DEBUG
                accept_template = False
            
            
            
            
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
            
        if self._plot_debug:
        #~ if not accept_template and label in []:
        #~ if not accept_template:
        #~ if accept_template:
        #~ if True:
        #~ if False:
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
            
            l0, l1 = strict_low[cluster_idx], strict_high[cluster_idx]
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
        sample_inds, chan_inds= np.nonzero(mask)
        sample_inds += self.n_span
        
        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        ax.set_title(f'nb peak {sample_inds.size}')
        
        plt.show()
        
    
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
        sample_inds, chan_inds= np.nonzero(mask)
        sample_inds += self.n_span
        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        
        
        
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

    


