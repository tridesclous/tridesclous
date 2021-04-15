import time
import numpy as np

from pprint import pprint

import sklearn.metrics

from .peeler_tools import *
from .peeler_tools import _dtype_spike
from . import signalpreprocessor

from .cltools import HAVE_PYOPENCL, OpenCL_Helper


from .peeler_engine_base import PeelerEngineGeneric

# import kernels
#~ from .signalpreprocessor import processor_kernel
from .peakdetector import opencl_kernel_geometrical_part2


try:
    import pyopencl
    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False




class PeelerEngineGeometricalCl(PeelerEngineGeneric):
    def change_params(self, adjacency_radius_um=100, use_opencl2=False, **kargs): # high_adjacency_radius_um=50, 
        assert HAVE_PYOPENCL
        
        PeelerEngineGeneric.change_params(self, **kargs)
        
        #~ assert self.use_sparse_template
        
        # Note the same radius as self.catalogue['peak_detector_params']['']
        # this one is used for explore template around the chann detection
        self.adjacency_radius_um = adjacency_radius_um
        
        self.use_opencl2 = use_opencl2
        
        self.shifts = np.arange(-self.maximum_jitter_shift, self.maximum_jitter_shift+1)


    def initialize(self, **kargs):
        assert not self.save_bad_label # TODO add this feature when this work
        
        assert self.internal_dtype == 'float32'
        
        OpenCL_Helper.initialize_opencl(self, cl_platform_index=self.cl_platform_index, cl_device_index=self.cl_device_index)
        
        PeelerEngineGeneric.initialize(self, processor_engine='opencl', **kargs)
        

        # some attrs
        self.shifts = np.arange(-self.maximum_jitter_shift, self.maximum_jitter_shift+1)
        self.nb_shift = self.shifts.size
        
        # make neighbours for peak detector CL
        d = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
        self.channel_distances = d.astype('float32')
        neighbour_mask = d<=self.catalogue['peak_detector_params']['adjacency_radius_um']
        nb_neighbour_per_channel = np.sum(neighbour_mask, axis=0)
        nb_max_neighbour = np.max(nb_neighbour_per_channel)
        self.nb_max_neighbour = nb_max_neighbour # include itself
        self.neighbours = np.zeros((self.nb_channel, nb_max_neighbour), dtype='int32')
        self.neighbours[:] = -1
        for c in range(self.nb_channel):
            neighb, = np.nonzero(neighbour_mask[c, :])
            self.neighbours[c, :neighb.size] = neighb
        
        #~ self.mask_peaks = np.zeros((self.fifo_size - 2 * self.n_span, self.nb_channel), dtype='uint8')  # bool
        
        # debug to check same ctx and queue as processor
        if self.signalpreprocessor is not None:
            assert self.ctx is self.signalpreprocessor.ctx
        
        wf_size = self.peak_width * self.nb_channel
        wf_size_long = self.peak_width_long * self.nb_channel
        
        if self.inter_sample_oversampling:
            subsample_ratio = self.catalogue['subsample_ratio']
        else:
            # not used
            subsample_ratio = 1  

        kernel_formated = kernel_peeler_cl % dict(
                    chunksize=self.chunksize,
                    n_span=self.n_span,
                    nb_channel=self.nb_channel,
                    nb_cluster=self.nb_cluster,
                    relative_threshold=self.relative_threshold,
                    peak_sign={'+':1, '-':-1}[self.peak_sign],
                    extra_size=self.extra_size,
                    fifo_size=self.fifo_size,
                    n_left=self.n_left,
                    n_right=self.n_right,
                    n_left_long=self.n_left_long,
                    n_right_long=self.n_right_long,
                    peak_width=self.peak_width,
                    peak_width_long=self.peak_width_long,
                    maximum_jitter_shift=self.maximum_jitter_shift,
                    n_cluster=self.nb_cluster,
                    wf_size=wf_size,
                    wf_size_long=wf_size_long,
                    subsample_ratio=subsample_ratio,
                    nb_neighbour=self.nb_max_neighbour, 
                    inter_sample_oversampling=int(self.inter_sample_oversampling),
                    nb_shift=self.nb_shift,
                    )
        
        # kernel size
        # explore_templates
        gsize = self.nb_cluster * self.nb_channel
        global_size = (self.nb_cluster, self.nb_channel)
        if gsize > self.max_wg_size:
            n = int(np.ceil(self.nb_channel / self.max_wg_size))
            local_size = (1, self.nb_channel)
        else:
            local_size = global_size
        self.sizes_explore_templates = (global_size, local_size)
        
        # get_candidate_template
        gsize = self.nb_cluster
        global_size = (self.nb_cluster, )
        if gsize > self.max_wg_size:
            local_size = (self.max_wg_size, )
        else:
            local_size = global_size
        self.sizes_get_candidate_template = (global_size, local_size)
        
        # explore_shifts
        global_size = (self.nb_cluster, self.nb_shift, self.nb_channel)
        if self.nb_cluster > self.max_wg_size:
            local_size = (self.max_wg_size, 1, 1)
        elif (self.nb_cluster * self.nb_shift) < self.max_wg_size:
            local_size = (self.nb_cluster, self.nb_shift, 1)
        else:
            local_size = (self.nb_cluster, 1, 1)
        self.sizes_explore_shifts = (global_size, local_size)
        

        if self.use_opencl2:
            kernel_opencl2_extention_formated = kernel_opencl2_extention % dict(
                ls_0 = self.sizes_explore_templates[1][0]
                )
            kernel_formated = kernel_formated + kernel_opencl2_extention_formated
            prg = pyopencl.Program(self.ctx, kernel_formated)
            #~ self.opencl_prg = prg.build(options='-cl-mad-enable')
            self.opencl_prg = prg.build(options='-cl-std=CL2.0')
        else:
            prg = pyopencl.Program(self.ctx, kernel_formated)
            self.opencl_prg = prg.build(options='')
        

        # create CL buffers
        self.fifo_residuals_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_residuals)
        
        self.channel_distances_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.channel_distances)
        
        self.sigs_chunk = np.zeros((self.chunksize, self.nb_channel), dtype='float32')
        self.sigs_chunk_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sigs_chunk)

        self.neighbours_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.neighbours)
        


        self.scalar_products = np.zeros((self.nb_cluster), dtype='float32')
        self.scalar_products_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.scalar_products)

        self.scalar_products_shifts = np.zeros((self.nb_cluster, self.shifts.size, ), dtype='float32')
        self.scalar_products_shifts_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.scalar_products_shifts)

        self.final_scalar_product = np.zeros(1, dtype='float32')
        self.final_scalar_product_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.final_scalar_product)
        
        
        
        self.distance_shifts = np.zeros((self.nb_cluster, self.shifts.size, ), dtype='float32')
        self.distance_shifts_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.distance_shifts)
        
        self.sparse_mask_level1_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask_level1.astype('u1'))
        self.sparse_mask_level2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask_level2.astype('u1'))
        self.sparse_mask_level3_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask_level3.astype('u1'))
        
        self.projections_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.projections.astype('float32'))
        self.boundaries_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.boundaries.astype('float32'))
        

        dtype_spike = [('sample_ind', 'int32'), ('cluster_idx', 'int32'), ('jitter', 'float32')]
        
        self.next_spike = np.zeros(1, dtype=dtype_spike)
        self.next_spike_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.next_spike)

        self.catalogue_center0_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers0'].astype('float32'))
        self.catalogue_center0_long_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers0_long'].astype('float32'))
        
        if self.inter_sample_oversampling:
            self.catalogue_center1_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers1'].astype('float32'))
            self.catalogue_center2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers2'].astype('float32'))
            self.catalogue_inter_center0_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['interp_centers0'].astype('float32'))
            
            self.wf1_norm2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['wf1_norm2'].astype('float32'))
            self.wf2_norm2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['wf2_norm2'].astype('float32'))
            self.wf1_dot_wf2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['wf1_dot_wf2'].astype('float32'))
        else:
            fake_buf = np.zeros((1), dtype='float32')
            self.catalogue_center1_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=fake_buf)
            self.catalogue_center2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=fake_buf)
            self.catalogue_inter_center0_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=fake_buf)
            
            self.wf1_norm2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=fake_buf)
            self.wf2_norm2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=fake_buf)
            self.wf1_dot_wf2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=fake_buf)
            
        extremum_channel = self.catalogue['clusters']['extremum_channel'].astype('int32')
        #~ self.extremum_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['extremum_channel'].astype('int32'))
        self.extremum_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=extremum_channel)
        
        #~ self.weight_per_template = np.zeros((self.nb_cluster, self.nb_channel), dtype='float32')
        #~ centers = self.catalogue['centers0']
        #~ for i, k in enumerate(self.catalogue['cluster_labels']):
            #~ mask = self.sparse_mask_level3[i, :]
            #~ wf = centers[i, :, :][:, mask]
            #~ self.weight_per_template[i, mask] = np.sum(wf**2, axis=0)
        #~ self.weight_per_template_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.weight_per_template.astype('float32'))
        
        
        # TODO estimate smaller 
        nb_max_spike_in_chunk = self.nb_channel * self.fifo_size
        #~ print('nb_max_spike_in_chunk', nb_max_spike_in_chunk)
        
        
        dtype_peak = [('sample_ind', 'int32'), ('chan_index', 'int32'), ('peak_value', 'float32')]
        
        self.next_peak = np.zeros(1, dtype=dtype_peak)
        self.next_peak_cl =  pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.next_peak)
        
        self.pending_peaks = np.zeros(nb_max_spike_in_chunk, dtype=dtype_peak)
        self.pending_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.pending_peaks)
        
        self.nb_pending_peaks = np.zeros(1, dtype='int32')
        self.nb_pending_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.nb_pending_peaks)

        self.good_spikes = np.zeros(nb_max_spike_in_chunk, dtype=dtype_spike)
        self.good_spikes_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.good_spikes)

        self.nb_good_spikes = np.zeros(1, dtype='int32')
        self.nb_good_spikes_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.nb_good_spikes)
        
        self.mask_already_tested = np.zeros((self.fifo_size, self.nb_channel), dtype='uint8') # bool
        self.mask_already_tested_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.mask_already_tested)
        
        
        dtype_candidate = [('strict', 'uint8'), ('flexible', 'uint8')]
        self.candidate_template = np.zeros(self.nb_cluster, dtype=dtype_candidate)
        self.candidate_template_cl =  pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.candidate_template)

        self.nb_candidate = np.zeros(1, dtype='int32')
        self.nb_candidate_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.nb_candidate)

        self.nb_flexible_candidate = np.zeros(1, dtype='int32')
        self.nb_flexible_candidate_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.nb_flexible_candidate)
        
        self.common_mask = np.zeros(self.nb_channel, dtype='uint8')
        self.common_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.common_mask)
        
        
        

        # kernel calls

        self.kern_roll_fifo = getattr(self.opencl_prg, 'roll_fifo')
        self.fifo_roll_size = self.fifo_size - self.chunksize
        assert self.fifo_roll_size < self.chunksize, 'roll fifo size is smaller thatn new buffer size'
        self.kern_roll_fifo.set_args(self.fifo_residuals_cl,
                                                                np.int32(self.fifo_roll_size))

        
        self.kern_add_fifo_residuals = getattr(self.opencl_prg, 'add_fifo_residuals')
        fifo_roll_size = self.fifo_size - self.chunksize
        self.kern_add_fifo_residuals.set_args(self.fifo_residuals_cl,
                                                                self.sigs_chunk_cl,
                                                                np.int32(fifo_roll_size))
        
        self.kern_detect_local_peaks = getattr(self.opencl_prg, 'detect_local_peaks')
        self.kern_detect_local_peaks.set_args(self.fifo_residuals_cl,
                                                                self.neighbours_cl,
                                                                self.mask_already_tested_cl,
                                                                self.pending_peaks_cl,
                                                                self.nb_pending_peaks_cl)
        
        
        if self.alien_value_threshold is None or np.isnan(self.alien_value_threshold):
            alien_value_threshold = np.float32(-1.)
        else:
            alien_value_threshold = np.float32(self.alien_value_threshold)
        #~ print('alien_value_threshold', alien_value_threshold)


        self.kern_select_next_peak = getattr(self.opencl_prg, 'select_next_peak')
        self.kern_select_next_peak.set_args(self.fifo_residuals_cl,
                                                            self.next_peak_cl,
                                                            self.pending_peaks_cl,
                                                            self.nb_pending_peaks_cl,
                                                            self.next_spike_cl,
                                                            alien_value_threshold,
                                                            )
    
        self.kern_explore_templates = getattr(self.opencl_prg, 'explore_templates')
        self.kern_explore_templates.set_args(self.fifo_residuals_cl,
                                                        self.next_spike_cl,
                                                        self.next_peak_cl,
                                                        self.catalogue_center0_cl,
                                                        self.sparse_mask_level1_cl,
                                                        self.scalar_products_cl,
                                                        self.projections_cl,
                                                        )
        
        
        self.kern_get_candidate_template = getattr(self.opencl_prg, 'get_candidate_template')
        self.kern_get_candidate_template.set_args(
                                                                self.next_spike_cl,
                                                                self.scalar_products_cl,
                                                                self.boundaries_cl,
                                                                self.candidate_template_cl,
                                                                self.nb_candidate_cl,
                                                                self.nb_flexible_candidate_cl,
                                                                )
        
        self.kern_make_common_mask = getattr(self.opencl_prg, 'make_common_mask')
        self.kern_make_common_mask.set_args(
                                                                self.next_spike_cl,
                                                                self.candidate_template_cl,
                                                                self.nb_candidate_cl,
                                                                self.nb_flexible_candidate_cl,
                                                                self.sparse_mask_level3_cl,
                                                                self.common_mask_cl,
                                                                )        
        
        self.kern_explore_shifts = getattr(self.opencl_prg, 'explore_shifts')
        self.kern_explore_shifts.set_args(self.fifo_residuals_cl,
                                                        self.next_spike_cl,
                                                        self.next_peak_cl,
                                                        self.catalogue_center0_cl,
                                                        self.sparse_mask_level1_cl,
                                                        self.projections_cl,
                                                        self.candidate_template_cl,
                                                        self.scalar_products_shifts_cl,
                                                        self.distance_shifts_cl,
                                                        self.common_mask_cl,
                                                        )

        self.kern_best_shift_and_jitter = getattr(self.opencl_prg, 'best_shift_and_jitter')
        self.kern_best_shift_and_jitter.set_args(self.fifo_residuals_cl,
                                                        self.next_spike_cl,
                                                        self.next_peak_cl,
                                                        
                                                        self.catalogue_center0_cl,
                                                        self.catalogue_center1_cl,
                                                        self.catalogue_center2_cl,
                                                        self.catalogue_inter_center0_cl,
                                                        self.extremum_channel_cl,
                                                        self.wf1_norm2_cl,
                                                        self.wf2_norm2_cl,
                                                        self.wf1_dot_wf2_cl,
                                                        
                                                        self.nb_candidate_cl,
                                                        self.candidate_template_cl,
                                                        self.scalar_products_shifts_cl,
                                                        self.distance_shifts_cl,
                                                        self.final_scalar_product_cl,
                                                        )
        
        
        self.kern_finalize_next_spike = getattr(self.opencl_prg, 'finalize_next_spike')
        self.kern_finalize_next_spike.set_args(self.fifo_residuals_cl,
                                                        self.next_spike_cl,
                                                        self.next_peak_cl,
                                                        self.pending_peaks_cl,
                                                        self.nb_pending_peaks_cl,
                                                        self.good_spikes_cl,
                                                        self.nb_good_spikes_cl,
                                                        self.mask_already_tested_cl,
                                                        self.catalogue_center0_cl,
                                                        self.catalogue_center1_cl,
                                                        self.catalogue_center2_cl,
                                                        self.catalogue_inter_center0_cl,
                                                        self.sparse_mask_level1_cl,
                                                        self.sparse_mask_level2_cl,
                                                        self.distance_shifts_cl,
                                                        self.extremum_channel_cl,
                                                        self.wf1_norm2_cl,
                                                        self.wf2_norm2_cl,
                                                        self.wf1_dot_wf2_cl,
                                                        self.boundaries_cl,
                                                        self.final_scalar_product_cl,
                                                        )

        self.kern_remove_spike_from_fifo = getattr(self.opencl_prg, 'remove_spike_from_fifo')
        self.kern_remove_spike_from_fifo.set_args(self.fifo_residuals_cl,
                                                        self.next_spike_cl,
                                                        self.catalogue_center0_long_cl,
                                                        self.catalogue_inter_center0_cl,
                                                        )

        self.kern_reset_tested_zone = getattr(self.opencl_prg, 'reset_tested_zone')
        self.kern_reset_tested_zone.set_args(
                                                self.mask_already_tested_cl,
                                                self.good_spikes_cl,
                                                self.nb_good_spikes_cl,
                                                self.sparse_mask_level1_cl,
                                                )

        if self.use_opencl2:
            self.kern_classify_and_align_next_spike = getattr(self.opencl_prg, 'classify_and_align_next_spike')
            self.kern_classify_and_align_next_spike.set_args(self.fifo_residuals_cl,
                                                            self.next_spike_cl,
                                                            self.next_peak_cl,
                                                            self.pending_peaks_cl,
                                                            self.nb_pending_peaks_cl,
                                                            alien_value_threshold,
                                                            
                                                            self.catalogue_center0_cl,
                                                            self.catalogue_center1_cl,
                                                            self.catalogue_center2_cl,
                                                            self.catalogue_inter_center0_cl,
                        
                                                            self.sparse_mask_level1_cl,
                                                            self.sparse_mask_level2_cl,
                                                            self.sparse_mask_level3_cl,
                                                            self.scalar_products_cl,
                                                            self.distance_shifts_cl,
                                                            
                                                            self.channel_distances_cl,
                                                            np.float32(self.adjacency_radius_um),
                                                            
                                                            self.extremum_channel_cl,
                                                            self.wf1_norm2_cl,
                                                            self.wf2_norm2_cl,
                                                            self.wf1_dot_wf2_cl,
                                                            #~ self.best_distance_cl,

                                                            self.good_spikes_cl,
                                                            self.nb_good_spikes_cl,
                                                            self.mask_already_tested_cl,
                                                            self.weight_per_template_cl,
                                                            self.distance_limit_cl,
                                                            
                                                            )


    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        #~ print('initialize_before_each_segment', kargs)
        if kargs['already_processed']:
            self.kern_add_fifo_residuals.set_args(self.fifo_residuals_cl,
                                                                    self.sigs_chunk_cl,
                                                                    np.int32(self.fifo_roll_size))
        else:
            # this contain the output of preprocessor
            # self.signalpreprocessor.output_backward_cl

            self.kern_add_fifo_residuals.set_args(self.fifo_residuals_cl,
                                                                    self.signalpreprocessor.output_backward_cl,
                                                                    np.int32(self.fifo_roll_size))
            #~ self.peakdetector.initialize_stream()
    
    def apply_processor(self, pos, sigs_chunk):
        if self._plot_debug:
            print('apply_processor')
        assert sigs_chunk.shape[0]==self.chunksize
        
        if not sigs_chunk.flags['C_CONTIGUOUS'] or sigs_chunk.dtype!=self.internal_dtype:
            sigs_chunk = np.ascontiguousarray(sigs_chunk, dtype=self.internal_dtype)
        
        if self.already_processed:
            abs_head_index, preprocessed_chunk =  pos, sigs_chunk
            pyopencl.enqueue_copy(self.queue,  self.sigs_chunk_cl, sigs_chunk)
        else:
            if self.signalpreprocessor.common_ref_removal:
                # because not done in kernel yet
                raise NotImplemenentedError
            
            abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_buffer_stream(pos, sigs_chunk)
        
        # roll fifo
        gsize = self.fifo_roll_size * self.nb_channel
        global_size = (self.fifo_roll_size, self.nb_channel)
        if gsize > self.max_wg_size:
            local_size = (1, self.nb_channel)
        else:
            local_size = global_size
        event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_roll_fifo, global_size, local_size,)
        
        # add new buffer to fifo residuals
        gsize = self.chunksize * self.nb_channel
        global_size = (self.chunksize, self.nb_channel)
        if gsize > self.max_wg_size:
            local_size = (1, self.nb_channel)
        else:
            local_size = global_size
        event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_add_fifo_residuals, global_size, local_size,)
            
            
            
            # TODO call preprocessor kernel and add_fifo_residuals
            
            #~ abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        
        #~ if self._debug_cl:
            #~ import matplotlib.pyplot as plt
            #~ pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.fifo_residuals)
            #~ plt.show()

        return abs_head_index, preprocessed_chunk 

    def detect_local_peaks_before_peeling_loop(self):
        if self._plot_debug:
            print('detect_local_peaks_before_peeling_loop')

        #~ self.global_size =  (self.max_wg_size * n, )
        #~ self.local_size = (self.max_wg_size,)
        
        # reset mask_already_tested
        #~ print('yep', self.mask_already_tested.size, self.mask_already_tested.shape)
        #~ pyopencl.enqueue_fill_buffer(self.queue, self.mask_already_tested_cl, np.zeros(1, dtype='uint8'), 0, self.mask_already_tested.size)
        #~ print('yop')

        self.mask_already_tested[:] = 0
        event = pyopencl.enqueue_copy(self.queue,  self.mask_already_tested_cl, self.mask_already_tested)
        #~ print('yop')
        
        gsize = self.fifo_size - (2 * self.n_span)
        if gsize > self.max_wg_size:
            n = int(np.ceil(gsize / self.max_wg_size))
            global_size =  (self.max_wg_size * n, )
            local_size = (self.max_wg_size,)
        else:
            global_size = (gsize, )
            local_size = (gsize, )
        #~ print('global_size', global_size, 'local_size', local_size)
        
        event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_detect_local_peaks, global_size, local_size,)

        #~ if self._plot_debug:
            #~ import matplotlib.pyplot as plt
            #~ pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
            #~ pyopencl.enqueue_copy(self.queue,  self.nb_pending_peaks, self.nb_pending_peaks_cl)
            #~ pyopencl.enqueue_copy(self.queue,  self.pending_peaks, self.pending_peaks_cl)
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.fifo_residuals)
            #~ plt.show()
        
        
    
    def classify_and_align_next_spike(self):
        if self.use_opencl2:
            # one unique meta kernel
            self.scalar_products[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.scalar_products_cl, self.scalar_products)
            self.distance_shifts[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.distance_shifts_cl, self.distance_shifts)
            self.scalar_products_shifts[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.scalar_products_shifts_cl, self.scalar_products_shifts)

            global_size = (1, )
            local_size = (1, )
            #~ t0 = time.perf_counter()
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_classify_and_align_next_spike, global_size, local_size,)            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_classify_and_align_next_spike',( t1-t0)*1000)

            #~ if True:
            if False:
                event = pyopencl.enqueue_copy(self.queue,  self.nb_pending_peaks, self.nb_pending_peaks_cl)
                event = pyopencl.enqueue_copy(self.queue,  self.pending_peaks, self.pending_peaks_cl)
                event = pyopencl.enqueue_copy(self.queue,  self.next_peak, self.next_peak_cl)
                event = pyopencl.enqueue_copy(self.queue,  self.next_spike, self.next_spike_cl)
                nb = self.nb_pending_peaks[0]
                print(self.pending_peaks[:nb])
                print('self.nex_peak', self.next_peak)
                
                event = pyopencl.enqueue_copy(self.queue,  self.scalar_products, self.scalar_products_cl)
                print(self.scalar_products)
                print('next_spike', self.next_spike)
                
                event = pyopencl.enqueue_copy(self.queue,  self.distance_shifts, self.distance_shifts_cl)
                print(self.distance_shifts)

                #~ event = pyopencl.enqueue_copy(self.queue,  self.best_distance, self.best_distance_cl)
                #~ print(self.best_distance)
                
                print('next_spike', self.next_spike)

            t0 = time.perf_counter()
            event = pyopencl.enqueue_copy(self.queue,  self.next_spike, self.next_spike_cl)
            event.wait()
            
            if self.next_spike[0]['cluster_idx'] >= 0:
                global_size = (self.peak_width, )
                local_size = (self.peak_width, )
                event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_remove_spike_from_fifo, global_size, local_size,)
            
            if self._plot_debug:
                event = pyopencl.enqueue_copy(self.queue,  self.next_peak, self.next_peak_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.best_distance, self.best_distance_cl)
                event.wait()
                print(self.next_peak)
                print(self.next_spike)
                #~ print('self.best_distance', self.best_distance, self.distance_limit)
                print()
            
            cluster_idx = self.next_spike[0]['cluster_idx']
            if self.next_spike[0]['cluster_idx'] >= 0:
                label = self.catalogue['cluster_labels'][cluster_idx]
            else:
                label = cluster_idx
                
            
        else:
            # OpenCL 1.2
        
        
            if self._plot_debug:
                print()
                print('classify_and_align_next_spike')
            
            
            # reset scalar_products and distance_shifts
            self.scalar_products[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.scalar_products_cl, self.scalar_products)
            self.distance_shifts[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.distance_shifts_cl, self.distance_shifts)
            self.scalar_products_shifts[:] = 0
            event = pyopencl.enqueue_copy(self.queue,  self.scalar_products_shifts_cl, self.scalar_products_shifts)
            
            
            #~ print()
            global_size = (1, )
            local_size = (1, )
            #~ t0 = time.perf_counter()
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_select_next_peak, global_size, local_size,)
            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_select_next_peak',( t1-t0)*1000)
            
            #~ if True:
                #~ event = pyopencl.enqueue_copy(self.queue,  self.nb_pending_peaks, self.nb_pending_peaks_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.pending_peaks, self.pending_peaks_cl)
                #~ nb = self.nb_pending_peaks[0]
                #~ print(self.pending_peaks[:nb])
            
            #~ t0 = time.perf_counter()
            global_size, local_size = self.sizes_explore_templates
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_explore_templates, global_size, local_size,)
            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_explore_templates',( t1-t0)*1000)

            #~ if True:
                #~ event = pyopencl.enqueue_copy(self.queue,  self.next_spike, self.next_spike_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.scalar_products, self.scalar_products_cl)
                #~ print(self.next_spike)
                #~ print(self.scalar_products)
            
            global_size, local_size = self.sizes_get_candidate_template
            #~ t0 = time.perf_counter()
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_get_candidate_template, global_size, local_size,)
            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_get_candidate_template', ( t1-t0)*1000)

            #~ if True:
                #~ event = pyopencl.enqueue_copy(self.queue,  self.candidate_template, self.candidate_template_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.nb_candidate, self.nb_candidate_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.nb_flexible_candidate, self.nb_flexible_candidate_cl)
                #~ print(self.candidate_template)
                #~ print(self.nb_candidate, self.nb_flexible_candidate)
            
            global_size = (1, )
            local_size = (1, )
            #~ t0 = time.perf_counter()
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_make_common_mask, global_size, local_size,)
            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_make_common_mask',( t1-t0)*1000)

            #~ if True:
                #~ event = pyopencl.enqueue_copy(self.queue,  self.common_mask, self.common_mask_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.nb_candidate, self.nb_candidate_cl)
                #~ print(self.common_mask)
                #~ print(self.nb_candidate)

            #~ t0 = time.perf_counter()
            (global_size, local_size) = self.sizes_explore_shifts
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_explore_shifts, global_size, local_size,)
            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_explore_shifts', ( t1-t0)*1000)        

            #~ if True:
                #~ event = pyopencl.enqueue_copy(self.queue,  self.scalar_products_shifts, self.scalar_products_shifts_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.distance_shifts, self.distance_shifts_cl)
                #~ print(self.scalar_products_shifts)
                #~ print(self.distance_shifts)
            

            global_size = (1, )
            local_size = (1, )
            #~ t0 = time.perf_counter()
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_best_shift_and_jitter, global_size, local_size,)
            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_best_shift_and_jitter',( t1-t0)*1000)
            
            #~ if True:
                #~ event = pyopencl.enqueue_copy(self.queue,  self.next_spike, self.next_spike_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.final_scalar_product, self.final_scalar_product_cl)
                
                #~ event.wait()
                #~ print('self.next_spike', self.next_spike, 'self.final_scalar_product', self.final_scalar_product)
            
            
            global_size = (1, )
            local_size = (1, )
            #~ t0 = time.perf_counter()
            event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_finalize_next_spike, global_size, local_size,)
            
            #~ event.wait()
            #~ t1 = time.perf_counter()
            #~ print('kern_finalize_next_spike',( t1-t0)*1000)        
            
            
            #~ if True:
                #~ event = pyopencl.enqueue_copy(self.queue,  self.next_spike, self.next_spike_cl)
                #~ event.wait()
                #~ print('self.next_spike', self.next_spike)
            
            
            
            
            
            #~ t0 = time.perf_counter()
            event = pyopencl.enqueue_copy(self.queue,  self.next_spike, self.next_spike_cl)
            event.wait()
            
            #~ t1 = time.perf_counter()
            #~ print('enqueue_copy',( t1-t0)*1000)        
            #~ print(' self.next_spike',  self.next_spike)
            #~ print()
            
            if self.next_spike[0]['cluster_idx'] >= 0:
                global_size = (self.peak_width_long, )
                local_size = (self.peak_width_long, )
                #~ t0 = time.perf_counter()
                event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_remove_spike_from_fifo, global_size, local_size,)
                
                #~ event.wait()
                #~ t1 = time.perf_counter()
                #~ print('kern_remove_spike_from_fifo',( t1-t0)*1000)        
            
            
            
            if self._plot_debug:
            #~ if True:
                event = pyopencl.enqueue_copy(self.queue,  self.next_peak, self.next_peak_cl)
                #~ event = pyopencl.enqueue_copy(self.queue,  self.best_distance, self.best_distance_cl)
                event.wait()
                print(self.next_peak)
                print(self.next_spike)
                #~ print('self.best_distance', self.best_distance, self.distance_limit)
                print()
            
            
            cluster_idx = self.next_spike[0]['cluster_idx']
            if self.next_spike[0]['cluster_idx'] >= 0:
                label = self.catalogue['cluster_labels'][cluster_idx]
            else:
                label = cluster_idx
        
        
        if self._plot_debug:
            print(Spike(self.next_spike[0]['sample_ind'], label, self.next_spike[0]['jitter']))
        return Spike(self.next_spike[0]['sample_ind'], label, self.next_spike[0]['jitter'])
        
        
    def reset_to_not_tested(self, good_spikes):
        #~ print('reset_to_not_tested')
        #~ print(good_spikes)
        
        n = len(good_spikes)
        global_size = (n, )
        if n > self.max_wg_size:
            local_size = (1, ) # otherwise too much complicated
        else:
            local_size = global_size
        event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_reset_tested_zone, global_size, local_size,)
        
        # mask_already_tested must NOT be reset here!!!

        gsize = self.fifo_size - (2 * self.n_span)
        if gsize > self.max_wg_size:
            n = int(np.ceil(gsize / self.max_wg_size))
            global_size =  (self.max_wg_size * n, )
            local_size = (self.max_wg_size,)
        else:
            global_size = (gsize, )
            local_size = (gsize, )
        #~ print('global_size', global_size, 'local_size', local_size)
        
        event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_detect_local_peaks, global_size, local_size,)
    
    
    def get_no_label_peaks(self):
        raise NotImplementedError
        # TODO
        #~ gsize = self.fifo_size - (2 * self.n_span)
        #~ if gsize > self.max_wg_size:
            #~ n = int(np.ceil(gsize / self.max_wg_size))
            #~ global_size =  (self.max_wg_size * n, )
            #~ local_size = (self.max_wg_size,)
        #~ else:
            #~ global_size = (gsize, )
            #~ local_size = (gsize, )
        
        #~ event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_detect_local_peaks, global_size, local_size,)
        
        #~ nolabel_indexes += self.n_span
        #~ nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        #~ bad_spikes = np.zeros(0, dtype=_dtype_spike)
        #~ bad_spikes['index'] = nolabel_indexes
        #~ bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        #~ return bad_spikes
        
    
    def _plot_before_peeling_loop(self):
        import matplotlib.pyplot as plt
        pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
        self._plot_sigs_before = self.fifo_residuals.copy()

        pyopencl.enqueue_copy(self.queue,  self.nb_pending_peaks, self.nb_pending_peaks_cl)
        pyopencl.enqueue_copy(self.queue,  self.pending_peaks, self.pending_peaks_cl)
        pending_peaks =  self.pending_peaks[:self.nb_pending_peaks[0]]
        sample_inds = pending_peaks['sample_ind']
        chan_inds = pending_peaks['chan_index']
        print(pending_peaks)

        fig, ax = plt.subplots()
        
        plot_sigs = self.fifo_residuals.copy()
        self._plot_sigs_before = plot_sigs
        
        for c in range(self.nb_channel):
            plot_sigs[:, c] += c*30
        
        ax.plot(plot_sigs, color='k')

        ax.axvline(self.fifo_size - self.n_right, color='r')
        ax.axvline(-self.n_left, color='r')

        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        
        #~ plt.show()
    
    def _plot_after_peeling_loop(self, good_spikes):
        import matplotlib.pyplot as plt
        pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
        
        self.mask_already_tested[:] = 0
        event = pyopencl.enqueue_copy(self.queue,  self.mask_already_tested_cl, self.mask_already_tested)

        gsize = self.fifo_size - (2 * self.n_span)
        if gsize > self.max_wg_size:
            n = int(np.ceil(gsize / self.max_wg_size))
            global_size =  (self.max_wg_size * n, )
            local_size = (self.max_wg_size,)
        else:
            global_size = (gsize, )
            local_size = (gsize, )
        event = pyopencl.enqueue_nd_range_kernel(self.queue,  self.kern_detect_local_peaks, global_size, local_size,)
        pyopencl.enqueue_copy(self.queue,  self.nb_pending_peaks, self.nb_pending_peaks_cl)
        pyopencl.enqueue_copy(self.queue,  self.pending_peaks, self.pending_peaks_cl)
        pending_peaks =  self.pending_peaks[:self.nb_pending_peaks[0]]
        print('ici')
        print(pending_peaks)
        keep = pending_peaks['peak_value'] > 0
        pending_peaks = pending_peaks[keep]
        sample_inds = pending_peaks['sample_ind']
        chan_inds = pending_peaks['chan_index']
        
        fig, ax = plt.subplots()
        #~ plot_sigs = self.fifo_residuals.copy()
        plot_sigs = self._plot_sigs_before
        
        
        #~ for c in range(self.nb_channel):
            #~ plot_sigs[:, c] += c*30
        #~ ax.plot(plot_sigs, color='k')best_shift_and_jitter
        
        ax.plot(self._plot_sigs_before, color='b')
        
        ax.axvline(self.fifo_size - self.n_right_long, color='r')
        ax.axvline(-self.n_left_long, color='r')

        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        
        

        plot_res = self.fifo_residuals.copy()
        for c in range(self.nb_channel):
            plot_res[:, c] += c*30
        ax.plot(plot_res, color='k', alpha=0.3)
        
        good_spikes = np.array(good_spikes, dtype=_dtype_spike)
        pred = make_prediction_signals(good_spikes, self.internal_dtype, plot_sigs.shape, self.catalogue, safe=True)
        plot_pred = pred.copy()
        for c in range(self.nb_channel):
            plot_pred[:, c] += c*30
        ax.plot(plot_pred, color='m')
        
        
        
        
        plt.show()

        
    def _plot_after_inner_peeling_loop(self):
        return
        
        import matplotlib.pyplot as plt
        
        #~ pyopencl.enqueue_copy(self.queue,  self.nb_pending_peaks, self.nb_pending_peaks_cl)
        #~ pyopencl.enqueue_copy(self.queue,  self.pending_peaks, self.pending_peaks_cl)
        pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)



        
        #~ self._plot_sigs_before = plot_sigs
        #~ chan_order = np.argsort(self.channel_distances[0, :])
        
        fig, ax = plt.subplots()
        
        plot_sigs = self._plot_sigs_before.copy()
        
        for c in range(self.nb_channel):
            plot_sigs[:, c] += c*30
        ax.plot(plot_sigs, color='b')
        
        plot_sigs = self.fifo_residuals.copy()
        
        
        for c in range(self.nb_channel):
            plot_sigs[:, c] += c*30
        
        ax.plot(plot_sigs, color='k')

        ax.axvline(self.fifo_size - self.n_right, color='r')
        ax.axvline(-self.n_left, color='r')

        #~ mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        #~ sample_inds, chan_inds= np.nonzero(mask)
        #~ sample_inds += self.n_span
        
        #~ ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        #~ ax.plot(self.fifo_residuals)
        plt.show()
        
        self._plot_sigs_before = self.fifo_residuals.copy()




kernel_peeler_cl = """
#define chunksize %(chunksize)d
#define n_span %(n_span)d
#define nb_channel %(nb_channel)d
#define nb_cluster %(nb_cluster)d
#define relative_threshold %(relative_threshold)f
#define peak_sign %(peak_sign)d
#define extra_size %(extra_size)d
#define fifo_size %(fifo_size)d
#define n_left %(n_left)d
#define n_right %(n_right)d
#define n_left_long %(n_left_long)d
#define n_right_long %(n_right_long)d
#define peak_width %(peak_width)d
#define peak_width_long %(peak_width_long)d
#define maximum_jitter_shift %(maximum_jitter_shift)d
#define n_cluster %(n_cluster)d
#define wf_size %(wf_size)d
#define wf_size_long %(wf_size_long)d
#define subsample_ratio %(subsample_ratio)d
#define nb_neighbour %(nb_neighbour)d
#define inter_sample_oversampling %(inter_sample_oversampling)d
#define nb_shift %(nb_shift)d



#define LABEL_LEFT_LIMIT -11
#define LABEL_RIGHT_LIMIT -12
#define LABEL_MAXIMUM_SHIFT -13
#define LABEL_TRASH -1
#define LABEL_NOISE -2
#define LABEL_ALIEN -9
#define LABEL_UNCLASSIFIED -10
#define LABEL_NO_WAVEFORM -11
#define LABEL_NOT_YET 0
#define LABEL_NO_MORE_PEAK -20


typedef struct st_spike{
    int sample_index;
    int cluster_idx;
    float jitter;
} st_spike;

typedef struct st_peak{
    int sample_index;
    int chan_index;
    float peak_value;
} st_peak;

typedef struct st_candidate{
    uchar strict;
    uchar flexible;
} st_candidate;






__kernel void roll_fifo(__global  float *fifo_residuals, int fifo_roll_size){
    int pos = get_global_id(0);
    int chan = get_global_id(1);
    
    if (pos>=fifo_roll_size){
        return;
    }
    if (chan>=nb_channel){
        return;
    }

    fifo_residuals[pos*nb_channel+chan] = fifo_residuals[(pos+chunksize)*nb_channel+chan];

}

__kernel void add_fifo_residuals(__global  float *fifo_residuals, __global  float *sigs_chunk, int fifo_roll_size){
    int pos = get_global_id(0);
    int chan = get_global_id(1);

    if (pos>=chunksize){
        return;
    }
    if (chan>=nb_channel){
        return;
    }
    
    fifo_residuals[(pos+fifo_roll_size)*nb_channel+chan] = sigs_chunk[pos*nb_channel+chan];
}


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



__kernel void detect_local_peaks(
                        __global  float *sigs,
                        __global  int *neighbours,
                        __global  uchar *mask_already_tested,
                        __global  st_peak *pending_peaks,
                        volatile __global int *nb_pending_peaks
                ){
    int pos = get_global_id(0);
    

    
    if (pos == 0){
        *nb_pending_peaks = 0;
    }
    // this barrier OK if the first group is run first
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (pos>=(fifo_size - (2 * n_span))){
        return;
    }
    

    float v;
    uchar peak;
    int chan_neigh;
    
    int i_peak;

    
    for (int chan=0; chan<nb_channel; chan++){
    
        v = sigs[(pos + n_span)*nb_channel + chan];

    
        if(peak_sign==1){
            if (v>relative_threshold){peak=1;}
            else if (v<=relative_threshold){peak=0;}
        }
        else if(peak_sign==-1){
            if (v<-relative_threshold){peak=1;}
            else if (v>=-relative_threshold){peak=0;}
        }
        
        // avoid peak already tested
        if (mask_already_tested[(pos + n_span)*nb_channel + chan] == 1){
            peak = 0;
        }
        
        if (peak == 1){
            for (int neighbour=0; neighbour<nb_neighbour; neighbour++){
                // when neighbour then chan==chan_neigh (itself)
                chan_neigh = neighbours[chan * nb_neighbour +neighbour];
                
                if (chan_neigh<0){continue;}
                
                if (chan != chan_neigh){
                    if(peak_sign==1){
                        peak = peak && (v>=sigs[(pos + n_span)*nb_channel + chan_neigh]);
                    }
                    else if(peak_sign==-1){
                        peak = peak && (v<=sigs[(pos + n_span)*nb_channel + chan_neigh]);
                    }
                }
                
                if (peak==0){break;}
                
                if(peak_sign==1){
                    for (int i=1; i<=n_span; i++){
                        peak = peak && (v>sigs[(pos + n_span - i)*nb_channel + chan_neigh]) && (v>=sigs[(pos + n_span + i)*nb_channel + chan_neigh]);
                        if (peak==0){break;}
                    }
                }
                else if(peak_sign==-1){
                    for (int i=1; i<=n_span; i++){
                        peak = peak && (v<sigs[(pos + n_span - i)*nb_channel + chan_neigh]) && (v<=sigs[(pos + n_span + i)*nb_channel + chan_neigh]);
                        if (peak==0){break;}
                    }
                }
            
            }
            
        }
        
        if (peak==1){
            //append to 
            i_peak = atomic_inc(nb_pending_peaks);
            // sample_index is LOCAL to fifo
            pending_peaks[i_peak].sample_index = pos + n_span;
            pending_peaks[i_peak].chan_index = chan;
            pending_peaks[i_peak].peak_value = fabs(v);
        }
    }
    
}



__kernel void select_next_peak(
                        __global float *fifo_residuals,
                        __global st_peak *next_peak,
                        __global  st_peak *pending_peaks,
                        __global int *nb_pending_peaks,
                        __global st_spike *next_spike,
                        float alien_value_threshold
                        ){
    
    // take max amplitude

    st_spike spike;
    st_peak peak;
    
    int i_peak = -1;
    float best_value = 0.0;
    
    int n = *nb_pending_peaks;
    for (int i=0; i<n; i++){
        if ((pending_peaks[i].peak_value > best_value)){
            i_peak = i;
            best_value = pending_peaks[i].peak_value;
        }
    }
    
    if (i_peak == -1){
        peak.sample_index = LABEL_NO_MORE_PEAK;
        peak.chan_index = -1;
        peak.peak_value = 0.0;
        *next_peak = peak;
        
        spike.sample_index = LABEL_NO_MORE_PEAK;
        spike.cluster_idx = LABEL_NO_MORE_PEAK;
        spike.jitter = 0.0f;
    }else{
        
        st_peak peak;
        
        peak = pending_peaks[i_peak];
        *next_peak = peak;

        // make this peak not selectable for next choice
        pending_peaks[i_peak].peak_value = 0.0;
        
        
        int left_ind_long = peak.sample_index + n_left_long;
        
        if (left_ind_long+peak_width_long+maximum_jitter_shift+1>=fifo_size){
            spike.sample_index = peak.sample_index;
            spike.cluster_idx = LABEL_RIGHT_LIMIT;
            spike.jitter = 0.0f;
        } else if (left_ind_long<=maximum_jitter_shift){
            spike.sample_index = peak.sample_index;
            spike.cluster_idx = LABEL_LEFT_LIMIT;
            spike.jitter = 0.0f;
        } else if (n_cluster==0){
            spike.sample_index = peak.sample_index;
            spike.cluster_idx  = LABEL_UNCLASSIFIED;
            spike.jitter = 0.0f;
        }else {
            
            
            spike.sample_index = peak.sample_index;
            spike.cluster_idx = LABEL_NOT_YET;
            spike.jitter = 0.0f;
            if (alien_value_threshold>0.0f){
                int left_ind = peak.sample_index + n_left;
                for (int s=0; s<peak_width; ++s){
                    if ((fifo_residuals[(left_ind+s)*nb_channel + peak.chan_index]>alien_value_threshold) || (fifo_residuals[(left_ind+s)*nb_channel + peak.chan_index]<-alien_value_threshold)){
                        spike.cluster_idx = LABEL_ALIEN;
                    }
                }
            }
        }    
    }
    
    *next_spike = spike;

}



__kernel void explore_templates(__global  float *fifo_residuals,
                                                                __global st_spike *next_spike,
                                                                __global  st_peak *next_peak,

                                                                __global  float *catalogue_center0,
                                                                
                                                                __global  uchar  *sparse_mask_level1,
                                                                __global  float *scalar_products,
                                                                __global  float *projections

                                                                ){

    int cluster_idx = get_global_id(0);
    int chan = get_global_id(1);

    if (cluster_idx>=nb_cluster){return;}
    if (chan>=nb_channel){return;}    
    
    // each worker
    int left_ind;
    st_spike spike;
    st_peak peak;
    
    spike = *next_spike;
    
    if (spike.cluster_idx!=LABEL_NOT_YET) {return;}
        
    peak = *next_peak;

    // compute scalar product
    if (sparse_mask_level1[nb_channel*cluster_idx+peak.chan_index] == 1){
        if (sparse_mask_level1[nb_channel*cluster_idx+chan] == 1){
            left_ind = spike.sample_index + n_left;
            float sum = 0;
            float ct, v, w;
            for (int s=0; s<peak_width; ++s){
                v = fifo_residuals[(left_ind+s)*nb_channel + chan];
                ct = catalogue_center0[wf_size*cluster_idx+nb_channel*s+chan];
                w = projections[wf_size*cluster_idx+nb_channel*s+chan];
                sum += (v - ct) * w;
            }
            atomic_add_float(&scalar_products[cluster_idx], sum);
        }
    }else{
        if (chan==0){
            scalar_products[cluster_idx] = FLT_MAX;
        }
    }

}


__kernel void get_candidate_template(
                                            __global st_spike *next_spike,
                                            __global  float *scalar_products,
                                            __global  float *boundaries,
                                            __global  st_candidate *candidate_template,
                                            volatile __global int *nb_candidate,
                                            volatile __global int *nb_flexible_candidate
                                        ){
    
    int cluster_idx = get_global_id(0);
    
    if (cluster_idx>=nb_cluster){return;}

    if (cluster_idx == 0){
        *nb_candidate = 0;
        *nb_flexible_candidate = 0;
    }
    // this barrier OK if the first group is run first
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    
    st_spike spike;
    spike = *next_spike;

    
    if (spike.cluster_idx==LABEL_NOT_YET) {
        
        float sp = scalar_products[cluster_idx];
        if ((sp > boundaries[cluster_idx*4 + 0]) && (sp < boundaries[cluster_idx*4 + 1])){
            candidate_template[cluster_idx].strict = 1;
            atomic_inc(nb_candidate);
        }else{
            candidate_template[cluster_idx].strict = 0;
        }
        if ((sp > boundaries[cluster_idx*4 + 2]) && (sp < boundaries[cluster_idx*4 + 3])){
            candidate_template[cluster_idx].flexible = 1;
            atomic_inc(nb_flexible_candidate);
        } else{
            candidate_template[cluster_idx].flexible = 0;
        }
    } else{
        candidate_template[cluster_idx].strict = 0;
        candidate_template[cluster_idx].flexible = 0;
    }
}


__kernel void make_common_mask(
                                            __global st_spike *next_spike,
                                            __global  st_candidate *candidate_template,
                                            __global int *nb_candidate,
                                            __global int *nb_flexible_candidate,
                                            __global  uchar *sparse_mask_level3,
                                            __global  uchar *common_mask
                                        ){
    st_spike spike;
    spike = *next_spike;
    
    if (spike.cluster_idx!=LABEL_NOT_YET) {return;}
    
    for (int chan=0; chan<nb_channel; chan++){
        common_mask[chan] = 0;
    }
    
    if (*nb_candidate >= 1){
        for (int cluster_idx=0; cluster_idx<nb_cluster; cluster_idx++){
            if (candidate_template[cluster_idx].strict){
                for (int chan=0; chan<nb_channel; chan++){
                    if (sparse_mask_level3[cluster_idx*nb_channel+chan]){
                        common_mask[chan] = 1;
                    }
                }
            }
        }
    }else if (*nb_flexible_candidate >= 1){
        // explore all flexible
        
        *nb_candidate = *nb_flexible_candidate;
        
        for (int cluster_idx=0; cluster_idx<nb_cluster; cluster_idx++){
            // put to strict to be explored anyway
            if (candidate_template[cluster_idx].flexible){
                candidate_template[cluster_idx].strict = 1;
            }
            if (candidate_template[cluster_idx].strict){
                for (int chan=0; chan<nb_channel; chan++){
                    if (sparse_mask_level3[cluster_idx*nb_channel+chan]){
                        common_mask[chan] = 1;
                    }
                }
            }
        }
    }else{
        // no candidate
        spike.cluster_idx  = LABEL_UNCLASSIFIED;
        *next_spike = spike;
    }
}


__kernel void explore_shifts(__global  float *fifo_residuals,
                                            __global st_spike *next_spike,
                                            __global  st_peak *next_peak,

                                            __global  float *catalogue_center0,
                                            __global  uchar  *sparse_mask_level1,
                                            __global  float *projections,
                                            __global  st_candidate *candidate_template,
                                            __global  float *scalar_products_shifts,
                                            __global  float *distance_shifts,
                                            __global  uchar *common_mask
                                        ){


    int cluster_idx = get_global_id(0);
    int shift_ind = get_global_id(1);
    int chan = get_global_id(2);
    
    if (cluster_idx>=nb_cluster){return;}
    if (shift_ind>=nb_shift){return;}
    if (chan>=nb_channel){return;}
    
    
    
    st_spike spike;
    spike = *next_spike;
    
    if (spike.cluster_idx!=LABEL_NOT_YET) {return;}
    
    
    int left_ind;
    left_ind = spike.sample_index + n_left;
    
    
    if (candidate_template[cluster_idx].strict == 0){
        if (chan ==0){
            int ind;
            ind = cluster_idx*nb_shift + shift_ind;
            // maybe this 2 assignement are useless TODO ?
            distance_shifts[ind] = MAXFLOAT;
            scalar_products_shifts[ind] = MAXFLOAT;
        }
        return;
    }

    if (common_mask[chan] || (sparse_mask_level1[nb_channel*cluster_idx + chan] == 1)){
        int shift;
        shift = shift_ind - maximum_jitter_shift;
        float sum_d = 0;
        float sum_sp = 0;
        float v, w, ct;
        
        for (int s=0; s<peak_width; ++s){
            v = fifo_residuals[(left_ind+s+shift)*nb_channel + chan];
            ct = catalogue_center0[wf_size*cluster_idx+nb_channel*s+chan];
            w = projections[wf_size*cluster_idx+nb_channel*s+chan];
            if (common_mask[chan]){
                sum_d += (v - ct) * (v - ct);
            }
            if (sparse_mask_level1[nb_channel*cluster_idx + chan] == 1){
                sum_sp += (v - ct) * w;
            }
        }
        
        int ind;
        ind = cluster_idx*nb_shift + shift_ind;
        atomic_add_float(&distance_shifts[ind], sum_d);
        atomic_add_float(&scalar_products_shifts[ind], sum_sp);
    }
}


float estimate_one_jitter(int left_ind, int cluster_idx,
                                        __global int *extremum_channel,
                                        __global float *fifo_residuals,
                                        
                                        __global float *catalogue_center0,
                                        __global float *catalogue_center1,
                                        __global float *catalogue_center2,
                                        
                                        __global float *wf1_norm2,
                                        __global float *wf2_norm2,
                                        __global float *wf1_dot_wf2){

    int chan_max;
    float jitter;
        
    chan_max= extremum_channel[cluster_idx];
    
    
    float h, wf1, wf2;
    float h0_norm2 = 0.0f;
    float h_dot_wf1 =0.0f;
    
    for (int s=0; s<peak_width; ++s){
        h = fifo_residuals[(left_ind+s)*nb_channel + chan_max] - catalogue_center0[wf_size*cluster_idx+nb_channel*s+chan_max];
        h0_norm2 +=  h*h;
        wf1 = catalogue_center1[wf_size*cluster_idx+nb_channel*s+chan_max];
        h_dot_wf1 += h * wf1;
        
    }
    
    jitter = h_dot_wf1/wf1_norm2[cluster_idx];
    
    float h1_norm2;
    h1_norm2 = 0.0f;
    for (int s=0; s<peak_width; ++s){
        h = fifo_residuals[(left_ind+s)*nb_channel + chan_max] - catalogue_center0[wf_size*cluster_idx+nb_channel*s+chan_max];
        wf1 = catalogue_center1[wf_size*cluster_idx+nb_channel*s+chan_max];
        h1_norm2 += (h - jitter * wf1) * (h - jitter * wf1);
    }
    
    if (h0_norm2 > h1_norm2){
        float h_dot_wf2 =0.0f;
        float rss_first, rss_second;
        for (int s=0; s<peak_width; ++s){
            h = fifo_residuals[(left_ind+s)*nb_channel + chan_max] - catalogue_center0[wf_size*cluster_idx+nb_channel*s+chan_max];
            wf2 = catalogue_center2[wf_size*cluster_idx+nb_channel*s+chan_max];
            h_dot_wf2 += h * wf2;
        }
        rss_first = -2*h_dot_wf1 + 2*jitter*(wf1_norm2[cluster_idx] - h_dot_wf2) + pown(3*jitter, 2)*wf1_dot_wf2[cluster_idx] + pown(jitter,3)*wf2_norm2[cluster_idx];
        rss_second = 2*(wf1_norm2[cluster_idx] - h_dot_wf2) + 6*jitter*wf1_dot_wf2[cluster_idx] + 3*pown(jitter, 2) * wf2_norm2[cluster_idx];
        jitter = jitter - rss_first/rss_second;
    } else{
        jitter = 0.0f;
    }
    
    return jitter;
}


__kernel void best_shift_and_jitter(
                                            __global  float *fifo_residuals,
                                            __global st_spike *next_spike,
                                            __global  st_peak *next_peak,
                                            
                                            __global  float *catalogue_center0,
                                            __global  float *catalogue_center1,
                                            __global  float *catalogue_center2,
                                            __global  float *catalogue_inter_center0,
                                            __global int *extremum_channel,
                                            __global float *wf1_norm2,
                                            __global float *wf2_norm2,
                                            __global float *wf1_dot_wf2,
                                            
                                            
                                            __global int *nb_candidate,
                                            __global  st_candidate *candidate_template,
                                            __global  float *scalar_products_shifts,
                                            __global  float *distance_shifts,
                                            __global  float *final_scalar_product
                                            
                                            ){
    
    st_spike spike;
    spike = *next_spike;
    
    if ((spike.cluster_idx!=LABEL_NOT_YET)) {return ;}
    
    if (*nb_candidate==0){
        spike.cluster_idx  = LABEL_UNCLASSIFIED;
        return ;
    }
    
    int left_ind;
    left_ind = spike.sample_index + n_left;
    
    // argmin for shifts
    int best_cluster_idx = -1;
    int best_shift = 0;
    float min_dist = MAXFLOAT;
    float best_sp=MAXFLOAT;
    
    for (int cluster_idx=0; cluster_idx<nb_cluster; cluster_idx++){
        if (candidate_template[cluster_idx].strict == 1){
            for (int shift_ind=0; shift_ind<nb_shift; ++shift_ind){
                if (distance_shifts[nb_shift*cluster_idx+shift_ind]<min_dist){
                    best_cluster_idx = cluster_idx;
                    best_shift = shift_ind - maximum_jitter_shift;
                    min_dist = distance_shifts[nb_shift*cluster_idx+shift_ind];
                    best_sp = scalar_products_shifts[nb_shift*cluster_idx+shift_ind];
                }
            }
        }
    }
    
    
    left_ind = left_ind + best_shift;
    
    spike.cluster_idx = best_cluster_idx;
    spike.sample_index = left_ind - n_left;
    *final_scalar_product = best_sp;
    
    
    
    // jitter
    float jitter, new_jitter;
    if (inter_sample_oversampling){
        int shift;
        
        jitter = estimate_one_jitter(left_ind, spike.cluster_idx,
                                extremum_channel, fifo_residuals,
                                catalogue_center0, catalogue_center1, catalogue_center2,
                                wf1_norm2, wf2_norm2, wf1_dot_wf2);
        
        // try better jitter
        if (inter_sample_oversampling && (fabs(jitter) > 0.5f) && ((left_ind+shift+peak_width)<fifo_size) && ((left_ind + shift) >= 0) ){
            shift = - ((int) round(jitter));
            new_jitter = estimate_one_jitter(left_ind+shift, spike.cluster_idx,
                                                extremum_channel, fifo_residuals,
                                                catalogue_center0, catalogue_center1, catalogue_center2,
                                                wf1_norm2, wf2_norm2, wf1_dot_wf2);
            if (fabs(new_jitter)<fabs(jitter)){
                jitter = new_jitter;
                left_ind = left_ind + shift;
            }
        }
        
        // security to not be outside the fifo
        int left_ind_long = spike.sample_index + n_left_long;
        
        shift = - ((int) round(jitter));
        if (abs(shift) >maximum_jitter_shift){
            spike.cluster_idx = LABEL_MAXIMUM_SHIFT;
        } else if ((left_ind_long+shift+peak_width_long) >= fifo_size){
            spike.cluster_idx = LABEL_RIGHT_LIMIT;
        } else if ((left_ind_long+shift)<0) {
            spike.cluster_idx = LABEL_LEFT_LIMIT;
        }
        
    } else {
        jitter = 0.0f;
    }
    spike.jitter = jitter;


    *next_spike = spike ;

}


int accept_tempate(int left_ind, int cluster_idx, float jitter,
                                        __global float *fifo_residuals,

                                        __global  float *boundaries,
                                        __global float *final_scalar_product

                                        ){


    if (fabs(jitter) > (maximum_jitter_shift - 0.5)){
        return 0;
    }
    
    float sp = *final_scalar_product;
    
    // flexible limit
    if ((sp >boundaries[cluster_idx*4 + 2]) && (sp <boundaries[cluster_idx*4 + 3])){
        return 1;
    }else{
        return 0;
    }
    
    //float v;
    //float pred;
    
    //float chan_in_mask = 0.0f;
    //float chan_with_criteria = 0.0f;
    //int idx;
    
    //float w;
    
    //int int_jitter = (int) ((jitter) * subsample_ratio) + (subsample_ratio / 2 );
    
    //for (int c=0; c<nb_channel; ++c){
    //    if (sparse_mask_level2[cluster_idx*nb_channel + c] == 1){

    //        float wf_nrj = 0.0f;
    //        float res_nrj = 0.0f;
    //    
    //        for (int s=0; s<peak_width; ++s){
    //            v = fifo_residuals[(left_ind+s)*nb_channel + c];
    //            wf_nrj += (v*v);
                
    //            //idx = wf_size*cluster_idx+nb_channel*s+c;
    //            //pred = catalogue_center0[idx] + jitter*catalogue_center1[idx] + jitter*jitter/2*catalogue_center2[idx];
    //            if (inter_sample_oversampling){
    //                pred = catalogue_inter_center0[subsample_ratio*wf_size*cluster_idx + nb_channel*(s*subsample_ratio+int_jitter) + c];
    //            }else{
    //                pred = catalogue_center0[wf_size*cluster_idx + nb_channel*s + c];
    //            }
                
                
    //            v -= pred;
    //            res_nrj += (v*v);
    //        }
            
    //        w = weight_per_template[cluster_idx*nb_channel + c];
    //        chan_in_mask += w;
    //        if (wf_nrj>res_nrj){
    //            chan_with_criteria += w;
    //        }
    //    }
    //}
    
    //if (chan_with_criteria>=(chan_in_mask*0.7f)){
    //    return 1;
    //}else{
    //    return 0;
    //}

}

__kernel void finalize_next_spike(
                                            __global  float *fifo_residuals,
                                            __global st_spike *next_spike,
                                            __global  st_peak *next_peak,
                                            __global  st_peak *pending_peaks,
                                            __global int *nb_pending_peaks,
                                            __global  st_spike *good_spikes,
                                            __global int *nb_good_spikes,
                                            __global  uchar *mask_already_tested,
                                            __global  float *catalogue_center0,
                                            __global  float *catalogue_center1,
                                            __global  float *catalogue_center2,
                                            __global  float *catalogue_inter_center0,
                                            __global  uchar  *sparse_mask_level1,
                                            __global  uchar  *sparse_mask_level2,
                                            __global  float *distance_shifts,
                                            __global int *extremum_channel,
                                            __global float *wf1_norm2,
                                            __global float *wf2_norm2,
                                            __global float *wf1_dot_wf2,
                                            __global  float *boundaries,
                                            __global float *final_scalar_product
                                            
                                            
                                            
                                            ){
    
    st_spike spike;
    spike = *next_spike;
    
    int left_ind;
    
    if (spike.cluster_idx>=0) {

        
        left_ind = spike.sample_index + n_left;    
        
        int ok;
        ok = accept_tempate(left_ind, spike.cluster_idx, spike.jitter,
                                    fifo_residuals, boundaries, final_scalar_product);
        
        if (ok == 0){
            spike.jitter = 0.0f;
            spike.cluster_idx = LABEL_UNCLASSIFIED;
        }
    }

    // second security check for borders
    // TODO remove this!!!!!
    if (inter_sample_oversampling){
        if (spike.cluster_idx >= 0){
            int left_ind_check;
            left_ind_check = left_ind - ((int) round(spike.jitter));
            if (left_ind_check < 0){
                spike.cluster_idx = LABEL_LEFT_LIMIT;
            } else if ((left_ind_check+peak_width) >= fifo_size){
                spike.cluster_idx = LABEL_RIGHT_LIMIT;
            }
        }    
    }
    
    // final
    if (spike.cluster_idx < 0){
        st_peak peak;
        peak = *next_peak;
        spike.sample_index = peak.sample_index;
        
        // set already tested
        mask_already_tested[peak.sample_index * nb_channel + peak.chan_index] =1;
        
    } else{
        if (inter_sample_oversampling){
            //TODO remove this!!!!!
            int shift;
            shift = - ((int) round(spike.jitter));
            if (shift !=0){
                spike.jitter +=  shift;
                left_ind += shift;
                spike.sample_index = left_ind - n_left;
            }
        }
        
        // on_accepted_spike
        
        // set_already_tested_spike_zone = remove spike in zone in pendings peaks
        // peaks will be tested after new peak detection in further loop
        int n;
        n = *nb_pending_peaks;
        for (int i=0; i<n; i++){
            if (pending_peaks[i].peak_value > 0.0){
                if (
                        (pending_peaks[i].sample_index  > (spike.sample_index+ n_left)) && 
                        (pending_peaks[i].sample_index < (spike.sample_index+ n_right)) && 
                        (sparse_mask_level1[spike.cluster_idx * nb_channel + pending_peaks[i].chan_index] == 1) ){
                    pending_peaks[i].peak_value = 0;
                }
            }
        }
        
        //add spike to good_spike stack
        good_spikes[*nb_good_spikes] = spike;
        *nb_good_spikes = *nb_good_spikes + 1;
        
        
    }
    
    *next_spike = spike ;
    
}


__kernel void remove_spike_from_fifo(
                                            __global  float *fifo_residuals,
                                            __global st_spike *next_spike,
                                            __global  float *catalogue_center0_long,
                                            __global  float *catalogue_inter_center0
                                            ){
                                            
    
    int s = get_global_id(0);
    
    if (s>=peak_width_long){
        return;
    }
    
    
    st_spike spike;
    spike = *next_spike;
    int left_ind_long;
    left_ind_long = spike.sample_index + n_left_long;
    
    // on_accepted_spike
    float v;
    
    if (inter_sample_oversampling){
        int int_jitter;
        
        int_jitter = (int) ((spike.jitter) * subsample_ratio) + (subsample_ratio / 2 );
        
        for (int c=0; c<nb_channel; ++c){
            v = catalogue_inter_center0[subsample_ratio*wf_size*spike.cluster_idx + nb_channel*(s*subsample_ratio+int_jitter) + c];
            fifo_residuals[(left_ind_long+s)*nb_channel + c] -= v;
        }
    } else{
        for (int c=0; c<nb_channel; ++c){
            v = catalogue_center0_long[wf_size_long*spike.cluster_idx + nb_channel*s + c];
            fifo_residuals[(left_ind_long+s)*nb_channel + c] -= v;
        }
    }

    
    
}



__kernel void reset_tested_zone(__global  uchar *mask_already_tested,
                __global st_spike *good_spikes,
                __global int *nb_good_spikes,
                __global uchar *sparse_mask_level1
                ){
    
    int n = get_global_id(0);
    
    st_spike spike = good_spikes[n];
    
    // reset tested zone around each spike
    for (int c=0; c<nb_channel; ++c){
        if (sparse_mask_level1[spike.cluster_idx * nb_channel + c]==1){
            
            
            //for (int s=-peak_width; s<=peak_width; ++s){
            // TODO study this case 
            for (int s=n_left; s<=n_right; ++s){
                mask_already_tested[(spike.sample_index + s) * nb_channel + c] = 0;
            }
        }
    }
    
    if (n == 0){
        *nb_good_spikes = 0;
    }
}





"""



kernel_opencl2_extention = """

#define ls_0 %(ls_0)d


__kernel void classify_and_align_next_spike(
                        __global float *fifo_residuals,
                        __global st_spike *next_spike,
                        __global st_peak *next_peak,
                        __global  st_peak *pending_peaks,
                        __global int *nb_pending_peaks,
                        float alien_value_threshold,
                        
                        __global  float *catalogue_center0,
                        __global  float *catalogue_center1,
                        __global  float *catalogue_center2,
                        __global  float *catalogue_inter_center0,
                        
                        __global  uchar  *sparse_mask_level1,
                        __global  uchar  *sparse_mask_level2,
                        __global  uchar  *sparse_mask_level3,
                        __global  float *scalar_products,
                        __global  float *distance_shifts,
                        
                        __global float *channel_distances,
                        float adjacency_radius_um,

                        __global int *extremum_channel,
                        __global float *wf1_norm2,
                        __global float *wf2_norm2,
                        __global float *wf1_dot_wf2,
                        __global float *best_distance,

                        __global  st_spike *good_spikes,
                        __global int *nb_good_spikes,
                        __global  uchar *mask_already_tested,
                        __global float *weight_per_template,
                        __global float *distance_limit
                        
                        
            ){
    
    ndrange_t ndrange1 = ndrange_1D(1);
    enqueue_kernel(get_default_queue(),
                            CLK_ENQUEUE_FLAGS_NO_WAIT,
                            ndrange1,
                           ^{select_next_peak(
                                    fifo_residuals,
                                    next_peak,
                                    pending_peaks,
                                    nb_pending_peaks,
                                    next_spike,
                                    alien_value_threshold);}
                            );
    
    size_t global_size_2[2] = {nb_cluster, nb_channel};
    size_t local_size_2[2] = {ls_0, nb_channel};
    ndrange_t ndrange2 = ndrange_2D(global_size_2, local_size_2);
    enqueue_kernel(get_default_queue(),
                            CLK_ENQUEUE_FLAGS_NO_WAIT,
                            ndrange2,
                           ^{explore_templates(
                                    fifo_residuals,
                                    next_spike,
                                    next_peak,
                                    catalogue_center0,
                                    sparse_mask_level3,
                                    scalar_products,
                                    channel_distances,
                                    adjacency_radius_um);}
                            );
    
    
    ndrange_t ndrange3 = ndrange_1D(1);
    enqueue_kernel(get_default_queue(),
                            CLK_ENQUEUE_FLAGS_NO_WAIT,
                            ndrange3,
                           ^{get_best_template(
                                    next_spike,
                                    scalar_products);}
                            );

    size_t global_size_4[2] = {nb_shift, nb_channel};
    size_t local_size_4[2] = {nb_shift, 1};
    ndrange_t ndrange4 = ndrange_2D(global_size_2, local_size_2);
    enqueue_kernel(get_default_queue(),
                            CLK_ENQUEUE_FLAGS_NO_WAIT,
                            ndrange4,
                           ^{explore_shifts(
                                    fifo_residuals,
                                    next_spike,
                                    next_peak,
                                    catalogue_center0,
                                    sparse_mask_level2,
                                    scalar_products,
                                    distance_shifts);}
                            );

    ndrange_t ndrange5 = ndrange_1D(1);
    enqueue_kernel(get_default_queue(),
                            CLK_ENQUEUE_FLAGS_NO_WAIT,
                            ndrange5,
                           ^{best_shift_and_jitter(
                                    fifo_residuals,
                                    next_spike,
                                    next_peak,
                                    catalogue_center0,
                                    catalogue_center1,
                                    catalogue_center2,
                                    catalogue_inter_center0,
                                    distance_shifts,
                                    extremum_channel,
                                    wf1_norm2,
                                    wf2_norm2,
                                    wf1_dot_wf2,
                                    best_distance);}
                            );
    

    ndrange_t ndrange6 = ndrange_1D(1);
    enqueue_kernel(get_default_queue(),
                            CLK_ENQUEUE_FLAGS_NO_WAIT,
                            ndrange6,
                           ^{finalize_next_spike(
                                    fifo_residuals,
                                    next_spike,
                                    next_peak,
                                    pending_peaks,
                                    nb_pending_peaks,
                                    good_spikes,
                                    nb_good_spikes,
                                    mask_already_tested,
                                    catalogue_center0,
                                    catalogue_center1,
                                    catalogue_center2,
                                    catalogue_inter_center0,
                                    sparse_mask_level1,
                                    sparse_mask_level2,
                                    distance_shifts,
                                    extremum_channel,
                                    wf1_norm2,
                                    wf2_norm2,
                                    wf1_dot_wf2,
                                    weight_per_template,
                                    distance_limit,
                                    best_distance);}
                            );
    
    
    
    
}


"""


