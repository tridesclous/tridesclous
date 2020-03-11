import time
import numpy as np

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
    def change_params(self, adjacency_radius_um=100, **kargs): # high_adjacency_radius_um=50, 
        assert HAVE_PYOPENCL
        
        PeelerEngineGeneric.change_params(self, **kargs)
        
        assert self.use_sparse_template
        
        self.adjacency_radius_um = adjacency_radius_um
        
        self.shifts = np.arange(-self.maximum_jitter_shift, self.maximum_jitter_shift+1)
        


    def initialize(self, **kargs):
        if self.argmin_method == 'opencl':
            OpenCL_Helper.initialize_opencl(self, cl_platform_index=self.cl_platform_index, cl_device_index=self.cl_device_index)
        
        PeelerEngineGeneric.initialize(self, **kargs)
        
        
        # make neighbours for peak detector CL
        d = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
        neighbour_mask = d<=self.catalogue['peak_detector_params']['adjacency_radius_um']
        nb_neighbour_per_channel = np.sum(neighbour_mask, axis=0)
        nb_max_neighbour = np.max(nb_neighbour_per_channel)
        self.nb_max_neighbour = nb_max_neighbour # include itself
        self.neighbours = np.zeros((self.nb_channel, nb_max_neighbour), dtype='int32')
        self.neighbours[:] = -1
        for c in range(self.nb_channel):
            neighb, = np.nonzero(neighbour_mask[c, :])
            self.neighbours[c, :neighb.size] = neighb
        
        
        # debug to check same ctx and queue as processor
        if self.signalpreprocessor is not None:
            assert self.ctx is self.signalpreprocessor.ctx
        
        # make kernels
        centers = self.catalogue['centers0']
        nb_channel = centers.shape[2]
        self.peak_width = centers.shape[1]
        
        self.n_cluster = self.catalogue['centers0'].shape[0]
        wf_size = self.catalogue['centers0'].shape[1]*self.catalogue['centers0'].shape[2]

        kernel_formated = kernel_peeler_cl % dict(
                    chunksize=self.chunksize,
                    n_span=self.n_span,
                    nb_channel=self.nb_channel,
                    relative_threshold=self.relative_threshold,
                    peak_sign={'+':1, '-':-1}[self.peak_sign],
                    extra_size=self.extra_size,
                    fifo_size=self.fifo_size,
                    n_left=self.n_left,
                    n_right=self.n_right,
                    peak_width=self.peak_width,
                    maximum_jitter_shift=self.maximum_jitter_shift,
                    n_cluster=self.n_cluster,
                    wf_size=wf_size,
                    subsample_ratio=self.catalogue['subsample_ratio'],
                    nb_neighbour=self.nb_max_neighbour)
        #~ print(kernel_formated)
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.kern_add_fifo_residuals = getattr(self.opencl_prg, 'add_fifo_residuals')
        self.kern_get_mask_spatiotemporal_peaks = getattr(self.opencl_prg, 'get_mask_spatiotemporal_peaks')
        
        exit()

        # create CL buffers
        self.fifo_residuals_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_residuals)
        self.channel_distances_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.channel_distances)
        
        self.waveform_distance_shifts = np.zeros((self.shifts.size, ), dtype='float32')
        self.waveform_distance_shifts_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.waveform_distance_shifts)

        self.mask_not_already_tested = np.ones((self.fifo_size - 2 * self.n_span,self.nb_channel),  dtype='bool')
        self.mask_not_already_tested_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.mask_not_already_tested)

        
        self.fifo_residuals_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_residuals)
        
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
        self.high_sparse_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.high_sparse_mask.astype('u1'))
        
        rms_waveform_channel = np.zeros(nb_channel, dtype='float32')
        self.rms_waveform_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=rms_waveform_channel)
        
        self.adjacency_radius_um_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=np.array([self.adjacency_radius_um], dtype='float32'))
        
        # attention : label in CL is the label index
        self.spike = np.zeros(1, dtype=[('peak_index', 'int32'), ('label_index', 'int32'), ('jitter', 'float32')])
        self.spike_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.spike)

        self.catalogue_center0_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers0'])
        self.catalogue_center1_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers1'])
        self.catalogue_center2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers2'])
        self.catalogue_inter_center0_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['interp_centers0'])
        
        
        self.extremum_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['extremum_channel'].astype('int32'))
        self.wf1_norm2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['wf1_norm2'].astype('float32'))
        self.wf2_norm2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['wf2_norm2'].astype('float32'))
        self.wf1_dot_wf2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['wf1_dot_wf2'].astype('float32'))

        self.weight_per_template = np.zeros((self.n_cluster, self.nb_channel), dtype='float32')
        centers = self.catalogue['centers0']
        for i, k in enumerate(self.catalogue['cluster_labels']):
            mask = self.sparse_mask[i, :]
            wf = centers[i, :, :][:, mask]
            self.weight_per_template[i, mask] = np.sum(wf**2, axis=0)
        self.weight_per_template_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.weight_per_template)
        

        
        
        
        #~ self.cl_global_size = (centers.shape[0], centers.shape[2])
        #~ self.cl_local_size = (centers.shape[0], 1) # faster a GPU because of memory access

        #~ self.cl_global_size2 = (len(self.shifts), centers.shape[2])
        #~ self.cl_local_size2 = (len(self.shifts), 1) # faster a GPU because of memory access
        


    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        self.peakdetector.reset_fifo_index()
        #~ self.mask_not_already_tested[:] = 1
        

            
    
    
    def apply_processor(self, pos, sigs_chunk):
        if self.already_processed:
            abs_head_index, preprocessed_chunk =  pos, sigs_chunk

            n = self.fifo_residuals.shape[0]-self.chunksize
            global_size = (self.chunksize, self.nb_channel)
            #~ local_size = None
            local_size = (1, self.nb_channel)
            event = self.kern_add_fifo_residuals(self.queue, global_size, local_size,
                        self.fifo_residuals_cl, sp.output_backward_cl, np.int32(n))

            
        else:
            abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        
        #shift residuals buffer and put the new one on right side
        fifo_roll_size = self.fifo_size-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_size:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk
        
        return abs_head_index, preprocessed_chunk 

    #~ def detect_local_peaks_before_peeling_loop(self):
    
    #~ def classify_and_align_next_spike(self):
        
    #~ def reset_to_not_tested(self):


kernel_peeler_cl = """
#define chunksize %(chunksize)d
#define n_span %(n_span)d
#define nb_channel %(nb_channel)d
#define relative_threshold %(relative_threshold)d
#define peak_sign %(peak_sign)d
#define extra_size %(extra_size)d
#define fifo_size %(fifo_size)d
#define n_left %(n_left)d
#define n_right %(n_right)d
#define peak_width %(peak_width)d
#define maximum_jitter_shift %(maximum_jitter_shift)d
#define n_cluster %(n_cluster)d
#define wf_size %(wf_size)d
#define subsample_ratio %(subsample_ratio)d
#define nb_neighbour %(nb_neighbour)d



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
    int peak_index;
    int label;
    float jitter;
} st_spike;


__kernel void add_fifo_residuals(__global  float *fifo_residuals, __global  float *sigs_chunk, int fifo_roll_size){
    int pos = get_global_id(0);
    int chan = get_global_id(1);
    
    //work on ly for n<chunksize
    if (pos<fifo_roll_size){
        fifo_residuals[pos*nb_channel+chan] = fifo_residuals[(pos+chunksize)*nb_channel+chan];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    
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


__kernel void estimate_one_jitter(int left_ind, int cluster, __local float *jitter,
                                        __global int *extremum_channel,
                                        __global float *fifo_residuals,
                                        
                                        __global float *catalogue_center0,
                                        __global float *catalogue_center1,
                                        __global float *catalogue_center2,
                                        
                                        __global float *wf1_norm2,
                                        __global float *wf2_norm2,
                                        __global float *wf1_dot_wf2){

    int cluster_idx = get_global_id(0);
    int chan = get_global_id(1);
    
    int chan_max;
    if ((cluster_idx!=0) || (chan!=0)){
        // nomally this is never call
        *jitter =0.0f;
    } else {
    
        
        chan_max= extremum_channel[cluster];
        
        
        float h, wf1, wf2;
        
        float h0_norm2 = 0.0f;
        float h_dot_wf1 =0.0f;
        for (int s=0; s<peak_width; ++s){
            h = fifo_residuals[(left_ind+s)*nb_channel + chan_max] - catalogue_center0[wf_size*cluster+nb_channel*s+chan_max];
            h0_norm2 +=  h*h;
            wf1 = catalogue_center1[wf_size*cluster+nb_channel*s+chan_max];
            h_dot_wf1 += h * wf1;
            
        }
        
        *jitter = h_dot_wf1/wf1_norm2[cluster];
        
        float h1_norm2;
        h1_norm2 = 0.0f;
        for (int s=0; s<peak_width; ++s){
            h = fifo_residuals[(left_ind+s)*nb_channel + chan_max] - catalogue_center0[wf_size*cluster+nb_channel*s+chan_max];
            wf1 = catalogue_center1[wf_size*cluster+nb_channel*s+chan_max];
            h1_norm2 += (h - *jitter * wf1) * (h - (*jitter) * wf1);
        }
        
        if (h0_norm2 > h1_norm2){
            float h_dot_wf2 =0.0f;
            float rss_first, rss_second;
            for (int s=0; s<peak_width; ++s){
                h = fifo_residuals[(left_ind+s)*nb_channel + chan_max] - catalogue_center0[wf_size*cluster+nb_channel*s+chan_max];
                wf2 = catalogue_center2[wf_size*cluster+nb_channel*s+chan_max];
                h_dot_wf2 += h * wf2;
            }
            rss_first = -2*h_dot_wf1 + 2*(*jitter)*(wf1_norm2[cluster] - h_dot_wf2) + pown(3* (*jitter), 2)*wf1_dot_wf2[cluster] + pown((*jitter),3)*wf2_norm2[cluster];
            rss_second = 2*(wf1_norm2[cluster] - h_dot_wf2) + 6* (*jitter)*wf1_dot_wf2[cluster] + 3*pown((*jitter), 2) * wf2_norm2[cluster];
            *jitter = (*jitter) - rss_first/rss_second;
        } else{
            *jitter = 0.0f;
        }
    }
}




"""


kernel_peeler_cl = kernel_peeler_cl + opencl_kernel_geometrical_part2

