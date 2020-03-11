import time
import numpy as np

from .peeler_tools import *
from .peeler_tools import _dtype_spike
from . import signalpreprocessor

from .cltools import HAVE_PYOPENCL, OpenCL_Helper

#from .signalpreprocessor import signalpreprocessor_engines, processor_kernel

from .peeler_engine_base import PeelerEngineGeneric


from .signalpreprocessor import processor_kernel

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
        
        if  self.catalogue['centers0'].size>0:
        #~ if self.use_opencl_with_sparse and self.catalogue['centers0'].size>0:
            OpenCL_Helper.initialize_opencl(self, cl_platform_index=self.cl_platform_index, cl_device_index=self.cl_device_index)
            
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
            
            
            self.cl_global_size = (centers.shape[0], centers.shape[2])
            #~ self.cl_local_size = None
            self.cl_local_size = (centers.shape[0], 1) # faster a GPU because of memory access
            #~ self.cl_local_size = (1, centers.shape[2])

            self.cl_global_size2 = (len(self.shifts), centers.shape[2])
            #~ self.cl_local_size = None
            self.cl_local_size2 = (len(self.shifts), 1) # faster a GPU because of memory access
            #~ self.cl_local_size = (1, centers.shape[2])
            
            # to check if distance is valid is a coeff (because maxfloat on opencl)
            #~ self.max_float32 = np.finfo('float32').max * 0.8

    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        
        #~ p = dict(self.catalogue['peak_detector_params'])
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
        
        #~ PeakDetector_class = get_peak_detector_class(self.peakdetector_method, self.peakdetector_engine)
        
        chunksize = self.fifo_size-2*self.n_span # not the real chunksize here
        #~ self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        #~ chunksize, self.internal_dtype, self.geometry)
        #~ self.peakdetector.change_params(**p)
        
        self.channel_distances = sklearn.metrics.pairwise.euclidean_distances(self.geometry).astype('float32')
        self.channels_adjacency = {}
        for c in range(self.nb_channel):
            nearest, = np.nonzero(self.channel_distances[c, :]<self.adjacency_radius_um)
            self.channels_adjacency[c] = nearest
        
        if self.catalogue['centers0'].size>0:
            self.channel_distances_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.channel_distances)
            self.all_distance = np.zeros((self.shifts.size, ), dtype='float32')
            self.all_distance_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.all_distance)
            
        self.mask_not_already_tested = np.ones((self.fifo_size - 2 * self.n_span,self.nb_channel),  dtype='bool')
        self.mask_not_already_tested_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.mask_not_already_tested)
    
    
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

    
    #~ def classify_and_align_next_spike(self):
        


kernel_opencl = """
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



"""