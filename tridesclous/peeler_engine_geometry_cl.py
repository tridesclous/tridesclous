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
        
        self.mask_peaks = np.zeros((self.fifo_size - 2 * self.n_span, self.nb_channel), dtype='uint8')  # bool
        
        # debug to check same ctx and queue as processor
        if self.signalpreprocessor is not None:
            assert self.ctx is self.signalpreprocessor.ctx
        
        # make kernels
        centers = self.catalogue['centers0']
        nb_channel = centers.shape[2]
        self.peak_width = centers.shape[1]
        self.nb_cluster = centers.shape[0]
        
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
        
        
        

        # create CL buffers
        self.fifo_residuals_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_residuals)
        self.channel_distances_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.channel_distances)

        self.neighbours_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.neighbours)
        self.mask_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.mask_peaks)

        
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

        self.waveform_distance = np.zeros((self.nb_cluster), dtype='float32')
        self.waveform_distance_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.waveform_distance)

        #~ mask[:] = 0
        self.sparse_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.sparse_mask.astype('u1'))
        self.high_sparse_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.high_sparse_mask.astype('u1'))
        
        rms_waveform_channel = np.zeros(nb_channel, dtype='float32')
        self.rms_waveform_channel_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=rms_waveform_channel)
        
        #~ self.adjacency_radius_um_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=np.array([self.adjacency_radius_um], dtype='float32'))
        
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
        
        
        # TODO
        nb_max_emnding_peaks = self.nb_channel * self.fifo_size
        print('nb_max_emnding_peaks', nb_max_emnding_peaks)
        
        self.pending_peaks = np.zeros(nb_max_emnding_peaks, dtype=[('peak_index', 'int32'), ('peak_chan', 'int32'), ('peak_value', 'float32')])
        self.pending_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.pending_peaks)
        
        self.nb_pending_peaks = np.zeros(1, dtype='int32')
        self.nb_pending_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.nb_pending_peaks)

        # kernel calls
        self.kern_add_fifo_residuals = getattr(self.opencl_prg, 'add_fifo_residuals')
        
        self.kern_detect_local_peaks = getattr(self.opencl_prg, 'detect_local_peaks')
        self.kern_detect_local_peaks.set_args(self.fifo_residuals_cl, self.neighbours_cl, self.mask_peaks_cl, self.pending_peaks_cl, self.nb_pending_peaks_cl)

        
        exit()

        


    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
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

    def detect_local_peaks_before_peeling_loop(self):
        
        self.kern_get_mask_spatiotemporal_peaks
        
    
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

typedef struct st_peak{
    int peak_index;
    int peak_chan;
    float peak_value;
} st_peak;



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


__kernel void detect_local_peaks(
                        __global  float *sigs,
                        __global  int *neighbours,
                        __global  uchar *mask_peaks,
                        __global  st_peak *pending_peaks,
                        volatile __global int *nb_pending_peaks
                ){
    int pos = get_global_id(0);
    
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
            // peak_index is LOCAL to fifo
            pending_peaks[i_peak].peak_index = pos + n_span;
            pending_peaks[i_peak].peak_chan = chan;
            pending_peaks[i_peak].peak_value = fabs(v);
        }
    }
    
}


void select_next_peak(__local st_peak *peak,
                        __global  st_peak *pending_peaks,
                        volatile __global int *nb_pending_peaks){
    
    // take max amplitude
    
    
    int i_peak = -1;
    float best_value = 0.0;
    
    for (int i=1; i<=n_span; i++){
        if (pending_peaks[i].peak_value > best_value){
            i_peak = i;
            best_value = pending_peaks[i].peak_value;
        }
    }
    
    if (i_peak == -1){
        peak->peak_index = LABEL_NO_MORE_PEAK;
        peak->peak_chan = -1;
        peak->peak_value = 0.0;
    }else{
        peak->peak_index = pending_peaks[i_peak].peak_index;
        peak->peak_chan = pending_peaks[i_peak].peak_chan;
        peak->peak_value = pending_peaks[i_peak].peak_value;
        
        // make this peak not selectable for next choice
        pending_peaks[i_peak].peak_value = 0.0;
        
    }
    
    
}


__kernel void classify_and_align_next_spike(__global  float *fifo_residuals,
                                                                __global st_spike *spike,


                                                                __global  st_peak *pending_peaks,
                                                                volatile __global int *nb_pending_peaks,
                                                                
                                                                
                                                                __global  uchar *mask_already_tested,
                                                                __global  float *catalogue_center0,
                                                                __global  float *catalogue_center1,
                                                                __global  float *catalogue_center2,
                                                                __global  float *catalogue_inter_center0,
                                                                __global  uchar  *sparse_mask,
                                                                __global  uchar  *high_sparse_mask,
                                                                __global  float *rms_waveform_channel,
                                                                __global  float *waveform_distance,
                                                                __global int *extremum_channel,
                                                                __global float *wf1_norm2,
                                                                __global float *wf2_norm2,
                                                                __global float *wf1_dot_wf2,
                                                                __global float *weight_per_template,
                                                                
                                                                __global float *channel_distances,
                                                                float adjacency_radius_um,
                                                                
                                                                float alien_value_threshold){

    int cluster_idx = get_global_id(0);
    int chan = get_global_id(1);
    
    int left_ind;
    __local float jitter;
    __local float new_jitter;
    
    __local st_peak peak;

    

    // non parralel code only for first worker
    if ((cluster_idx==0) && (chan==0)){
        
        select_next_peak(&peak, pending_peaks, nb_pending_peaks);
        
        if (spike->peak_index == LABEL_NO_MORE_PEAK){
            spike->label = LABEL_NO_MORE_PEAK;
            spike->jitter = 0.0f;
        } else if (left_ind+peak_width+maximum_jitter_shift+1>=fifo_size){
            spike->label = LABEL_RIGHT_LIMIT;
        } else if (left_ind<=maximum_jitter_shift){
            spike->label = LABEL_LEFT_LIMIT;
        } else if (n_cluster==0){
            spike->label  = LABEL_UNCLASSIFIED;
        }else {
            spike->label = LABEL_NOT_YET;
            if (alien_value_threshold>0){
                for (int s=1; s<=(wf_size); s++){
                    if ((fifo_residuals[left_ind*nb_channel +s]>alien_value_threshold) || (fifo_residuals[left_ind*nb_channel +s]<-alien_value_threshold)){
                        spike->label = LABEL_ALIEN;
                        spike->jitter = 0.0f;
                    }
                }
            }
        }
        
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (spike->label==LABEL_NOT_YET) {

        //initialize parralel on cluster_idx
        if (chan==0){
            // the peak_chan do not overlap spatialy this cluster
            if (high_sparse_mask[nb_channel*cluster_idx+peak.peak_chan] == 0){
                if (chan==0){
                    waveform_distance[cluster_idx] = FLT_MAX;
                }
            }
            else {
                // candidate initialize sum by cluster
                if (chan==0){
                    waveform_distance[cluster_idx] = 0.0f;
                }
            }
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // this is parralel (cluster_idx, chan)
        if (high_sparse_mask[nb_channel*cluster_idx+peak.peak_chan] == 1){
        
            if (channel_distances[chan * nb_channel + peak.peak_chan] < adjacency_radius_um){
                float sum = 0;
                float d;
                for (int s=0; s<peak_width; ++s){
                    d =  fifo_residuals[(left_ind+s)*nb_channel + chan] - catalogue_center0[wf_size*cluster_idx+nb_channel*s+chan];
                    sum += d*d;
                }
                atomic_add_float(&waveform_distance[cluster_idx], sum);
            }
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        // not parralel zone only first worker
        if ((chan==0) && (cluster_idx==0)){
            //argmin  not paralel
            float min_dist = MAXFLOAT;
            spike->label = -1;
            
            for (int clus=0; clus<n_cluster; ++clus){
                if (waveform_distance[clus]<min_dist){
                    spike->label = clus;
                    min_dist = waveform_distance[clus];
                }
            }
            
            // explore shifts
            int shift;
            
            
            
            int ok;
            
            int int_jitter;
        
        }
        
        
        
        
    
    }

    
    

}




"""


kernel_peeler_cl = kernel_peeler_cl + opencl_kernel_geometrical_part2

