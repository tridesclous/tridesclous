import time
import numpy as np

from .peeler_engine_classic import PeelerEngineClassic
from .peeler_tools import *
from .peeler_tools import _dtype_spike
from .tools import make_color_dict
from . import signalpreprocessor

from .signalpreprocessor import signalpreprocessor_engines, processor_kernel




try:
    import pyopencl
    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False




class PeelerEngineClassicOpenCl(PeelerEngineClassic):
    
    def change_params(self, **kargs):
        
        # TODO force open_cl use sparse ????
        kargs['use_sparse_template'] = True
        kargs['argmin_method'] = 'opencl'
        
        PeelerEngineClassic.change_params(self, **kargs)
        
    
            

    def initialize_before_each_segment(self, *args, **kargs):

        PeelerEngineClassic.initialize_before_each_segment(self, *args, **kargs)
        
        # kernel processor
        # TODO make compilation explicit here
        SignalPreprocessor_class = signalpreprocessor_engines['opencl']
        self.signalpreprocessor = SignalPreprocessor_class(self.sample_rate, self.nb_channel, self.chunksize, self.source_dtype)
        p = dict(self.catalogue['signal_preprocessor_params'])
        p.pop('engine')
        p['normalize'] = True
        p['signals_medians'] = self.catalogue['signals_medians']
        p['signals_mads'] = self.catalogue['signals_mads']
        p['cl_platform_index'] = None
        p['cl_device_index'] = None
        p['ctx'] = self.ctx
        p['queue'] = self.queue
        
        self.signalpreprocessor.change_params(**p)
        # map kernel and buffer in self for simplicity
        #~ self.kern_forward_backward_filter = self.signalpreprocessor.kern_forward_backward_filter
        
        assert not self.signalpreprocessor.common_ref_removal, 'common_ref_removal in CL peeler not inmplemented'
        
        
        
        
        
        n = self.fifo_residuals.shape[0]-self.chunksize
        assert n<self.chunksize, 'opencl kernel for fifo add not work'
        
        self.n_cluster = self.catalogue['centers0'].shape[0]
        wf_size = self.catalogue['centers0'].shape[1]*self.catalogue['centers0'].shape[2]
        
        kernel_formated = kernel_peeler_cl % dict(chunksize=self.chunksize,  n_span=self.n_span, nb_channel=self.nb_channel,
                    relative_threshold=self.relative_threshold, peak_sign={'+':1, '-':-1}[self.peak_sign],
                    n_side=self.n_side, fifo_size=self.chunksize+self.n_side, n_left=self.n_left, peak_width=self.catalogue['peak_width'],
                    maximum_jitter_shift=self.maximum_jitter_shift, n_cluster=self.n_cluster, wf_size=wf_size,
                    subsample_ratio=self.catalogue['subsample_ratio'])
        #~ print(kernel_formated)
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        #~ self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        
        #~ self.preprocessed_chunk = np.zeros((self.chunksize, self.nb_channel), dtype=self.internal_dtype)
        #~ self.preprocessed_chunk_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.preprocessed_chunk)
        
        self.fifo_residuals_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_residuals)
        
        self.fifo_sum = np.zeros((self.chunksize+self.n_side,), dtype=self.internal_dtype)
        self.fifo_sum_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_sum)
        
        self.local_peaks_mask = np.zeros((self.chunksize + self.n_side - 2 * self.n_span), dtype='uint8')
        self.local_peaks_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.local_peaks_mask)

        self.mask_already_tested = np.ones((self.chunksize + self.n_side - 2 * self.n_span), dtype='uint8')
        self.mask_already_tested_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.local_peaks_mask)
        
        self.min_peak_index_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=4) # int32
        
        # attention : label in CL is the label index
        self.spike = np.zeros(1, dtype=[('peak_index', 'int32'), ('label_index', 'int32'), ('jitter', 'float32')])
        self.spike_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.spike)
        
        # self.sparse_mask_cl, self.rms_waveform_channel_cl done in peeler_engine_classic
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
        
        
        
        #kernels links
        self.kern_add_fifo_residuals = getattr(self.opencl_prg, 'add_fifo_residuals')
        self.kern_detect_boolean_peaks = getattr(self.opencl_prg, 'detect_boolean_peaks')
        self.kern_select_next_peak = getattr(self.opencl_prg, 'select_next_peak')
        self.kern_classify_and_align_next_spike = getattr(self.opencl_prg, 'classify_and_align_next_spike')
        
        
        ## self.kern_classify_and_align_one_spike = getattr(self.opencl_prg, 'classify_and_align_one_spike')
        ## self.kern_make_prediction_signals = getattr(self.opencl_prg, 'make_prediction_signals')
        #~ self.kern_waveform_distance = getattr(self.opencl_prg, 'waveform_distance')




    def process_one_chunk(self,  pos, sigs_chunk):
        #~ print('*'*5)
        #~ print('chunksize', self.chunksize, '=', self.chunksize/self.sample_rate*1000, 'ms')


        assert sigs_chunk.shape[0]==self.chunksize
        
        if not sigs_chunk.flags['C_CONTIGUOUS'] or sigs_chunk.dtype!=self.internal_dtype:
            sigs_chunk = np.ascontiguousarray(sigs_chunk, dtype=self.internal_dtype)
        
        
        #Online filtfilt
        sp = self.signalpreprocessor
        event = pyopencl.enqueue_copy(self.queue,  sp.input_cl, sigs_chunk)
        event = sp.kern_forward_backward_filter(self.queue,  (self.nb_channel,), (self.nb_channel,),
                            sp.input_cl, sp.coefficients_cl, sp.zi1_cl, sp.zi2_cl,
                            sp.fifo_input_backward_cl, sp.signals_medians_cl, sp.signals_mads_cl,  
                            sp.output_backward_cl)
        event.wait()
        abs_head_index = pos - self.signalpreprocessor.lostfront_chunksize
        
        # add in fifo residuals TODO merge this online filtfilt (one call)
        n = self.fifo_residuals.shape[0]-self.chunksize
        global_size = (self.chunksize, self.nb_channel)
        #~ local_size = None
        local_size = (1, self.nb_channel)
        event = self.kern_add_fifo_residuals(self.queue, global_size, local_size,
                    self.fifo_residuals_cl, sp.output_backward_cl, np.int32(n))

        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        # TODO remove from peak the very begining of the signal because of border filtering effects

        
     
        
        #~ already_tested = []
        
        # negative mask 1: not tested 0: already tested
        #~ mask_already_tested = np.ones(self.fifo_residuals.shape[0] - 2 * self.n_span, dtype='bool')
        
        # TODO do it in the kernel
        
        #debug
        self.mask_already_tested[:] = 1
        #end debug
        
        event = pyopencl.enqueue_copy(self.queue,  self.mask_already_tested_cl, self.mask_already_tested)
        
        #~ local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)

        
        
        # debug
        #~ event = pyopencl.enqueue_copy(self.queue,  self.local_peaks_mask, self.local_peaks_mask_cl)
        #~ event = pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
        #~ self.fifo_residuals = self.fifo_residuals.copy()
        #~ local_peaks_mask = self.local_peaks_mask
        #~ mask_already_tested = self.mask_already_tested.copy()
        # end debug

        #~ print('yep')
        #~ exit()
        
        
        
        #~ print('sum(local_peaks_mask)', np.sum(local_peaks_mask))
        
        t1 = time.perf_counter()
        good_spikes = []
        while True:
            #~ print()
            
            # TODO remove this when boolean detectiona round peak is done
            global_size = (self.chunksize+self.n_side,  )
            #~ local_size = (min(self.max_wg_size, self.chunksize+self.n_side), )
            local_size = None
            #~ print(global_size, local_size)
            event = self.kern_detect_boolean_peaks(self.queue,  global_size, local_size,
                                    self.fifo_residuals_cl, self.fifo_sum_cl, self.local_peaks_mask_cl)
            
            

            
            global_size = (self.chunksize+self.n_side - 2*self.n_span,  )
            #~ local_size = (min(self.max_wg_size, self.chunksize+self.n_side), )
            local_size = None
            # TODO : bug in this function!!!!
            event = self.kern_select_next_peak(self.queue,  global_size, local_size,
                                    self.local_peaks_mask_cl, self.mask_already_tested_cl, self.min_peak_index_cl, self.spike_cl)
            


            global_size = (self.n_cluster, self.nb_channel)
            local_size = (self.n_cluster, 1) # faster a GPU because of memory access ????? True argmin only ?????
            self.kern_classify_and_align_next_spike(self.queue,  global_size, local_size,
                                    self.fifo_residuals_cl, self.spike_cl, self.mask_already_tested_cl,
                                    self.catalogue_center0_cl, self.catalogue_center1_cl, self.catalogue_center2_cl,
                                    self.catalogue_inter_center0_cl,
                                    self.sparse_mask_cl, self.rms_waveform_channel_cl,
                                    self.waveform_distance_cl, self.extremum_channel_cl,
                                    self.wf1_norm2_cl, self.wf2_norm2_cl, self.wf1_dot_wf2_cl,
                                    self.weight_per_template_cl, np.float32(self.alien_value_threshold))
            event = pyopencl.enqueue_copy(self.queue,  self.spike, self.spike_cl)


            # debug
            #~ event = pyopencl.enqueue_copy(self.queue,  self.local_peaks_mask, self.local_peaks_mask_cl)
            #~ event = pyopencl.enqueue_copy(self.queue,  self.mask_already_tested, self.mask_already_tested_cl)
            #~ local_peaks,  = np.nonzero(self.local_peaks_mask & self.mask_already_tested)
            #~ local_peaks = local_peaks + self.n_span
            #~ print()
            #~ print('debug local_peaks', local_peaks)
            #~ event = pyopencl.enqueue_copy(self.queue,  self.spike, self.spike_cl)
            #~ print(self.spike)
            #~ event = pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
            #~ import matplotlib.pyplot as plt
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.fifo_residuals)
            #~ ax.plot(np.arange(self.mask_already_tested.size) + self.n_span, self.mask_already_tested.astype(float)*10, color='k')
            #~ ax.scatter(local_peaks, np.min(self.fifo_residuals[local_peaks, :], axis=1), color='k')
            #~ plt.show()
            # end debug
            
            #~ print(self.spike)
            
            #~ print(self.spike[0])
            if self.spike[0]['label_index'] ==LABEL_NO_MORE_PEAK:
                # no more peak
                break
            elif self.spike[0]['label_index'] >=0:
                spike = np.zeros(1, dtype=_dtype_spike)
                spike[0]['index'] = self.spike[0]['peak_index'] + shift
                spike[0]['cluster_label'] = self.catalogue['cluster_labels'][self.spike[0]['label_index']] # label is the label_idx
                spike[0]['jitter'] = self.spike[0]['jitter']
                good_spikes.append(spike)
                #~ print(spike)
                
                # TODO kern_detect_boolean_peaks only around peakindex
                
                
            #~ elif self.spike[0]['label_index'] >=0:
                

        ##################
        ### TODO
        ### prediction with interpolate
        #### bad spike
        
        
        # TODO
        bad_spikes = np.zeros(0, dtype=_dtype_spike)
        
        #concatenate, sort and count
        # here the trick is to keep spikes at the right border
        # and keep then until the next loop this avoid unordered spike
        if len(good_spikes)>0:
            good_spikes = np.concatenate(good_spikes)
            near_border = (good_spikes['index'] - shift)>=(self.chunksize+self.n_span)
            near_border_good_spikes = good_spikes[near_border].copy()
            good_spikes = good_spikes[~near_border]

            all_spikes = np.concatenate([good_spikes] + [bad_spikes] + self.near_border_good_spikes)
            self.near_border_good_spikes = [near_border_good_spikes] # for next chunk
        else:
            all_spikes = np.concatenate([bad_spikes] + self.near_border_good_spikes)
            self.near_border_good_spikes = []
        
        # all_spikes = all_spikes[np.argsort(all_spikes['index'])]
        all_spikes = all_spikes.take(np.argsort(all_spikes['index']))
        self.total_spike += all_spikes.size


        # TODO: do beeter than copy/paste from signal processor
        start = pos - sp.backward_chunksize
        event = pyopencl.enqueue_copy(self.queue,  sp.output_backward, sp.output_backward_cl)
        #~ print(sp.output_backward)
        #~ print('*'*50)
        if start>0:
            preprocessed_chunk = sp.output_backward[:self.chunksize, :]
        else:
            preprocessed_chunk = sp.output_backward[sp.lostfront_chunksize:self.chunksize, :]
        
        return abs_head_index, preprocessed_chunk, self.total_spike, all_spikes





kernel_peeler_cl = """
#define chunksize %(chunksize)d
#define n_span %(n_span)d
#define nb_channel %(nb_channel)d
#define relative_threshold %(relative_threshold)d
#define peak_sign %(peak_sign)d
#define n_side %(n_side)d
#define fifo_size %(fifo_size)d
#define n_left %(n_left)d
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


__kernel void add_fifo_residuals(__global  float *fifo_residuals, __global  float *sigs_chunk, int n){
    int pos = get_global_id(0);
    int chan = get_global_id(1);
    
    //work on ly for n<chunksize
    if (pos<n){
        fifo_residuals[pos*nb_channel+chan] = fifo_residuals[(pos+chunksize)*nb_channel+chan];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    fifo_residuals[(pos+n)*nb_channel+chan] = sigs_chunk[pos*nb_channel+chan];
}




__kernel void detect_boolean_peaks(__global  float *fifo_residuals,
                                            __global  float *fifo_sum,
                                            __global  uchar *local_peaks_mask){

    int pos = get_global_id(0); //fifo_residuals.shape[0]
    
    int idx;
    float v;
    
    
    // sum all channels
    float sum=0.0f;
    for (int chan=0; chan<nb_channel; chan++){
        idx = pos*nb_channel + chan;
        
        v = fifo_residuals[idx];
        
        //retify signals
        if(peak_sign==1){
            if (v<relative_threshold){v=0.0f;}
        }
        else if(peak_sign==-1){
            if (v>-relative_threshold){v=0.0f;}
        }
        
        sum = sum + v;
        
    }
    fifo_sum[pos] = sum;
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    
    // peaks span
    uchar peak=0;
    if ((pos<n_span)||(pos>=(chunksize+n_side-n_span))){
        // nothing
    }
    else{
        if(peak_sign==1){
            if (fifo_sum[pos]>relative_threshold){
                peak=1;
                for (int i=1; i<=n_span; i++){
                    peak = peak && (fifo_sum[pos]>fifo_sum[pos-i]) && (fifo_sum[pos]>=fifo_sum[pos+i]);
                }
            }
        }
        else if(peak_sign==-1){
            if (fifo_sum[pos]<-relative_threshold){
                peak=1;
                for (int i=1; i<=n_span; i++){
                    peak = peak && (fifo_sum[pos]<fifo_sum[pos-i]) && (fifo_sum[pos]<=fifo_sum[pos+i]);
                }
            }
        }
        local_peaks_mask[pos - n_span]=peak;
    }
}


__kernel void select_next_peak(__global  uchar *local_peaks_mask, __global uchar *mask_already_tested,
                                                volatile __global int *min_peak_index, __global struct st_spike *spike){

    int pos = get_global_id(0); //fifo_residuals.shape[0] - 2 * nspan
    
    if (pos == 0){
        // outside limit
        *min_peak_index = fifo_size +1; 
        spike->peak_index = -1;
        spike->jitter = 0.0f;
        spike->label = LABEL_NO_MORE_PEAK;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if ((local_peaks_mask[pos] == 1) && (mask_already_tested[pos] == 1)){
        atomic_min(min_peak_index, pos);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (pos == 0){
        if (*min_peak_index<(fifo_size +1)){
            spike->peak_index = *min_peak_index + n_span;
            spike->label = LABEL_NOT_YET;
       }
    } 
    
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

int accept_tempate(int left_ind, int cluster, float jitter,
                                        __global float *fifo_residuals,
                                        __global uchar *sparse_mask,
                                        __global float *catalogue_center0,
                                        __global float *catalogue_center1,
                                        __global float *catalogue_center2,
                                        __global float *weight_per_template){


    int cluster_idx = get_global_id(0);
    int chan = get_global_id(1);
    
    //not parallel
    if ((cluster_idx!=0) || (chan!=0)){
        // nomally this is never call
        return 0;
    }
    
    float v;
    float pred;
    
    float chan_in_mask = 0.0f;
    float chan_with_criteria = 0.0f;
    int idx;

    float wf_nrj = 0.0f;
    float res_nrj = 0.0f;
    
    for (int c=0; c<nb_channel; ++c){
        if (sparse_mask[cluster*nb_channel + c] == 1){
            for (int s=0; s<peak_width; ++s){
                v = fifo_residuals[(left_ind+s)*nb_channel + c];
                wf_nrj += v*v;
                
                idx = wf_size*cluster+nb_channel*s+c;
                pred = catalogue_center0[idx] + jitter*catalogue_center1[idx] + jitter*jitter/2*catalogue_center2[idx];
                v = fifo_residuals[(left_ind+s)*nb_channel + c] - pred;
                res_nrj += v*v;
            }
            
            chan_in_mask += weight_per_template[cluster*nb_channel + c];
            if (wf_nrj>res_nrj){
                chan_with_criteria += weight_per_template[cluster*nb_channel + c];
            }
        }
    }
    
    if (chan_with_criteria>(chan_in_mask*0.9f)){
        return 1;
    }else{
        return 0;
    }

}

__kernel void classify_and_align_next_spike(__global  float *fifo_residuals,
                                                                        __global st_spike *spike,
                                                                        __global  uchar *mask_already_tested,
                                                                        __global  float *catalogue_center0,
                                                                        __global  float *catalogue_center1,
                                                                        __global  float *catalogue_center2,
                                                                        __global  float *catalogue_inter_center0,
                                                                        __global  uchar  *sparse_mask,
                                                                        __global  float *rms_waveform_channel,
                                                                        __global  float *waveform_distance,
                                                                        __global int *extremum_channel,
                                                                        __global float *wf1_norm2,
                                                                        __global float *wf2_norm2,
                                                                        __global float *wf1_dot_wf2,
                                                                        __global float *weight_per_template,
                                                                        float alien_value_threshold){

    int cluster_idx = get_global_id(0);
    int chan = get_global_id(1);
    
    int left_ind;
    __local float jitter;
    __local float new_jitter;
    
    
    left_ind = spike->peak_index + n_left;
    

    
    // non parralel code only for first worker
    if ((cluster_idx==0) && (chan==0)){
        
        if (spike->peak_index == -1){
            //spike->label = LABEL_NO_MORE_PEAK;
            //spike->jitter = 0.0f;
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


        // initialize sum waveform distance by cluster
        if (chan==0){
            //parallel reset by cluster
            waveform_distance[cluster_idx] = 0.0f;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // compute rms waveform
        if (cluster_idx==0){
            //paralel waveform sum by chan
            rms_waveform_channel[chan] = 0.0f;
            float v;
            for (int s=1; s<=(peak_width); s++){
                v = fifo_residuals[(left_ind+s)*nb_channel + chan];
                rms_waveform_channel[chan] += (v*v);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // parralel distance for cluster  clusterXchan
        float sum = 0.0f;
        float d;
        if (sparse_mask[nb_channel*cluster_idx+chan]>0){
            for (int s=0; s<peak_width; ++s){
                d = fifo_residuals[(left_ind+s)*nb_channel + chan] - catalogue_center0[wf_size*cluster_idx+nb_channel*s+chan];
                sum += d*d;
            }
        }
        else{
            sum = rms_waveform_channel[chan];
        }
        atomic_add_float(&waveform_distance[cluster_idx], sum);
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
            
            int ok;
            int shift;
            int int_jitter;
            
            estimate_one_jitter(left_ind, spike->label, &jitter,
                                        extremum_channel, fifo_residuals,
                                        catalogue_center0, catalogue_center1, catalogue_center2,
                                        wf1_norm2, wf2_norm2, wf1_dot_wf2);
            ok = accept_tempate(left_ind, spike->label, jitter,
                                        fifo_residuals, sparse_mask,
                                        catalogue_center0, catalogue_center1, catalogue_center2,
                                        weight_per_template);
            
            if (ok==0){
                spike->label = LABEL_UNCLASSIFIED;
                spike->jitter = 0.0f;
            }else{
                if ((fabs(jitter) > 0.5f)){
            
                    shift = - ((int) round(jitter));
                    if (abs(shift) >maximum_jitter_shift){
                        spike->label = LABEL_MAXIMUM_SHIFT;
                    }else if (left_ind+peak_width+shift>=fifo_size){
                        spike->label = LABEL_RIGHT_LIMIT;
                    }else if((left_ind+shift)<0){
                        spike->label = LABEL_LEFT_LIMIT;
                    }else{
                        estimate_one_jitter(left_ind+shift, spike->label, &new_jitter,
                                                            extremum_channel, fifo_residuals,
                                                            catalogue_center0, catalogue_center1, catalogue_center2,
                                                            wf1_norm2, wf2_norm2, wf1_dot_wf2);
                        
                        ok = accept_tempate(left_ind+shift, spike->label, new_jitter,
                                                            fifo_residuals, sparse_mask,
                                                            catalogue_center0, catalogue_center1, catalogue_center2,
                                                            weight_per_template);
                        
                        if ((fabs(new_jitter)<fabs(jitter)) && (ok==1)){
                            jitter = new_jitter;
                            left_ind += shift;
                        }
                        
                        
                    }
                }
                // one spike!!!!!
                spike->jitter = jitter;
                spike->peak_index = left_ind - n_left;

                // unset already tested near peak
                for (int s=0; s<peak_width; ++s){
                    mask_already_tested[left_ind+s-n_span] = 1;
                }
                
                // remove prediction from residual
                shift = - (int) round(jitter);
                int_jitter = (int) ((jitter+shift) * subsample_ratio) + (subsample_ratio / 2 );
                
                for (int s=0; s<peak_width; ++s){
                    for (int c=0; c<nb_channel; ++c){
                        fifo_residuals[(left_ind+s+shift)*nb_channel + c] -= catalogue_inter_center0[subsample_ratio*wf_size*spike->label + nb_channel*(s*subsample_ratio+int_jitter) + c];
                        // fifo_residuals[(left_ind+s+shift)*nb_channel + c] -= catalogue_center0[wf_size*spike->label + nb_channel*s + c];
                        
                    }
                }
            }
            
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    
    // not parrarel : set flag already tested
    if ((chan==0) && (cluster_idx==0)){
        if ((spike->label<0) && (spike->peak_index>=0)){
            mask_already_tested[spike->peak_index-n_span] = 0;
            spike->jitter = 0.0f;
            
        }
    }

}
"""



