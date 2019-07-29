import time
import numpy as np

from .peeler_engine_classic import PeelerEngineClassic
from .peeler_tools import *
from .peeler_tools import _dtype_spike
from .tools import make_color_dict
from . import signalpreprocessor

from .signalpreprocessor import signalpreprocessor_engines, processor_kernel
from .peakdetector import  detect_peaks_in_chunk



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
        p.pop('signalpreprocessor_engine')
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
        
        
        kernel = self.kernel%dict(chunksize=self.chunksize, nb_channel=self.nb_channel, n_span=self.n_span,
                    relative_threshold=self.relative_threshold, peak_sign={'+':1, '-':-1}[self.peak_sign],
                    n_side=self.n_side, fifo_size=self.chunksize+self.n_side, peak_width=self.catalogue['peak_width'])
        
        prg = pyopencl.Program(self.ctx, kernel)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        
        #~ self.preprocessed_chunk = np.zeros((self.chunksize, self.nb_channel), dtype=self.internal_dtype)
        #~ self.preprocessed_chunk_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.preprocessed_chunk)
        
        self.fifo_residuals_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_residuals)
        
        self.fifo_sum = np.zeros((self.chunksize+self.n_side,), dtype=self.internal_dtype)
        self.fifo_sum_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_sum)
        
        self.local_peaks_mask = np.zeros((self.chunksize + self.n_side - 2 * self.n_span), dtype='uint8')
        self.local_peaks_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.local_peaks_mask)

        self.mask_already_tested = np.ones((self.chunksize + self.n_side - 2 * self.n_span), dtype='uint8')
        self.mask_already_tested_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.local_peaks_mask)
        
        
        
        #~ self.peak_index = np.zeros((self.chunksize+self.n_side,), dtype='int32')
        #~ self.peak_index_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.peak_index)
        
        #~ self.nb_peak_index = np.zeros((1), dtype='int32')
        #~ self.nb_peak_index_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.nb_peak_index)

        self.peak_counter_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=4) # int32
        self.peak_index_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=4) # int32
        
        
        
        #
        #~ wf_shape = self.catalogue['centers0'].shape[1:]
        #~ self.one_waveform = np.zeros(wf_shape, dtype='float32')
        #~ self.one_waveform_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.one_waveform)
        
        #~ self.catalogue_center_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers0'])
        
        #~ nb_cluster = self.catalogue['centers0'].shape[0]
        #~ self.waveform_distance = np.zeros((nb_cluster), dtype='float32')
        #~ self.waveform_distance_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.waveform_distance)

        #kernels links
        self.kern_add_fifo_residuals = getattr(self.opencl_prg, 'add_fifo_residuals')
        self.kern_detect_boolean_peaks = getattr(self.opencl_prg, 'detect_boolean_peaks')
        self.kern_select_next_peak = getattr(self.opencl_prg, 'select_next_peak')
        
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
        local_size = None
        event = self.kern_add_fifo_residuals(self.queue, global_size, local_size,
                    self.fifo_residuals_cl, sp.output_backward_cl, np.int32(n))

        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        # TODO remove from peak the very begining of the signal because of border filtering effects

        
     
        good_spikes = []
        #~ already_tested = []
        
        # negative mask 1: not tested 0: already tested
        #~ mask_already_tested = np.ones(self.fifo_residuals.shape[0] - 2 * self.n_span, dtype='bool')
        
        # TODO do it in the kernel
        event = pyopencl.enqueue_copy(self.queue,  self.mask_already_tested_cl, self.mask_already_tested)
        
        #~ local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)

        global_size = (self.chunksize+self.n_side,  )
        local_size = (min(self.max_wg_size, self.chunksize+self.n_side), )
        event = self.kern_detect_boolean_peaks(self.queue,  global_size, local_size,
                                self.fifo_residuals_cl, self.fifo_sum_cl, self.local_peaks_mask_cl)
        
        
        # debug
        event = pyopencl.enqueue_copy(self.queue,  self.local_peaks_mask, self.local_peaks_mask_cl)
        event = pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
        self.fifo_residuals = self.fifo_residuals.copy()
        local_peaks_mask = self.local_peaks_mask
        mask_already_tested = self.mask_already_tested.copy()
        # end debug

        #~ print('yep')
        #~ exit()
        
        
        
        #~ print('sum(local_peaks_mask)', np.sum(local_peaks_mask))
        
        t1 = time.perf_counter()
        while True:
            
            global_size = (self.chunksize+self.n_side - 2*self.n_span,  )
            local_size = (min(self.max_wg_size, self.chunksize+self.n_side), )
            event = self.kern_select_next_peak(self.queue,  global_size, local_size,
                                    self.local_peaks_mask_cl, self.mask_already_tested_cl, self.peak_counter_cl, self.peak_index_cl)
            
            
            global_size = (self.chunksize+self.n_side - 2*self.n_span,  )
            local_size = (min(self.max_wg_size, self.chunksize+self.n_side), )
            
            self.kern_classify_and_align_one_spike(self.queue,  global_size, local_size,
            
            
            n_ok = 0
            for local_ind in local_peaks_indexes:
                #~ print('    local_peak', local_peak, 'i', i)
                t3 = time.perf_counter()
                spike = self.classify_and_align_one_spike(local_ind, self.fifo_residuals, self.catalogue)
                t4 = time.perf_counter()
                #~ print('    classify_and_align_one_spike', (t4-t3)*1000., spike.cluster_label)
                
                #~ exit()
                
                if spike.cluster_label>=0:
                    t3 = time.perf_counter()
                    #~ print('     >>spike.index', spike.index, spike.cluster_label, 'abs index', spike.index+shift)
                    
                    
                    #~ spikes = np.array([spike], dtype=_dtype_spike)
                    #~ prediction = make_prediction_signals(spikes, self.fifo_residuals.dtype, self.fifo_residuals.shape, self.catalogue, safe=False)
                    #~ self.fifo_residuals -= prediction
                    #~ spikes['index'] += shift
                    #~ good_spikes.append(spikes)
                    #~ n_ok += 1
                    
                    # substract one spike
                    pos, pred = make_prediction_one_spike(spike.index, spike.cluster_label, spike.jitter, self.fifo_residuals.dtype, self.catalogue)
                    self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
                    
                    # append
                    spikes = np.array([spike], dtype=_dtype_spike)
                    spikes['index'] += shift
                    good_spikes.append(spikes)
                    n_ok += 1
                                    
                    # recompute peak in neiborhood
                    # here indexing is tricky 
                    # sl1 : we need n_pan more in each side
                    # sl2: we need a shift of n_span because smaler shape
                    sl1 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1 + self.n_span)
                    sl2 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1- self.n_span)
                    local_peaks_mask[sl2] = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals[sl1, :])
                    
                    
                    #~ print('    already_tested before', already_tested)
                    #~ already_tested = [ind for ind in already_tested if np.abs(spike.index-ind)>self.peak_width]
                    
                    # set neighboor untested
                    mask_already_tested[local_ind - self.peak_width - self.n_span:local_ind + self.peak_width - self.n_span] = True

                    t4 = time.perf_counter()
                    #~ print('    make_prediction_signals and sub', (t4-t3)*1000.)
                    
                    #~ print('    already_tested new deal', already_tested)
                else:
                    # set peak tested
                    #~ print(mask_already_tested.shape)
                    #~ print(self.fifo_residuals.shape)
                    #~ print(self.n_span)
                    mask_already_tested[local_ind - self.n_span] = False
                    #~ print('already tested', local_ind)
                    #~ already_tested.append(local_peak)
            
            if n_ok==0:
                # no peak can be labeled
                # reserve bad spikes on the right limit for next time

                #~ local_peaks_indexes = local_peaks_indexes[local_peaks_indexes<(self.chunksize+self.n_span)]
                #~ bad_spikes = np.zeros(local_peaks_indexes.shape[0], dtype=_dtype_spike)
                #~ bad_spikes['index'] = local_peaks_indexes + shift
                #~ bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                
                nolabel_indexes, = np.nonzero(~mask_already_tested)
                nolabel_indexes += self.n_span
                nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
                bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
                bad_spikes['index'] = nolabel_indexes + shift
                bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                
                break
        
        t2 = time.perf_counter()
        #~ print('LOOP classify_and_align_one_spike', (t2-t1)*1000)
        
        
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





    def _OLD_process_one_chunk(self,  pos, sigs_chunk):
        
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)

        if preprocessed_chunk.shape[0]!=self.chunksize:
            self.preprocessed_chunk[:] =0
            self.preprocessed_chunk[-preprocessed_chunk.shape[0]:, :] = preprocessed_chunk
            pyopencl.enqueue_copy(self.queue,  self.preprocessed_chunk_cl, self.preprocessed_chunk)
        else:
            pyopencl.enqueue_copy(self.queue,  self.preprocessed_chunk_cl, preprocessed_chunk)
        
        
        
        #note abs_head_index is smaller than pos because prepcorcessed chunk
        # is late because of local filfilt in signalpreprocessor
        
        #shift rsiruals buffer and put the new one on right side
        fifo_roll_size = self.fifo_residuals.shape[0]-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_residuals.shape[0]:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk

        
        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        # TODO remove from peak the very begining of the signal because of border filtering effects
        
        good_spikes = []
        all_ready_tested = []
        while True:
            #detect peaks
            t3 = time.perf_counter()
            local_peaks = detect_peaks_in_chunk(self.fifo_residuals, self.n_span, self.relative_threshold, self.peak_sign)
            t4 = time.perf_counter()
            
            if len(all_ready_tested)>0:
                local_peaks_to_check = local_peaks[~np.in1d(local_peaks, all_ready_tested)]
            else:
                local_peaks_to_check = local_peaks
            
            n_ok = 0
            for i, local_peak in enumerate(local_peaks_to_check):
                spike = self.classify_and_align_one_spike(local_peak, self.fifo_residuals, self.catalogue)
                
                if spike.cluster_label>=0:
                    spikes = np.array([spike], dtype=_dtype_spike)
                    prediction = make_prediction_signals(spikes, self.fifo_residuals.dtype, self.fifo_residuals.shape, self.catalogue, safe=False)
                    self.fifo_residuals -= prediction
                    spikes['index'] += shift
                    good_spikes.append(spikes)
                    n_ok += 1
                    
                    all_ready_tested = [ind for ind in all_ready_tested if np.abs(spike.index-ind)>self.peak_width]
                else:
                    all_ready_tested.append(local_peak)
            
            if n_ok==0:
                # no peak can be labeled
                # reserve bad spikes on the right limit for next time
                local_peaks = local_peaks[local_peaks<(self.chunksize+self.n_span)]
                bad_spikes = np.zeros(local_peaks.shape[0], dtype=_dtype_spike)
                bad_spikes['index'] = local_peaks + shift
                bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                break
        
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
        
        return abs_head_index, preprocessed_chunk, self.total_spike, all_spikes

    kernel = """
    #define chunksize %(chunksize)d
    #define n_span %(n_span)d
    #define nb_channel %(nb_channel)d
    #define relative_threshold %(relative_threshold)d
    #define peak_sign %(peak_sign)d
    #define n_side %(n_side)d
    #define fifo_size %(fifo_size)d
    #define peak_width %(peak_width)d
    
    
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
        float sum=0;
        for (int chan=0; chan<nb_channel; chan++){
            idx = pos*nb_channel + chan;
            
            v = fifo_residuals[idx];
            
            //retify signals
            if(peak_sign==1){
                if (v<relative_threshold){v=0;}
            }
            else if(peak_sign==-1){
                if (v>-relative_threshold){v=0;}
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
                                                    __global int *peak_counter, __global int *peak_index,){
        int pos = get_global_id(0); //fifo_residuals.shape[0] - 2 * nspan
        
        if (pos == 0){
            peak_counter = 0;
            peak_index[0] = -1;
            
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        if (local_peaks_mask[pos+n_span] == 1) && (mask_already_tested[pos+n_span] == 1){
            total = atomic_add(peak_counter); //TODO read doc
            if total ==1{
                peak_index[0] = pos + n_span;
            }
        }
    }
    
    __kernel void bool_to_index(__global  uchar *peak_bool, __global int *peak_index, __global int *nb_peak_index){
        
        
        int n=0;
        
        for (int pos=0; pos<fifo_size; pos++){
            if (peak_bool[pos]==1){
                peak_index[n] = pos;
                n +=1;
            }
        }
        nb_peak_index[0] = n;
        
    }
    
    
    __kernel void classify_and_align_one_spike(){
    //__global  float *fifo_residuals, __global int *peak_index, __global int *nb_peak_index, 
    //                            __global  float ){
    
    }

    __kernel void make_prediction_signals(){
    
    }
    
    
    __kernel void waveform_distance(__global  float *one_waveform, __global  float *catalogue_center, __global  float *waveform_distance){
        
        int cluster_idx = get_global_id(0);
        
        float s = 0;
        float d;
        
        int total = nb_channel*peak_width;
        
        for (int c=0; c<total; c++){
            d = one_waveform[c] - catalogue_center[cluster_idx*total+c];
            s = s + d*d;
        }
        
        waveform_distance[cluster_idx] = s;
    }
    
    """


