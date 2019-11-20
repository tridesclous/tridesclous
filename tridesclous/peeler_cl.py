"""
WIP for porting peeler to opencl.



"""

from .peeler import * 
from .peeler import _dtype_spike

try:
    import pyopencl
    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False




class Peeler_OpenCl(Peeler):
    def _initialize_before_each_segment(self, *args, **kargs):
        Peeler._initialize_before_each_segment(self, *args, **kargs)
        
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines['opencl']
        self.signalpreprocessor = SignalPreprocessor_class(self.sample_rate, self.nb_channel, self.chunksize, self.source_dtype)
        
        p = dict(self.catalogue['signal_preprocessor_params'])
        p['normalize'] = True
        p['signals_medians'] = self.catalogue['signals_medians']
        p['signals_mads'] = self.catalogue['signals_mads']
        self.signalpreprocessor.change_params(**p)
        
        
        n = self.fifo_residuals.shape[0]-self.chunksize
        assert n<self.chunksize, 'opencl kernel for fifo add not work'
        
        
        #~ self.ctx = pyopencl.create_some_context()
        #~ self.queue = pyopencl.CommandQueue(self.ctx)
        
        self.ctx = self.signalpreprocessor.ctx
        self.queue = self.signalpreprocessor.queue
        
        kernel = self.kernel%dict(chunksize=self.chunksize, nb_channel=self.nb_channel, n_span=self.n_span,
                    relative_threshold=self.relative_threshold, peak_sign={'+':1, '-':-1}[self.peak_sign],
                    n_side=self.n_side, fifo_size=self.chunksize+self.n_side, peak_width=self.catalogue['peak_width'])
        
        prg = pyopencl.Program(self.ctx, kernel)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        
        self.preprocessed_chunk = np.zeros((self.chunksize, self.nb_channel), dtype=self.internal_dtype)
        self.preprocessed_chunk_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.preprocessed_chunk)
        
        self.fifo_residuals_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_residuals)
        
        self.fifo_sum = np.zeros((self.chunksize+self.n_side,), dtype=self.internal_dtype)
        self.fifo_sum_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_sum)
        
        self.peak_bool = np.zeros((self.chunksize+self.n_side,), dtype='uint8')
        self.peak_bool_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.peak_bool)
        
        self.peak_index = np.zeros((self.chunksize+self.n_side,), dtype='int32')
        self.peak_index_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.peak_index)
        
        self.nb_peak_index = np.zeros((1), dtype='int32')
        self.nb_peak_index_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.nb_peak_index)
        
        
        #
        wf_shape = self.catalogue['centers0'].shape[1:]
        self.one_waveform = np.zeros(wf_shape, dtype='float32')
        self.one_waveform_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.one_waveform)
        
        self.catalogue_center_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.catalogue['centers0'])
        
        nb_cluster = self.catalogue['centers0'].shape[0]
        self.waveform_distance = np.zeros((nb_cluster), dtype='float32')
        self.waveform_distance_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.waveform_distance)

        #kernels links
        self.kern_add_fifo_residuals = getattr(self.opencl_prg, 'add_fifo_residuals')
        self.kern_detect_boolean_peaks = getattr(self.opencl_prg, 'detect_boolean_peaks')
        self.kern_bool_to_index = getattr(self.opencl_prg, 'bool_to_index')
        
        #~ self.kern_classify_and_align_one_spike = getattr(self.opencl_prg, 'classify_and_align_one_spike')
        #~ self.kern_make_prediction_signals = getattr(self.opencl_prg, 'make_prediction_signals')
        self.kern_waveform_distance = getattr(self.opencl_prg, 'waveform_distance')
    
    
    def process_one_chunk(self,  pos, sigs_chunk):
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        
        #note abs_head_index is smaller than pos because prepcorcessed chunk
        # is late because of local filfilt in signalpreprocessor
        if abs_head_index is  None:
            return
        
        #~ print(preprocessed_chunk_cl)
        #~ print('pos', pos, 'abs_head_index', abs_head_index)
        
        if preprocessed_chunk.shape[0]!=self.chunksize:
            self.preprocessed_chunk[:] =0
            self.preprocessed_chunk[-preprocessed_chunk.shape[0]:, :] = preprocessed_chunk
            pyopencl.enqueue_copy(self.queue,  self.preprocessed_chunk_cl, self.preprocessed_chunk)
        else:
            pyopencl.enqueue_copy(self.queue,  self.preprocessed_chunk_cl, preprocessed_chunk)
        
        
        #shift rsiruals buffer and put the new one on right side
        #~ n = self.fifo_residuals.shape[0]-preprocessed_chunk.shape[0]
        n = self.fifo_residuals.shape[0]-self.chunksize
        #~ self.fifo_residuals[:n,:] = self.fifo_residuals[-n:,:]
        #~ self.fifo_residuals[n:,:] = preprocessed_chunk
        #~ assert n<self.chunksize
        #~ print('n', n)
        #~ print(self.fifo_residuals.shape, self.chunksize, self.n_side)
        global_size = (self.chunksize, self.nb_channel)
        local_size = None
        event = self.kern_add_fifo_residuals(self.queue, global_size, local_size,
                    self.fifo_residuals_cl, self.preprocessed_chunk_cl, np.int32(n))
                    #~ self.fifo_residuals_cl, preprocessed_chunk_cl, np.int32(n))
                    
        
        
        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        #~ print('shift', shift, self.n_side)
        
        
        all_spikes = []
        while True:
            
            global_size = (self.chunksize+self.n_side, )
            local_size = None
            event = self.kern_detect_boolean_peaks(self.queue,  global_size, local_size,
                                    self.fifo_residuals_cl, self.fifo_sum_cl, self.peak_bool_cl)
            
            #~ pyopencl.enqueue_copy(self.queue,  self.peak_bool, self.peak_bool_cl)
            #~ local_peaks,  = np.nonzero(self.peak_bool)
            #~ print(pos, 'local_peaks', local_peaks, local_peaks.shape)
            #~ exit()
            
            global_size = (1, )
            local_size = None
            event = self.kern_bool_to_index(self.queue,  global_size, local_size,
                        self.peak_bool_cl, self.peak_index_cl, self.nb_peak_index_cl)
            
            #~ print('level', level)
            #DEBUG
            
            pyopencl.enqueue_copy(self.queue,  self.peak_index, self.peak_index_cl)
            pyopencl.enqueue_copy(self.queue,  self.nb_peak_index, self.nb_peak_index_cl)
            
            local_peaks = self.peak_index[:self.nb_peak_index[0]]
            
            #~ print(pos, 'local_peaks', local_peaks, local_peaks.shape)
            #~ fig, ax = plt.subplots()
            #~ pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
            #~ ax.plot(self.fifo_residuals)
            #~ ax.plot(local_peaks, self.fifo_residuals[local_peaks], ls='None', marker='o')
            #~ plt.show()
            #CL OK here
            #~ print('local_index', local_index)
            #~ exit()
            
            pyopencl.enqueue_copy(self.queue,  self.fifo_residuals, self.fifo_residuals_cl)
            
            n_ok = 0
            for i, local_peak in enumerate(local_peaks):
                spike = self.classify_and_align_one_spike(local_peak, self.fifo_residuals, self.catalogue)
                if spike.cluster_label>=0:
                    spikes = np.array([spike], dtype=_dtype_spike)
                    prediction = make_prediction_signals(spikes, self.fifo_residuals.dtype, self.fifo_residuals.shape, self.catalogue, safe=False)
                    self.fifo_residuals -= prediction
                    
                    spikes['index'] += shift
                    all_spikes.append(spikes)
                    n_ok += 1
            pyopencl.enqueue_copy(self.queue,  self.fifo_residuals_cl, self.fifo_residuals)
            
            if n_ok==0:
                bad_spikes = np.zeros(local_peaks.shape[0], dtype=_dtype_spike)
                bad_spikes['index'] = local_peaks + shift
                bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                break

        # append bad spike
        all_spikes.append(bad_spikes)
        
        #concatenate sort and count
        all_spikes = np.concatenate(all_spikes)
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
                                                __global  uchar *peak_bools){
    
        int pos = get_global_id(0);
        
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
        int pos2 = pos + n_span;
        
        uchar peak=0;
        if ((pos<n_span)||(pos>=(chunksize+n_side-n_span))){
            peak_bools[pos] = 0;
        }
        else{
            if(peak_sign==1){
                if (fifo_sum[pos2]>relative_threshold){
                    peak=1;
                    for (int i=1; i<=n_span; i++){
                        peak = peak && (fifo_sum[pos2]>fifo_sum[pos2-i]) && (fifo_sum[pos2]>=fifo_sum[pos2+i]);
                    }
                }
            }
            else if(peak_sign==-1){
                if (fifo_sum[pos2]<-relative_threshold){
                    peak=1;
                    for (int i=1; i<=n_span; i++){
                        peak = peak && (fifo_sum[pos2]<fifo_sum[pos2-i]) && (fifo_sum[pos2]<=fifo_sum[pos2+i]);
                    }
                }
            }
            peak_bools[pos2]=peak;
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

    def estimate_one_jitter(self, waveform, catalogue):
        # This line is the slower part !!!!!!
        # cluster_idx = np.argmin(np.sum(np.sum((catalogue['centers0']-waveform)**2, axis = 1), axis = 1))
        #~ print()
        #~ # replace by this (indentique but faster, a but)
        #~ t1 = time.perf_counter()
        #~ d = catalogue['centers0']-waveform[None, :, :]
        #~ d *= d
        #~ s = np.einsum('ijk->i', d) # a bit faster
        #~ cluster_idx = np.argmin(s)
        #~ t2 = time.perf_counter()
        #~ print('    np.argmin V2', t2-t1, cluster_idx)
        
        # 
        #~ t1 = time.perf_counter()
        #~ print(catalogue['centers0'].shape)
        global_size = ( int(catalogue['centers0'].shape[0]), )
        #~ print(global_size)
        local_size = None
        #~ print(waveform.shape, waveform.dtype)
        #~ print(self.one_waveform.shape, self.one_waveform.dtype)
        pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)
        event = self.kern_waveform_distance(self.queue,  global_size, local_size,
                    self.one_waveform_cl, self.catalogue_center_cl, self.waveform_distance_cl)
        pyopencl.enqueue_copy(self.queue,  self.waveform_distance, self.waveform_distance_cl)
        #~ print(self.waveform_distance)
        cluster_idx = np.argmin(self.waveform_distance)
        #~ print(self.catalogue['peak_width'])
        #~ t2 = time.perf_counter()
        #~ print('    np.argmin CL', t2-t1, cluster_idx)
        
        #~ exit()
        

        k = catalogue['cluster_labels'][cluster_idx]
        chan = catalogue['extremum_channel'][cluster_idx]
        #~ print('cluster_idx', cluster_idx, 'k', k, 'chan', chan)

        
        #~ return k, 0.

        wf0 = catalogue['centers0'][cluster_idx,: , chan]
        wf1 = catalogue['centers1'][cluster_idx,: , chan]
        wf2 = catalogue['centers2'][cluster_idx,: , chan]
        wf = waveform[:, chan]
        #~ print()
        #~ print(wf0.shape, wf.shape)
        
        
        #it is  precompute that at init speedup 10%!!! yeah
        #~ wf1_norm2 = wf1.dot(wf1)
        #~ wf2_norm2 = wf2.dot(wf2)
        #~ wf1_dot_wf2 = wf1.dot(wf2)
        wf1_norm2= catalogue['wf1_norm2'][cluster_idx]
        wf2_norm2 = catalogue['wf2_norm2'][cluster_idx]
        wf1_dot_wf2 = catalogue['wf1_dot_wf2'][cluster_idx]
        
        
        h = wf - wf0
        h0_norm2 = h.dot(h)
        h_dot_wf1 = h.dot(wf1)
        jitter0 = h_dot_wf1/wf1_norm2
        h1_norm2 = np.sum((h-jitter0*wf1)**2)
        #~ print(h0_norm2, h1_norm2)
        #~ print(h0_norm2 > h1_norm2)
        
        
        
        if h0_norm2 > h1_norm2:
            #order 1 is better than order 0
            h_dot_wf2 = np.dot(h,wf2)
            rss_first = -2*h_dot_wf1 + 2*jitter0*(wf1_norm2 - h_dot_wf2) + 3*jitter0**2*wf1_dot_wf2 + jitter0**3*wf2_norm2
            rss_second = 2*(wf1_norm2 - h_dot_wf2) + 6*jitter0*wf1_dot_wf2 + 3*jitter0**2*wf2_norm2
            jitter1 = jitter0 - rss_first/rss_second
            #~ h2_norm2 = np.sum((h-jitter1*wf1-jitter1**2/2*wf2)**2)
            #~ if h1_norm2 <= h2_norm2:
                #when order 2 is worse than order 1
                #~ jitter1 = jitter0
        else:
            jitter1 = 0.
        #~ print('jitter1', jitter1)
        #~ return k, 0.
        
        #~ print(np.sum(wf**2), np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2))
        #~ print(np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2))
        #~ return k, jitter1

        
        if np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2):
            #prediction should be smaller than original (which have noise)
            return k, jitter1
        else:
            #otherwise the prediction is bad
            #~ print('bad prediction')
            return LABEL_UNCLASSIFIED, 0.

