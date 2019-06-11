"""
Here 2 version for peakdetector.


"""

import numpy as np

#~ from pyacq.core.stream.ringbuffer import RingBuffer
from .tools import FifoBuffer

try:
    import pyopencl
    mf = pyopencl.mem_flags
    HAVE_PYOPENCL = True
except ImportError:
    HAVE_PYOPENCL = False


def detect_peaks_in_chunk(sig, k, thresh, peak_sign):
    sig = sig.copy()
    
    if peak_sign == '+':
        sig[sig<thresh] = 0.
        #~ sig[sig<3.] = 0.
    else:
        sig[sig>-thresh] = 0.
        #~ sig[sig>-3.] = 0.

    if sig.shape[1]>1:
        sum_rectified = np.sum(sig, axis=1)
    else:
        sum_rectified = sig[:,0]
    
    ind_peaks = _detect_peaks_in_rectified(sum_rectified, k, thresh, peak_sign)
    
    return ind_peaks


def _detect_peaks_in_rectified(sig_rectified, k, thresh, peak_sign):
    sig_center = sig_rectified[k:-k]
    if peak_sign == '+':
        peaks = sig_center>thresh
        for i in range(k):
            peaks &= sig_center>sig_rectified[i:i+sig_center.size]
            peaks &= sig_center>=sig_rectified[k+i+1:k+i+1+sig_center.size]
    elif peak_sign == '-':
        peaks = sig_center<-thresh
        for i in range(k):
            peaks &= sig_center<sig_rectified[i:i+sig_center.size]
            peaks &= sig_center<=sig_rectified[k+i+1:k+i+1+sig_center.size]
    
    ind_peaks,  = np.nonzero(peaks)

    ind_peaks += k
    return ind_peaks


class PeakDetectorEngine_Numpy:
    def __init__(self, sample_rate, nb_channel, chunksize, dtype,):
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.dtype = dtype
        
        self.n_peak = 0
        
    def process_data(self, pos, newbuf):
        newbuf = newbuf.copy()
        
        if self.peak_sign == '+':
            newbuf[newbuf<self.relative_threshold] = 0.
            #~ newbuf[newbuf<3.] = 0.
            
        else:
            newbuf[newbuf>-self.relative_threshold] = 0.
            #~ newbuf[newbuf>-3.] = 0.

        if self.nb_channel>1:
            sum_rectified = np.sum(newbuf, axis=1)
        else:
            sum_rectified = newbuf[:,0]
        
        #~ self.ring_sum.new_chunk(sum_rectified, index=pos)
        self.fifo_sum_rectified.new_chunk(sum_rectified, pos)
        
        k = self.n_span
        if pos-(newbuf.shape[0]+2*k)<0:
            # the very first buffer is sacrified because of peak span
            return None, None
        
        #~ sig = self.ring_sum.get_data(pos-(newbuf.shape[0]+2*k), pos)
        sig_rectified = self.fifo_sum_rectified.get_data(pos-(newbuf.shape[0]+2*k), pos)
        
        ind_peaks = _detect_peaks_in_rectified(sig_rectified, k, self.relative_threshold, self.peak_sign)
        
        if ind_peaks.size>0:
            ind_peaks = ind_peaks + pos - newbuf.shape[0] -2*k
            self.n_peak += ind_peaks.size
            return self.n_peak, ind_peaks

        return None, None
        #~ sig_center = sig[k:-k]
        #~ if self.peak_sign == '+':
            #~ peaks = sig_center>self.relative_threshold
            #~ for i in range(k):
                #~ peaks &= sig_center>sig[i:i+sig_center.size]
                #~ peaks &= sig_center>=sig[k+i+1:k+i+1+sig_center.size]
        #~ elif self.peak_sign == '-':
            #~ peaks = sig_center<-self.relative_threshold
            #~ for i in range(k):
                #~ peaks &= sig_center<sig[i:i+sig_center.size]
                #~ peaks &= sig_center<=sig[k+i+1:k+i+1+sig_center.size]
        
        #~ ind_peaks,  = np.where(peaks)
        
        #~ if ind_peaks.size>0:
            #~ ind_peaks = ind_peaks + pos - newbuf.shape[0] - k
            #~ self.n_peak += ind_peaks.size
            #~ return self.n_peak, ind_peaks
        
        #~ return None, None
        
    def change_params(self, peak_sign=None, relative_threshold=None, peak_span_ms=None, peak_span=None):
        self.peak_sign = peak_sign
        self.relative_threshold = relative_threshold
        
        
        if peak_span_ms is None:
            # kept for compatibility with previous version
            assert peak_span is not None
            peak_span_ms = peak_span * 1000.
        
        self.peak_span_ms = peak_span_ms
        
        self.n_span = int(self.sample_rate * self.peak_span_ms / 1000.)//2
        #~ print('self.n_span', self.n_span)
        self.n_span = max(1, self.n_span)
        
        #~ self.ring_sum = RingBuffer((self.chunksize*2,), self.dtype, double=True)
        self.fifo_sum_rectified = FifoBuffer((self.chunksize*2,), self.dtype)
        
        


class PeakDetectorEngine_OpenCL:
    """
    Same as PeakDetectorEngine but implemented with OpenCl.
    With a strong GPU  perf a little bit better than on CPU.
    For standard GPU PeakDetectorEngine implemented with numpy is faster.
    I wasted my time here....
    """
    def __init__(self, sample_rate, nb_channel, chunksize, dtype,):
        assert HAVE_PYOPENCL
        
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.dtype = np.dtype(dtype)
        
        self.n_peak = 0

        self.ctx = pyopencl.create_some_context(interactive=False)
        #~ print(self.ctx)
        #TODO : add arguments gpu_platform_index/gpu_device_index
        #self.devices =  [pyopencl.get_platforms()[self.gpu_platform_index].get_devices()[self.gpu_device_index] ]
        #self.ctx = pyopencl.Context(self.devices)        
        self.queue = pyopencl.CommandQueue(self.ctx)

    def process_data(self, pos, newbuf):
        #~ assert newbuf.shape[0] ==self.chunksize
        if newbuf.shape[0] <self.chunksize:
            newbuf2 = np.zeros((self.chunksize, self.nb_channel), dtype=self.dtype)
            newbuf2[-newbuf.shape[0]:, :] = newbuf
            newbuf = newbuf2

        if not newbuf.flags['C_CONTIGUOUS']:
            newbuf = newbuf.copy()

        pyopencl.enqueue_copy(self.queue,  self.sigs_cl, newbuf)
        event = self.kern_detect_peaks(self.queue,  (self.chunksize,), (self.max_wg_size,),
                                self.sigs_cl, self.ring_sum_cl, self.peaks_cl)
        event.wait()
        
        if pos-(newbuf.shape[0]+2*self.n_span)<0:
            # the very first buffer is sacrified because of peak span
            return None, None
        
        pyopencl.enqueue_copy(self.queue,  self.peaks, self.peaks_cl)
        ind_peaks,  = np.nonzero(self.peaks)
        
        if ind_peaks.size>0:
            #~ ind_peaks += pos - newbuf.shape[0] - self.n_span
            ind_peaks += pos - self.chunksize - self.n_span
            self.n_peak += ind_peaks.size
            #~ peaks = np.zeros(ind_peaks.size, dtype = [('index', 'int64'), ('code', 'int64')])
            #~ peaks['index'] = ind_peaks
            #~ return self.n_peak, peaks
            return self.n_peak, ind_peaks
        
        return None, None
        
    def change_params(self, peak_sign=None, relative_threshold=None, peak_span_ms=None, peak_span=None):
        self.peak_sign = peak_sign
        self.relative_threshold = relative_threshold

        if peak_span_ms is None:
            # kept for compatibility with previous version
            assert peak_span is not None
            peak_span_ms = peak_span *1000.
        
        self.peak_span_ms = peak_span_ms
        self.n_span = int(self.sample_rate * self.peak_span_ms / 1000.)//2
        self.n_span = max(1, self.n_span)
        
        chunksize2=self.chunksize+2*self.n_span
        
        self.sum_rectified = np.zeros((self.chunksize), dtype=self.dtype)
        self.peaks = np.zeros((self.chunksize), dtype='uint8')
        ring_sum = np.zeros((chunksize2), dtype=self.dtype)
        
        #GPU buffers
        self.sigs_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=self.nb_channel*self.chunksize*self.dtype.itemsize)
        
        self.ring_sum_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=ring_sum)
        self.peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.peaks)
        
        kernel = self.kernel%dict(chunksize=self.chunksize, nb_channel=self.nb_channel, n_span=self.n_span,
                    relative_threshold=relative_threshold, peak_sign={'+':1, '-':-1}[self.peak_sign])
        
        prg = pyopencl.Program(self.ctx, kernel)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)


        self.kern_detect_peaks = getattr(self.opencl_prg, 'detect_peaks')

    kernel = """
    #define chunksize %(chunksize)d
    #define n_span %(n_span)d
    #define nb_channel %(nb_channel)d
    #define relative_threshold %(relative_threshold)d
    #define peak_sign %(peak_sign)d
    
    __kernel void detect_peaks(__global  float *sigs,
                                                __global  float *ring_sum,
                                                __global  uchar *peaks){
    
        int pos = get_global_id(0);
        
        int idx;
        float v;

        // roll_ring_sum and wait for friends
        if (pos<(n_span*2)){
            ring_sum[pos] = ring_sum[pos+chunksize];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        
        // sum all channels
        float sum=0;
        for (int chan=0; chan<nb_channel; chan++){
            idx = pos*nb_channel + chan;
            
            v = sigs[idx];
            
            //retify signals
            if(peak_sign==1){
                if (v<relative_threshold){v=0;}
            }
            else if(peak_sign==-1){
                if (v>-relative_threshold){v=0;}
            }
            
            sum = sum + v;
            
        }
        ring_sum[pos+2*n_span] = sum;
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        
        // peaks span
        int pos2 = pos + n_span;
        
        uchar peak=0;
        
        if(peak_sign==1){
            if (ring_sum[pos2]>relative_threshold){
                peak=1;
                for (int i=1; i<=n_span; i++){
                    peak = peak && (ring_sum[pos2]>ring_sum[pos2-i]) && (ring_sum[pos2]>=ring_sum[pos2+i]);
                }
            }
        }
        else if(peak_sign==-1){
            if (ring_sum[pos2]<-relative_threshold){
                peak=1;
                for (int i=1; i<=n_span; i++){
                    peak = peak && (ring_sum[pos2]<ring_sum[pos2-i]) && (ring_sum[pos2]<=ring_sum[pos2+i]);
                }
            }
        }
        peaks[pos]=peak;
    
    }
    
    """


peakdetector_engines = { 'numpy' : PeakDetectorEngine_Numpy, 'opencl' : PeakDetectorEngine_OpenCL}


