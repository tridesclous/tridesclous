"""
Here 2 version for peakdetector.


"""

import numpy as np

import sklearn.metrics.pairwise

#~ from pyacq.core.stream.ringbuffer import RingBuffer
from .tools import FifoBuffer

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags


def detect_peaks_in_chunk(sig, n_span, thresh, peak_sign, spatial_matrix=None):
    sum_rectified = make_sum_rectified(sig, thresh, peak_sign, spatial_matrix)
    ind_peaks = detect_peaks_in_rectified(sum_rectified, n_span, thresh, peak_sign)
    return ind_peaks


def make_sum_rectified(sig, thresh, peak_sign, spatial_matrix):
    if spatial_matrix is None:
        sig = sig.copy()
    else:
        sig = np.dot(sig, spatial_matrix)
    
    
    if peak_sign == '+':
        sig[sig<thresh] = 0.
    else:
        sig[sig>-thresh] = 0.
    
    if sig.shape[1]>1:
        sum_rectified = np.sum(sig, axis=1)
    else:
        sum_rectified = sig[:,0]
    
    return sum_rectified
    

def detect_peaks_in_rectified(sig_rectified, n_span, thresh, peak_sign):
    sig_center = sig_rectified[n_span:-n_span]
    if peak_sign == '+':
        peaks = sig_center>thresh
        for i in range(n_span):
            peaks &= sig_center>sig_rectified[i:i+sig_center.size]
            peaks &= sig_center>=sig_rectified[n_span+i+1:n_span+i+1+sig_center.size]
    elif peak_sign == '-':
        peaks = sig_center<-thresh
        for i in range(n_span):
            peaks &= sig_center<sig_rectified[i:i+sig_center.size]
            peaks &= sig_center<=sig_rectified[n_span+i+1:n_span+i+1+sig_center.size]
    
    ind_peaks,  = np.nonzero(peaks)

    ind_peaks += n_span
    return ind_peaks



class PeakDetectorEngine_Numpy:
    def __init__(self, sample_rate, nb_channel, chunksize, dtype, geometry):
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.dtype = dtype
        self.geometry = geometry # 2D array (nb_channel, 2 or 3) 
        
        self.n_peak = 0
        
    def process_data(self, pos, newbuf):
        # this is used by catalogue constructor
        # here the fifo is only the rectified sum
        
        sum_rectified = make_sum_rectified(newbuf, self.relative_threshold, self.peak_sign, self.spatial_matrix)
        self.fifo_sum_rectified.new_chunk(sum_rectified, pos)
        
        if pos-(newbuf.shape[0]+2*self.n_span)<0:
            # the very first buffer is sacrified because of peak span
            return None, None
        
        #~ sig = self.ring_sum.get_data(pos-(newbuf.shape[0]+2*k), pos)
        sig_rectified = self.fifo_sum_rectified.get_data(pos-(newbuf.shape[0]+2*self.n_span), pos)
        
        ind_peaks = detect_peaks_in_rectified(sig_rectified, self.n_span, self.relative_threshold, self.peak_sign)
        
        if ind_peaks.size>0:
            ind_peaks = ind_peaks + pos - newbuf.shape[0] -2*self.n_span
            self.n_peak += ind_peaks.size
            return self.n_peak, ind_peaks

        return None, None
    
    def detect_peaks_in_chunk(self, fifo_residuals):
        # this is used by peeler which handle externaly
        # a fifo residual
        sum_rectified = make_sum_rectified(fifo_residuals, self.relative_threshold, self.peak_sign, self.spatial_matrix)
        ind_peaks = detect_peaks_in_rectified(sum_rectified, self.n_span, self.relative_threshold, self.peak_sign)
        return ind_peaks        
    
    
    def change_params(self, peak_sign=None, relative_threshold=None,
                                            peak_span_ms=None, peak_span=None,
                                            adjacency_radius_um=None,
                                            ):
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
        
        self.adjacency_radius_um = adjacency_radius_um
        if self.adjacency_radius_um is None or self.adjacency_radius_um <= 0.:
            self.spatial_matrix = None
        else:
            d = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
            self.spatial_matrix = np.exp(-d/self.adjacency_radius_um)
            
            # make it sparse
            self.spatial_matrix[self.spatial_matrix<0.01] = 0.
            ## self.spatial_matrix = self.spatial_matrix / np.sum(self.spatial_matrix, axis=0)[None, :] ## BAD IDEA this is worst
            
            #~ print(np.sum(self.spatial_matrix, axis=0))
                    
        #~ self.ring_sum = RingBuffer((self.chunksize*2,), self.dtype, double=True)
        self.fifo_sum_rectified = FifoBuffer((self.chunksize*2,), self.dtype)
        
        


class PeakDetectorEngine_OpenCL:
    """
    Same as PeakDetectorEngine but implemented with OpenCl.
    With a strong GPU  perf a little bit better than on CPU.
    For standard GPU PeakDetectorEngine implemented with numpy is faster.
    I wasted my time here....
    """
    def __init__(self, sample_rate, nb_channel, chunksize, dtype, geometry):
        assert HAVE_PYOPENCL
        
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.dtype = np.dtype(dtype)
        self.geometry = geometry
        
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


