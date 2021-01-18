"""
Here several method and implementation for peakdetector.

  * method = "threshold_and_sum"
     very fast, no spatial information
    2 implementations:
      * numpy
      * opencl
  * method = "spatiotemporal_extrema"
     detect local spatiotemporal extrema
     quite slow
     2 implementations:
       * numpy
       * opencl


"""
import time

import numpy as np

import sklearn.metrics.pairwise

#~ from pyacq.core.stream.ringbuffer import RingBuffer
from .tools import FifoBuffer

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_get_mask_spatiotemporal_peaks
except ImportError:
    HAVE_NUMBA = False





def detect_peaks_in_chunk(sig, n_span, thresh, peak_sign, spatial_smooth_kernel=None):
    sum_rectified = make_sum_rectified(sig, thresh, peak_sign, spatial_smooth_kernel)
    mask_peaks = detect_peaks_in_rectified(sum_rectified, n_span, thresh, peak_sign)
    time_ind_peaks,  = np.nonzero(mask_peaks)
    time_ind_peaks += n_span
    return time_ind_peaks



class BasePeakDetector:
    offline_function = None

    def __init__(self, sample_rate, nb_channel, chunksize, dtype, geometry):
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.dtype = np.dtype(dtype)
        self.geometry = geometry # 2D array (nb_channel, 2 or 3) 

    def change_params(self, peak_sign=None, relative_threshold=None,
                                            peak_span_ms=None, peak_span=None,
                                            smooth_radius_um=None,
                                            adjacency_radius_um=None,
                                            cl_platform_index=None, # only for CL
                                            cl_device_index=None):
        self.peak_sign = peak_sign
        self.relative_threshold = relative_threshold
        
        #~ print('peak_span_ms', peak_span_ms)
        if peak_span_ms is None:
            # kept for compatibility with previous version
            assert peak_span is not None
            peak_span_ms = peak_span * 1000.
        
        self.peak_span_ms = peak_span_ms
        
        self.adjacency_radius_um = adjacency_radius_um
        
        self.n_span = int(self.sample_rate * self.peak_span_ms / 1000.)//2
        #~ print('self.n_span', self.n_span)
        self.n_span = max(1, self.n_span)
        #~ print('self.n_span', self.n_span)
        
        self.smooth_radius_um = smooth_radius_um

        if self.smooth_radius_um is None or self.smooth_radius_um <= 0.:
            self.spatial_smooth_kernel = None
        else:
            d = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
            self.spatial_smooth_kernel = np.exp(-d/self.smooth_radius_um)
            # make it sparse
            self.spatial_smooth_kernel[self.spatial_smooth_kernel<0.01] = 0.
        
        
        self.cl_platform_index = cl_platform_index
        self.cl_device_index = cl_device_index
    

    def process_buffer(self, data):
        # used for offline processing when parralisation  is possible
        raise(NotImplmentedError)
    
    def process_buffer_stream(self, pos, newbuf):
        raise(NotImplmentedError)
    
    def initialize_stream(self):
        # must be for each new segment when index 
        # start back
        raise(NotImplmentedError)
    
    #~ def _after_params(self):
        #~ raise(NotImplmentedError)



def make_sum_rectified(sig, thresh, peak_sign, spatial_smooth_kernel):
    if spatial_smooth_kernel is None:
        sig = sig.copy()
    else:
        sig = np.dot(sig, spatial_smooth_kernel)
    
    
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
        mask_peaks = sig_center>thresh
        for i in range(n_span):
            mask_peaks &= sig_center>sig_rectified[i:i+sig_center.size]
            mask_peaks &= sig_center>=sig_rectified[n_span+i+1:n_span+i+1+sig_center.size]
    elif peak_sign == '-':
        mask_peaks = sig_center<-thresh
        for i in range(n_span):
            mask_peaks &= sig_center<sig_rectified[i:i+sig_center.size]
            mask_peaks &= sig_center<=sig_rectified[n_span+i+1:n_span+i+1+sig_center.size]
    return mask_peaks
    
    #~ time_ind_peaks,  = np.nonzero(mask_peaks)
    #~ time_ind_peaks += n_span
    #~ return time_ind_peaks



class PeakDetectorGlobalNumpy(BasePeakDetector):
    def process_buffer(self, data):
        sig_rectified = make_sum_rectified(data, self.relative_threshold, self.peak_sign, self.spatial_smooth_kernel)
        
        mask_peaks = detect_peaks_in_rectified(sig_rectified, self.n_span, self.relative_threshold, self.peak_sign)
        time_ind_peaks,  = np.nonzero(mask_peaks)
        time_ind_peaks += self.n_span
        chan_ind_peaks = None # not in this method
        peak_val_peaks = None  # not in this method
        
        return time_ind_peaks, chan_ind_peaks, peak_val_peaks
        


    def process_buffer_stream(self, pos, newbuf):
        # this is used by catalogue constructor
        # here the fifo is only the rectified sum
        
        sum_rectified = make_sum_rectified(newbuf, self.relative_threshold, self.peak_sign, self.spatial_smooth_kernel)
        self.fifo_sum_rectified.new_chunk(sum_rectified, pos)
        
        #~ if pos-(newbuf.shape[0]+2*self.n_span)<0:
            # the very first buffer is sacrified because of peak span
            #~ return None, None
        
        #~ sig = self.ring_sum.get_data(pos-(newbuf.shape[0]+2*k), pos)
        sig_rectified = self.fifo_sum_rectified.get_data(pos-(newbuf.shape[0]+2*self.n_span), pos)
        
        mask_peaks = detect_peaks_in_rectified(sig_rectified, self.n_span, self.relative_threshold, self.peak_sign)
        time_ind_peaks,  = np.nonzero(mask_peaks)
        time_ind_peaks += self.n_span
        
        chan_ind_peaks = None # not in this method
        
        if time_ind_peaks.size>0:
            time_ind_peaks = time_ind_peaks + pos - newbuf.shape[0] -2*self.n_span
            return time_ind_peaks, chan_ind_peaks, None

        return None, None, None
    
    def get_mask_peaks_in_chunk(self, fifo_residuals):
        # this is used by peeler which handle externaly
        # a fifo residual
        sum_rectified = make_sum_rectified(fifo_residuals, self.relative_threshold, self.peak_sign, self.spatial_smooth_kernel)
        mask_peaks = detect_peaks_in_rectified(sum_rectified, self.n_span, self.relative_threshold, self.peak_sign)
        return mask_peaks        
    
    def change_params(self, **kargs):
        BasePeakDetector.change_params(self,  **kargs)
        self.fifo_sum_rectified = FifoBuffer((self.chunksize*2,), self.dtype)
    
    def initialize_stream(self):
        self.fifo_sum_rectified = FifoBuffer((self.chunksize*2,), self.dtype)

class PeakDetectorGlobalOpenCL(BasePeakDetector, OpenCL_Helper):
    def process_buffer(self, data):
        #TODO
        raise(NotImplmentedError)
    
    def process_buffer_stream(self, pos, newbuf):
        if newbuf.shape[0] <self.chunksize:
            newbuf2 = np.zeros((self.chunksize, self.nb_channel), dtype=self.dtype)
            newbuf2[-newbuf.shape[0]:, :] = newbuf
            newbuf = newbuf2

        if not newbuf.flags['C_CONTIGUOUS']:
            newbuf = newbuf.copy()

        pyopencl.enqueue_copy(self.queue,  self.sigs_cl, newbuf)
        event = self.kern_detect_peaks(self.queue,  self.global_size, self.local_size,
                                self.sigs_cl, self.ring_sum_cl, self.peak_mask_cl)
        event.wait()
        
        #~ if pos-(newbuf.shape[0]+2*self.n_span)<0:
            # the very first buffer is sacrified because of peak span
            #~ return None, None
        
        pyopencl.enqueue_copy(self.queue,  self.peak_mask, self.peak_mask_cl)
        time_ind_peaks,  = np.nonzero(self.peak_mask)
        
        if time_ind_peaks.size>0:
            time_ind_peaks += pos - self.chunksize - self.n_span
            chan_ind_peaks = None# not in this method
            return time_ind_peaks, chan_ind_peaks, None
        
        return None, None, None
    
    def initialize_stream(self):
        pass
    
    def change_params(self, cl_platform_index=None, cl_device_index=None, **kargs):
        BasePeakDetector.change_params(self,  **kargs)
        OpenCL_Helper.initialize_opencl(self, cl_platform_index=cl_platform_index, cl_device_index=cl_device_index)

        if self.chunksize > self.max_wg_size:
            n = int(np.ceil(self.chunksize / self.max_wg_size))
            self.global_size =  (self.max_wg_size * n, )
            self.local_size = (self.max_wg_size,)
        else:
            self.global_size = (self.chunksize, )
            self.local_size = (self.chunksize, )


        chunksize2=self.chunksize+2*self.n_span
        
        self.sum_rectified = np.zeros((self.chunksize), dtype=self.dtype)
        self.peak_mask = np.zeros((self.chunksize), dtype='uint8')
        ring_sum = np.zeros((chunksize2), dtype=self.dtype)
        
        #GPU buffers
        self.sigs_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=self.nb_channel*self.chunksize*self.dtype.itemsize)
        
        self.ring_sum_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=ring_sum)
        self.peak_mask_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.peak_mask)
        
        kernel = self.kernel%dict(chunksize=self.chunksize, nb_channel=self.nb_channel, n_span=self.n_span,
                    relative_threshold=self.relative_threshold, peak_sign={'+':1, '-':-1}[self.peak_sign])
        
        prg = pyopencl.Program(self.ctx, kernel)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
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

        if (pos>=chunksize){
            return;
        }
        
        
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


def get_mask_spatiotemporal_peaks(sigs, n_span, thresh, peak_sign, neighbours):
    sig_center = sigs[n_span:-n_span, :]

    if peak_sign == '+':
        mask_peaks = sig_center>thresh
        for chan in range(sigs.shape[1]):
            for neighbour in neighbours[chan, :]:
                if neighbour<0:
                    continue
                for i in range(n_span):
                    if chan != neighbour:
                        mask_peaks[:, chan] &= sig_center[:, chan] >= sig_center[:, neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan] > sigs[i:i+sig_center.shape[0], neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan]>=sigs[n_span+i+1:n_span+i+1+sig_center.shape[0], neighbour]
        
    elif peak_sign == '-':
        mask_peaks = sig_center<-thresh
        for chan in range(sigs.shape[1]):
            for neighbour in neighbours[chan, :]:
                if neighbour<0:
                    continue
                for i in range(n_span):
                    if chan != neighbour:
                        mask_peaks[:, chan] &= sig_center[:, chan] <= sig_center[:, neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan] < sigs[i:i+sig_center.shape[0], neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan]<=sigs[n_span+i+1:n_span+i+1+sig_center.shape[0], neighbour]
    
    return mask_peaks



class PeakDetectorGeometricalNumpy(BasePeakDetector):
    def process_buffer(self, sigs):
    
        if self.spatial_smooth_kernel is None:
            sigs = sigs
        else:
            sigs = np.dot(sigs, self.spatial_smooth_kernel)
        
        mask_peaks = self.get_mask_peaks_in_chunk(sigs)
        time_ind_peaks, chan_ind_peaks = np.nonzero(mask_peaks)
        time_ind_peaks += self.n_span
        peak_val_peaks = sigs[time_ind_peaks, chan_ind_peaks]
        
        return time_ind_peaks, chan_ind_peaks, peak_val_peaks
    
    def process_buffer_stream(self, pos, newbuf):
        self.fifo_sigs.new_chunk(newbuf, pos)
        sigs = self.fifo_sigs.get_data(pos-(newbuf.shape[0]+2*self.n_span), pos)
        
        time_ind_peaks, chan_ind_peaks, peak_val_peaks = self.process_buffer(sigs)

        if time_ind_peaks.size>0:
            time_ind_peaks += (pos - newbuf.shape[0] - 2 * self.n_span)
            return time_ind_peaks, chan_ind_peaks, peak_val_peaks

        return None, None, None
    
    def get_mask_peaks_in_chunk(self, fifo_residuals):
        
        # this is used by peeler engine geometry which handle externaly
        # a fifo residual
        mask_peaks = get_mask_spatiotemporal_peaks(fifo_residuals, self.n_span, self.relative_threshold, self.peak_sign, self.neighbours)
        return mask_peaks
    
    def change_params(self, adjacency_radius_um=200., **kargs):
        BasePeakDetector.change_params(self,  **kargs)
        
        self.adjacency_radius_um = adjacency_radius_um
        
        d = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
        neighbour_mask = d<=self.adjacency_radius_um
        nb_neighbour_per_channel = np.sum(neighbour_mask, axis=0)
        nb_max_neighbour = np.max(nb_neighbour_per_channel)
        
        self.nb_max_neighbour = nb_max_neighbour # include itself
        self.neighbours = np.zeros((self.nb_channel, nb_max_neighbour), dtype='int32')
        
        self.neighbours[:] = -1
        for c in range(self.nb_channel):
            neighb, = np.nonzero(neighbour_mask[c, :])
            self.neighbours[c, :neighb.size] = neighb
        
        self.fifo_sigs = FifoBuffer((self.chunksize+2*self.n_span, self.nb_channel), self.dtype)
    
    def initialize_stream(self):
        self.fifo_sigs = FifoBuffer((self.chunksize+2*self.n_span, self.nb_channel), self.dtype)
    

class PeakDetectorGeometricalNumba(PeakDetectorGeometricalNumpy):
    def get_mask_peaks_in_chunk(self, fifo_residuals):
        mask_peaks = numba_get_mask_spatiotemporal_peaks(fifo_residuals, self.n_span, self.relative_threshold, self.peak_sign, self.neighbours)
        return mask_peaks
    


class PeakDetectorGeometricalOpenCL(PeakDetectorGeometricalNumpy, OpenCL_Helper):
    def process_buffer(self, sigs):
        
        if sigs.shape[0] <self.chunksize:
            sigs2 = np.zeros((self.chunksize, self.nb_channel), dtype=self.dtype)
            sigs2[:sigs.shape[0], :] = sigs
            sigs = sigs2
    
        if self.spatial_smooth_kernel is None:
            sigs = sigs
        else:
            sigs = np.dot(sigs, self.spatial_smooth_kernel)
        
        mask_peaks = self.get_mask_peaks_in_chunk(sigs)
        time_ind_peaks, chan_ind_peaks = np.nonzero(mask_peaks)
        time_ind_peaks += self.n_span
        peak_val_peaks = sigs[time_ind_peaks, chan_ind_peaks]
        
        return time_ind_peaks, chan_ind_peaks, peak_val_peaks
        
        
    def process_buffer_stream(self, pos, newbuf):
        if newbuf.shape[0] <self.chunksize:
            newbuf2 = np.zeros((self.chunksize, self.nb_channel), dtype=self.dtype)
            newbuf2[-newbuf.shape[0]:, :] = newbuf
            newbuf = newbuf2

        self.fifo_sigs.new_chunk(newbuf, pos)
        sigs = self.fifo_sigs.get_data(pos-(newbuf.shape[0]+2*self.n_span), pos)
        
        time_ind_peaks, chan_ind_peaks, peak_val_peaks = self.process_buffer(sigs)
        
        if time_ind_peaks.size>0:
            time_ind_peaks += (pos - newbuf.shape[0] - 2 * self.n_span)
            
            return time_ind_peaks, chan_ind_peaks, peak_val_peaks

        return None, None, None
    
    def get_mask_peaks_in_chunk(self, fifo_residuals):
        pyopencl.enqueue_copy(self.queue,  self.fifo_sigs_cl, fifo_residuals)
        #~ print(self.chunksize, self.max_wg_size)
        event = self.kern_get_mask_spatiotemporal_peaks(self.queue,  self.global_size, self.local_size,
                                self.fifo_sigs_cl, self.neighbours_cl, self.mask_peaks_cl)
        event.wait()
        pyopencl.enqueue_copy(self.queue,  self.mask_peaks, self.mask_peaks_cl)
        
        return self.mask_peaks
    
    def initialize_stream(self):
        self._make_gpu_buffer()
        
    def _make_gpu_buffer(self):
        # TODO fifo size should be : chunksize+2*self.n_span
        self.fifo_size = self.chunksize + 2*self.n_span
        self.fifo_sigs = FifoBuffer((self.fifo_size, self.nb_channel), self.dtype)
        self.mask_peaks = np.zeros((self.chunksize, self.nb_channel), dtype='uint8')  # bool
        
        #GPU buffers
        self.fifo_sigs_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_sigs.buffer)
        self.neighbours_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.neighbours)
        self.mask_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.mask_peaks)        
        
    
    def change_params(self, cl_platform_index=None, cl_device_index=None, **kargs):
        PeakDetectorGeometricalNumpy.change_params(self,  **kargs)
        
        OpenCL_Helper.initialize_opencl(self, cl_platform_index=cl_platform_index, cl_device_index=cl_device_index)
        #~ print(self.ctx)
        #~ print(self.chunksize)
        
        if self.chunksize > self.max_wg_size:
            n = int(np.ceil(self.chunksize / self.max_wg_size))
            self.global_size =  (self.max_wg_size * n, )
            self.local_size = (self.max_wg_size,)
        else:
            self.global_size = (self.chunksize, )
            self.local_size = (self.chunksize, )
        
        #~ print('self.global_size', self.global_size, 'self.chunksize', self.chunksize)
        
        
        self._make_gpu_buffer()
        
        # TODO fifo size should be : chunksize+2*self.n_span
        self.fifo_sigs = FifoBuffer((self.chunksize + 2*self.n_span, self.nb_channel), self.dtype)
        self.mask_peaks = np.zeros((self.chunksize, self.nb_channel), dtype='uint8')  # bool
        

        #GPU buffers
        self.fifo_sigs_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.fifo_sigs.buffer)
        self.neighbours_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.neighbours)
        self.mask_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=self.mask_peaks)
        
        kernel_ = opencl_kernel_geometrical_part1 + opencl_kernel_geometrical_part2
        
        kernel = kernel_ % dict(fifo_size=self.fifo_size, nb_channel=self.nb_channel, n_span=self.n_span,
                    relative_threshold=self.relative_threshold, peak_sign={'+':1, '-':-1}[self.peak_sign], nb_neighbour=self.nb_max_neighbour)
        
        prg = pyopencl.Program(self.ctx, kernel)

        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.kern_get_mask_spatiotemporal_peaks = getattr(self.opencl_prg, 'get_mask_spatiotemporal_peaks')

    
opencl_kernel_geometrical_part1 =  """
#define fifo_size %(fifo_size)d
#define n_span %(n_span)d
#define nb_channel %(nb_channel)d
#define relative_threshold %(relative_threshold)f
#define peak_sign %(peak_sign)d
#define nb_neighbour %(nb_neighbour)d
"""


opencl_kernel_geometrical_part2 =  """
__kernel void get_mask_spatiotemporal_peaks(__global  float *sigs,
                                            __global  int *neighbours,
                                            __global  uchar *mask_peaks){

    int pos = get_global_id(0);
    
    if (pos>=(fifo_size - (2 * n_span))){
        return;
    }
    

    float v;
    uchar peak;
    int chan_neigh;

    
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
        
        mask_peaks[pos * nb_channel + chan] = peak;
        
    }
}
"""


# TODO atomic_inc for spatiotemporal_opencl
# TODO rename engine propagate to GUI/examples/online

def get_peak_detector_class(method, engine):
    if engine == 'numba':
        assert HAVE_NUMBA, 'You must install numba'
    if engine == 'opencl':
        assert HAVE_PYOPENCL, 'You must install opencl'
    
    
    
    if method == 'global' and engine =='numba':
        print('WARNING : no peak detector global + numba use numpy instead')
        engine ='numpy'
    
    class_ = peakdetector_classes.get((method, engine))
    
    return class_
    


peakdetector_classes = { 
    ('global', 'numpy') : PeakDetectorGlobalNumpy, 
    ('global', 'opencl') : PeakDetectorGlobalOpenCL,
    ('geometrical', 'numpy') : PeakDetectorGeometricalNumpy,
    ('geometrical', 'numba') : PeakDetectorGeometricalNumba,
    ('geometrical', 'opencl'): PeakDetectorGeometricalOpenCL,
}


