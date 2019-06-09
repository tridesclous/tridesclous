import scipy.signal
import numpy as np

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags


#~ from pyacq.dsp.overlapfiltfilt import SosFiltfilt_Scipy
from .tools import FifoBuffer, median_mad


def offline_signal_preprocessor(sigs, sample_rate, common_ref_removal=True,
        highpass_freq=300., lowpass_freq=None, output_dtype='float32', normalize=True, **unused):
    #cast
    sigs = sigs.astype(output_dtype)
    
    #filter
    if highpass_freq is not None:
        b, a = scipy.signal.iirfilter(5, highpass_freq/sample_rate*2, analog=False,
                                        btype = 'highpass', ftype = 'butter', output = 'ba')
        filtered_sigs = scipy.signal.filtfilt(b, a, sigs, axis=0)
    else:
        filtered_sigs = sigs.copy()
    
    if lowpass_freq is not None:
        b, a = scipy.signal.iirfilter(5, lowpass_freq/sample_rate*2, analog=False,
                                        btype = 'lowpass', ftype = 'butter', output = 'ba')
        filtered_sigs = scipy.signal.filtfilt(b, a, filtered_sigs, axis=0)
        

    # common reference removal
    if common_ref_removal:
        filtered_sigs = filtered_sigs - np.median(filtered_sigs, axis=1)[:, None]
    
    # normalize
    if normalize:
        #~ med = np.median(filtered_sigs, axis=0)
        #~ mad = np.median(np.abs(filtered_sigs-med),axis=0)*1.4826
        med, mad = median_mad(filtered_sigs, axis=0)
        
        normed_sigs = (filtered_sigs - med)/mad
    else:
        normed_sigs = filtered_sigs
    
    return normed_sigs.astype(output_dtype)


def estimate_medians_mads_after_preprocesing(sigs, sample_rate, **params):
    params2 = dict(params)
    params2['normalize'] = False
    
    filtered_sigs = offline_signal_preprocessor(sigs, sample_rate, **params2)
    med, mad = median_mad(filtered_sigs, axis=0)
    return med, mad
    
    
    


class SignalPreprocessor_base:
    def __init__(self,sample_rate, nb_channel, chunksize, input_dtype):
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.input_dtype = input_dtype

    
    def change_params(self, common_ref_removal=True,
                                            highpass_freq=300.,
                                            lowpass_freq=None,
                                            smooth_size=0,
                                            output_dtype='float32', 
                                            normalize=True,
                                            lostfront_chunksize = None,
                                            signals_medians=None, signals_mads=None):
                
        self.signals_medians = signals_medians
        self.signals_mads = signals_mads
        
        self.common_ref_removal = common_ref_removal
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.smooth_size = int(smooth_size)
        self.output_dtype = np.dtype(output_dtype)
        self.normalize = normalize
        self.lostfront_chunksize = lostfront_chunksize
        
        # set default lostfront_chunksize if none is provided
        if self.lostfront_chunksize is None or self.lostfront_chunksize==0 or self.lostfront_chunksize==-1:
            assert self.highpass_freq is not None, 'lostfront_chunksize=None needs a highpass_freq'
            self.lostfront_chunksize = int(self.sample_rate/self.highpass_freq*3)
            #~ print('self.lostfront_chunksize', self.lostfront_chunksize)
        
        self.backward_chunksize = self.chunksize + self.lostfront_chunksize
        #~ print('self.lostfront_chunksize', self.lostfront_chunksize)
        #~ print('self.backward_chunksize', self.backward_chunksize)
        #~ assert self.backward_chunksize>self.chunksize
        
        self.coefficients = np.zeros((0, 6))
        
        nyquist = self.sample_rate/2.
        
        if self.highpass_freq is not None:
            if self.highpass_freq>0 and self.highpass_freq<nyquist:
                coeff_hp = scipy.signal.iirfilter(5, highpass_freq/self.sample_rate*2, analog=False,
                                        btype = 'highpass', ftype = 'butter', output = 'sos')
                self.coefficients = np.concatenate((self.coefficients, coeff_hp))
        
        if self.lowpass_freq is not None:
            if self.lowpass_freq>0 and self.lowpass_freq<nyquist:
            #~ if self.lowpass_freq>(self.sample_rate/2.):
                #~ self.lowpass_freq=(self.sample_rate/2.01)
                coeff_lp = scipy.signal.iirfilter(5, lowpass_freq/self.sample_rate*2, analog=False,
                                        btype = 'lowpass', ftype = 'butter', output = 'sos')
                self.coefficients = np.concatenate((self.coefficients, coeff_lp))
        
        if self.smooth_size>0:
            b0 = (1./3)**.5
            b1 = (1-b0)
            b2 = 0.
            coeff_smooth = np.array([[b0, b1, b2, 1,0,0]], dtype=self.output_dtype)
            coeff_smooth = np.tile(coeff_smooth, (self.smooth_size, 1))
            self.coefficients = np.concatenate((self.coefficients, coeff_smooth))
        
        
        
        
        if self.coefficients.shape[0]==0:
            #this is the null filter
            self.coefficients = np.array([[1, 0, 0, 1,0,0]], dtype=self.output_dtype)
        
        #~ self.filtfilt_engine = SosFiltfilt_Scipy(self.coefficients, self.nb_channel, output_dtype, self.chunksize, lostfront_chunksize)
        self.nb_section =self. coefficients.shape[0]
        self.forward_buffer = FifoBuffer((self.backward_chunksize, self.nb_channel), self.output_dtype)
        self.zi = np.zeros((self.nb_section, 2, self.nb_channel), dtype= self.output_dtype)
        
        #~ print('self.normalize', self.normalize)
        if self.normalize:
            assert self.signals_medians is not None
            assert self.signals_mads is not None
            


class SignalPreprocessor_Numpy(SignalPreprocessor_base):
    """
    This apply chunk by chunk on a multi signal:
       * baseline removal
       * hight pass filtfilt
       * normalize (optional)
    
    """
        
    def process_data(self, pos, data):
        
        

        #TODO this cause problem for peakdetector_opencl
        # because pos is not multiple  chunksize

        #~ data = data.astype(self.output_dtype)
        #~ pos2, data2 = self.filtfilt_engine.compute_one_chunk(pos, data)
        #~ if pos2 is None:
            #~ return None, None
        
        
        #Online filtfilt
        chunk = data.astype(self.output_dtype)
        forward_chunk_filtered, self.zi = scipy.signal.sosfilt(self.coefficients, chunk, zi=self.zi, axis=0)
        forward_chunk_filtered = forward_chunk_filtered.astype(self.output_dtype)
        
        self.forward_buffer.new_chunk(forward_chunk_filtered, index=pos)
        
        #OLD implementation
        #~ start = pos-self.backward_chunksize
        #~ if start<-self.lostfront_chunksize:
            #~ return None, None
        #~ if start>0:
            #~ backward_chunk = self.forward_buffer.get_data(start,pos)
        #~ else:
            #~ backward_chunk = self.forward_buffer.get_data(0,pos)
        #~ backward_filtered = scipy.signal.sosfilt(self.coefficients, backward_chunk[::-1, :], zi=None, axis=0)
        #~ backward_filtered = backward_filtered[::-1, :]
        #~ backward_filtered = backward_filtered.astype(self.output_dtype)
            
        #~ if start>0:
            #~ backward_filtered = backward_filtered[:self.chunksize]
            #~ assert data.shape[0] == self.chunksize
        #~ else:
            #~ backward_filtered = backward_filtered[:-self.lostfront_chunksize]
        #~ data2 = backward_filtered
        #~ pos2 = pos-self.lostfront_chunksize
            
        # NEW IMPLENTATION
        backward_chunk = self.forward_buffer.buffer
        backward_filtered = scipy.signal.sosfilt(self.coefficients, backward_chunk[::-1, :], zi=None, axis=0)
        backward_filtered = backward_filtered[::-1, :]
        backward_filtered = backward_filtered.astype(self.output_dtype)
        
        pos2 = pos-self.lostfront_chunksize
        if pos2<0:
            return None, None
        
        i1 = self.backward_chunksize-self.lostfront_chunksize-chunk.shape[0]
        i2 = self.chunksize

        assert i1<i2
        data2 = backward_filtered[i1:i2]
        if (pos2-data2.shape[0])<0:
            data2 = data2[data2.shape[0]-pos2:]
        
        #~ print('pos', pos, 'pos2', pos2, data2.shape)
        
        # removal ref
        if self.common_ref_removal:
            data2 -= np.median(data2, axis=1)[:, None]
        
        #normalize
        if self.normalize:
            data2 -= self.signals_medians
            data2 /= self.signals_mads
        
        return pos2, data2
    

        
        



class SignalPreprocessor_OpenCL(SignalPreprocessor_base, OpenCL_Helper):
    """
    Implementation in OpenCL depending on material and nb_channel
    this can lead to a smal speed improvement...
    
    """
    def __init__(self,sample_rate, nb_channel, chunksize, input_dtype):
        SignalPreprocessor_base.__init__(self,sample_rate, nb_channel, chunksize, input_dtype)
    
    def process_data(self, pos, data):
        
        assert data.shape[0]==self.chunksize
                
        if not data.flags['C_CONTIGUOUS'] or data.dtype!=self.output_dtype:
            chunk = np.ascontiguousarray(data, dtype=self.output_dtype)
        
        
        #Online filtfilt
        
        
        #forward filter
        event = pyopencl.enqueue_copy(self.queue,  self.input_cl, chunk)
        #~ event.wait()
        
        event = self.kern_forward_filter(self.queue,  (self.nb_channel,), (self.nb_channel,),
                                self.input_cl, self.output_forward_cl, self.coefficients_cl, self.zi1_cl)
        #~ event.wait()
        
        #roll and add to bacward fifo
        event = self.kern_roll_fifo(self.queue,  (self.nb_channel, self.lostfront_chunksize), (self.nb_channel, 1),
                                self.fifo_input_backward_cl)
        #~ event.wait()
        
        event = self.kern_new_chunk_fifo(self.queue,  (self.nb_channel, self.chunksize), (self.nb_channel, 1),
                                self.fifo_input_backward_cl,  self.output_forward_cl)
        #~ event.wait()
        
        # backwward
        self.zi2[:]=0
        pyopencl.enqueue_copy(self.queue,  self.zi2_cl, self.zi2)

        event = self.kern_backward_filter(self.queue,  (self.nb_channel,), (self.nb_channel,),
                                self.fifo_input_backward_cl, self.output_backward_cl, self.coefficients_cl, self.zi2_cl)
        event.wait()
        
        
        #~ event.wait()
        
        start = pos-self.backward_chunksize
        if start<-self.lostfront_chunksize:
            return None, None
        
        pos2 = pos-self.lostfront_chunksize
        
        
        event = pyopencl.enqueue_copy(self.queue,  self.output_backward, self.output_backward_cl)
        
        if start>0:
            data2 = self.output_backward[:self.chunksize, :]
        else:
            data2 = self.output_backward[self.lostfront_chunksize:self.chunksize, :]
        
        data2 = data2.copy()
        

        #~ print('pos', pos, 'start', start, 'pos2', pos2, data2.shape)
        
        #TODO make OpenCL for this
        # removal ref
        if self.common_ref_removal:
            data2 -= np.median(data2, axis=1)[:, None]
        
        #TODO make OpenCL for this
        #normalize
        if self.normalize:
            data2 -= self.signals_medians
            data2 /= self.signals_mads
        
        return pos2, data2        
        
        
    def change_params(self, **kargs):

        cl_platform_index=kargs.pop('cl_platform_index', None)
        cl_device_index=kargs.pop('cl_device_index', None)
        OpenCL_Helper.initialize_opencl(self,cl_platform_index=cl_platform_index, cl_device_index=cl_device_index)
        
        SignalPreprocessor_base.change_params(self, **kargs)
        assert self.output_dtype=='float32', 'SignalPreprocessor_OpenCL support only float32 at the moment'
        assert self.lostfront_chunksize<self.chunksize, 'OpenCL fifo work only for self.lostfront_chunksize<self.chunksize'
        
        
        
        self.coefficients = np.ascontiguousarray(self.coefficients, dtype=self.output_dtype)
        #~ print(self.coefficients.shape)
        
        
        self.zi1 = np.zeros((self.nb_channel, self.nb_section, 2), dtype= self.output_dtype)
        self.zi2 = np.zeros((self.nb_channel, self.nb_section, 2), dtype= self.output_dtype)
        self.output_forward = np.zeros((self.chunksize, self.nb_channel), dtype= self.output_dtype)
        self.fifo_input_backward = np.zeros((self.backward_chunksize, self.nb_channel), dtype= self.output_dtype)
        self.output_backward = np.zeros((self.backward_chunksize, self.nb_channel), dtype= self.output_dtype)
        
        #GPU buffers
        self.coefficients_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.coefficients)
        self.zi1_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.zi1)
        self.zi2_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.zi2)
        self.input_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=self.output_forward.nbytes)
        self.output_forward_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=self.output_forward.nbytes)
        self.fifo_input_backward_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fifo_input_backward)
        self.output_backward_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=self.output_backward.nbytes)

        #CL prog
        kernel = self.kernel%dict(forward_chunksize=self.chunksize, backward_chunksize=self.backward_chunksize,
                        lostfront_chunksize=self.lostfront_chunksize, nb_section=self.nb_section, nb_channel=self.nb_channel)
        #~ print(kernel)
        #~ exit()
        prg = pyopencl.Program(self.ctx, kernel)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)


        self.kern_roll_fifo = getattr(self.opencl_prg, 'roll_fifo')
        self.kern_new_chunk_fifo = getattr(self.opencl_prg, 'new_chunk_fifo')
        self.kern_forward_filter = getattr(self.opencl_prg, 'forward_filter')
        self.kern_backward_filter = getattr(self.opencl_prg, 'backward_filter')

    kernel = """
    #define forward_chunksize %(forward_chunksize)d
    #define backward_chunksize %(backward_chunksize)d
    #define lostfront_chunksize %(lostfront_chunksize)d
    #define nb_section %(nb_section)d
    #define nb_channel %(nb_channel)d


    __kernel void roll_fifo(__global  float *fifo){
    
        int chan = get_global_id(0);
        int pos = get_global_id(1);
        
        fifo[(pos)*nb_channel+chan] = fifo[(pos+forward_chunksize)*nb_channel+chan];
    }
    
    __kernel void new_chunk_fifo(__global  float *fifo, __global  float *input){

        int chan = get_global_id(0);
        int pos = get_global_id(1);
        
        fifo[(pos+lostfront_chunksize)*nb_channel+chan] = input[(pos)*nb_channel+chan];
        
    }
    


    __kernel void sos_filter(__global  float *input, __global  float *output, __constant  float *coefficients, 
                                                                            __global float *zi, int chunksize, int direction) {

        int chan = get_global_id(0); //channel indice
        
        int offset_filt2;  //offset channel within section
        int offset_zi = chan*nb_section*2;
        
        int idx;

        float w0, w1,w2;
        float res;
        
        for (int section=0; section<nb_section; section++){
        
            //offset_filt2 = chan*nb_section*6+section*6;
            offset_filt2 = section*6;
            
            w1 = zi[offset_zi+section*2+0];
            w2 = zi[offset_zi+section*2+1];
            
            for (int s=0; s<chunksize;s++){
                
                if (direction==1) {idx = s*nb_channel+chan;}
                else if (direction==-1) {idx = (chunksize-s-1)*nb_channel+chan;}
                
                if (section==0)  {w0 = input[idx];}
                else {w0 = output[idx];}
                
                w0 -= coefficients[offset_filt2+4] * w1;
                w0 -= coefficients[offset_filt2+5] * w2;
                res = coefficients[offset_filt2+0] * w0 + coefficients[offset_filt2+1] * w1 +  coefficients[offset_filt2+2] * w2;
                w2 = w1; w1 =w0;
                
                output[idx] = res;
            }
            
            zi[offset_zi+section*2+0] = w1;
            zi[offset_zi+section*2+1] = w2;

        }
       
    }
    
    __kernel void forward_filter(__global  float *input, __global  float *output, __constant  float *coefficients, __global float *zi){
        sos_filter(input, output, coefficients, zi, forward_chunksize, 1);
    }

    __kernel void backward_filter(__global  float *input, __global  float *output, __constant  float *coefficients, __global float *zi) {
        sos_filter(input, output, coefficients, zi, backward_chunksize, -1);
    }
    
    """







signalpreprocessor_engines = { 'numpy' : SignalPreprocessor_Numpy,
                                                'opencl' : SignalPreprocessor_OpenCL}
