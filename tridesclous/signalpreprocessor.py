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
        else:
            chunk = data
        
        
        #Online filtfilt
        
        event = pyopencl.enqueue_copy(self.queue,  self.input_cl, chunk)
        event = self.kern_forward_backward_filter(self.queue,  (self.nb_channel,), (self.nb_channel,),
                            self.input_cl, self.coefficients_cl, self.zi1_cl, self.zi2_cl,
                            self.fifo_input_backward_cl, self.signals_medians_cl, self.signals_mads_cl,  self.output_backward_cl)
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
        
        if self.common_ref_removal:
            #TODO make OpenCL for this
            # removal ref
            if self.common_ref_removal:
                data2 -= np.median(data2, axis=1)[:, None]
            
            
            #normalize
            if self.normalize:
                # OpenCL for this when no common_ref_removal
                data2 -= self.signals_medians
                data2 /= self.signals_mads
            
        return pos2, data2        
        
        
    def change_params(self, **kargs):

        cl_platform_index=kargs.pop('cl_platform_index', None)
        cl_device_index=kargs.pop('cl_device_index', None)
        ctx=kargs.pop('ctx', None)
        queue=kargs.pop('queue', None)
        OpenCL_Helper.initialize_opencl(self,cl_platform_index=cl_platform_index, cl_device_index=cl_device_index, ctx=ctx, queue=queue)
        
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
        if self.signals_medians is not None:
            self.signals_medians_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.signals_medians)
            self.signals_mads_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.signals_mads)
        else:
            self.signals_medians_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(self.nb_channel, dtype= self.output_dtype))
            self.signals_mads_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(self.nb_channel, dtype= self.output_dtype))
        
        #CL prog
        if not self.common_ref_removal and  self.normalize:
            extra_code_nomalize = _extra_code_nomalize
        else:
            extra_code_nomalize = ''
        
        kernel_formated = processor_kernel%dict(forward_chunksize=self.chunksize, backward_chunksize=self.backward_chunksize,
                        lostfront_chunksize=self.lostfront_chunksize, nb_section=self.nb_section, nb_channel=self.nb_channel, 
                        extra_code_nomalize=extra_code_nomalize)
        #~ print(kernel_formated)
        #~ exit()
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)


        self.kern_forward_backward_filter = getattr(self.opencl_prg, 'forward_backward_filter')
        
        

processor_kernel = """
#define forward_chunksize %(forward_chunksize)d
#define backward_chunksize %(backward_chunksize)d
#define lostfront_chunksize %(lostfront_chunksize)d
#define nb_section %(nb_section)d
#define nb_channel %(nb_channel)d


__kernel void sos_filter(__global  float *input, __global  float *output, __constant  float *coefficients, 
                                                                        __global float *zi, int chunksize, int direction, int out_offset_index) {

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
            else {w0 = output[idx+out_offset_index];}
            
            w0 -= coefficients[offset_filt2+4] * w1;
            w0 -= coefficients[offset_filt2+5] * w2;
            res = coefficients[offset_filt2+0] * w0 + coefficients[offset_filt2+1] * w1 +  coefficients[offset_filt2+2] * w2;
            w2 = w1; w1 =w0;
            
            output[idx+out_offset_index] = res;
        }
        
        zi[offset_zi+section*2+0] = w1;
        zi[offset_zi+section*2+1] = w2;

    }
   
}


__kernel void forward_backward_filter(__global  float *input,
                                                            __constant  float * coefficients,
                                                            __global float * zi1,
                                                            __global float * zi2,
                                                            __global  float *fifo_input_backward,
                                                            __global  float *signals_medians,
                                                            __global  float *signals_mads,
                                                            __global  float *output_backward){

    
    int chan = get_global_id(0); //channel indice


    //roll
    for (int s=0; s<lostfront_chunksize;s++){
        fifo_input_backward[(s)*nb_channel+chan] = fifo_input_backward[(s+forward_chunksize)*nb_channel+chan];
    }

    int out_offset_index = lostfront_chunksize*nb_channel;
    sos_filter(input, fifo_input_backward, coefficients, zi1, forward_chunksize, 1, out_offset_index);
    
    //set zi2 to zeros
    for (int s=0; s<nb_section;s++){
        zi2[chan*nb_section*2+s] = 0;
        zi2[chan*nb_section*2+s+1] = 0;
    }
    
    //filter backward
    sos_filter(fifo_input_backward, output_backward, coefficients, zi2, backward_chunksize, -1, 0);
    
    // nomalize optional
    %(extra_code_nomalize)s

}

"""

_extra_code_nomalize = """
    float v;
    for (int s=0; s<forward_chunksize;s++){
        v = output_backward[(s)*nb_channel+chan];
        output_backward[(s)*nb_channel+chan] = (v - signals_medians[chan]) / signals_mads[chan];
    }

"""





signalpreprocessor_engines = { 'numpy' : SignalPreprocessor_Numpy,
                                                'opencl' : SignalPreprocessor_OpenCL}
