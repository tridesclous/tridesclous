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
                                            pad_width = None,
                                            signals_medians=None, signals_mads=None):
                
        self.signals_medians = signals_medians
        self.signals_mads = signals_mads
        
        self.common_ref_removal = common_ref_removal
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.smooth_size = int(smooth_size)
        self.output_dtype = np.dtype(output_dtype)
        self.normalize = normalize
        self.pad_width = pad_width
        
        # set default pad_width if none is provided
        if self.pad_width is None or self.pad_width<=0:
            assert self.highpass_freq is not None, 'pad_width=None needs a highpass_freq'
            self.pad_width = int(self.sample_rate/self.highpass_freq*3)
            #~ print('self.pad_width', self.pad_width)
        
        self.chunksize_1pad = self.chunksize + self.pad_width
        self.chunksize_2pad = self.chunksize + 2 * self.pad_width
        #~ print('self.pad_width', self.pad_width)
        #~ print('self.chunksize_1pad', self.chunksize_1pad)
        #~ assert self.chunksize_1pad>self.chunksize
        
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
        self.forward_buffer = FifoBuffer((self.chunksize_1pad, self.nb_channel), self.output_dtype)
        self.zi = np.zeros((self.nb_section, 2, self.nb_channel), dtype= self.output_dtype)
        
        #~ print('self.normalize', self.normalize)
        if self.normalize:
            assert self.signals_medians is not None
            assert self.signals_mads is not None

    def process_buffer(self, data):
        # used for offline processing when parralisation  is possible
        raise(NotImplmentedError)

    def initialize_stream(self):
        # must be for each new segment when index 
        # start back
        raise(NotImplmentedError)
    
    def process_buffer_stream(self, pos, data):
        # used in real time mode when chunk are given one after another
        raise(NotImplmentedError)
    


class SignalPreprocessor_Numpy(SignalPreprocessor_base):
    """
    This apply chunk by chunk on a multi signal:
       * baseline removal
       * hight pass filtfilt
       * normalize (optional)
    
    """
    def process_buffer(self, data):
        data = data.astype(self.output_dtype)
        processed_data = scipy.signal.sosfiltfilt(self.coefficients, data, axis=0)
        # TODO find why sosfiltfilt reverse strides!!!
        processed_data = np.ascontiguousarray(processed_data, dtype=self.output_dtype)
        # removal ref
        if self.common_ref_removal:
            processed_data -= np.median(processed_data, axis=1)[:, None]

        #normalize
        if self.normalize:
            processed_data -= self.signals_medians
            processed_data /= self.signals_mads

        return processed_data
    
    def process_buffer_stream(self, pos, data):
        # TODO rewrite this with self.process_buffer()
        
        #Online filtfilt
        chunk = data.astype(self.output_dtype)
        
        forward_chunk_filtered, self.zi = scipy.signal.sosfilt(self.coefficients, chunk, zi=self.zi, axis=0)
        forward_chunk_filtered = forward_chunk_filtered.astype(self.output_dtype)
        
        self.forward_buffer.new_chunk(forward_chunk_filtered, index=pos)
        
        backward_chunk = self.forward_buffer.buffer
        backward_filtered = scipy.signal.sosfilt(self.coefficients, backward_chunk[::-1, :], zi=None, axis=0)
        backward_filtered = backward_filtered[::-1, :]
        backward_filtered = backward_filtered.astype(self.output_dtype)
        
        pos2 = pos-self.pad_width
        if pos2<0:
            return None, None
        
        i1 = self.chunksize_1pad-self.pad_width-chunk.shape[0]
        i2 = self.chunksize

        assert i1<i2
        data2 = backward_filtered[i1:i2]
        if (pos2-data2.shape[0])<0:
            data2 = data2[data2.shape[0]-pos2:]
        
        # removal ref
        if self.common_ref_removal:
            data2 -= np.median(data2, axis=1)[:, None]
        
        #normalize
        if self.normalize:
            data2 -= self.signals_medians
            data2 /= self.signals_mads
        
        return pos2, data2
    
    def initialize_stream(self):
        self.forward_buffer.reset()
        self.zi[:] = 0
    
        
        



class SignalPreprocessor_OpenCL(SignalPreprocessor_base, OpenCL_Helper):
    """
    Implementation in OpenCL depending on material and nb_channel
    this can lead to a smal speed improvement...
    
    """
    def __init__(self,sample_rate, nb_channel, chunksize, input_dtype):
        SignalPreprocessor_base.__init__(self,sample_rate, nb_channel, chunksize, input_dtype)
    
    def _check_data(self, data):
        if not data.flags['C_CONTIGUOUS'] or data.dtype!=self.output_dtype:
            data = np.ascontiguousarray(data, dtype=self.output_dtype)

        return data
        
    
    def process_buffer(self, data):
        data = self._check_data(data)
        #~ print(data.shape, self.chunksize, self.chunksize_2pad, self.pad_width)
        #~ assert data.shape[0] == self.chunksize_2pad
        if data.shape[0] == self.chunksize_2pad:
            # OK
            unpad = 0
        elif data.shape[0] < self.chunksize_2pad:
            # put some zero
            unpad = self.chunksize_2pad - data.shape[0]
            data_pad = np.zeros((self.chunksize_2pad, data.shape[1]), dtype=data.dtype)
            #~ print('Apply a data pad')
            data = data_pad
        else:
            raise ValueError(f'data have wring shape{data.shape[0]} { self.chunksize_2pad}')
        
        event = pyopencl.enqueue_copy(self.queue,  self.input_2pad_cl, data)

        event = self.kern_forward_backward_filter(self.queue,  (self.nb_channel,), (self.nb_channel,),
                            self.input_2pad_cl, self.coefficients_cl, self.zi1_cl, self.zi2_cl,
                            self.signals_medians_cl, self.signals_mads_cl,  self.output_2pad_cl)
        #~ event.wait()
        
        event = pyopencl.enqueue_copy(self.queue,  self.output_2pad, self.output_2pad_cl)
        event.wait()
        
        data2 = self.output_2pad.copy()
        
        if self.common_ref_removal:
            # at the moment common_ref_removal is done on CPU
            # and so to avoid transfer normalize is also done on CPU
            #TODO implement OpenCL for removal ref
            if self.common_ref_removal:
                data2 -= np.median(data2, axis=1)[:, None]
            
            #normalize
            if self.normalize:
                # OpenCL for this when no common_ref_removal
                data2 -= self.signals_medians
                data2 /= self.signals_mads
        
        if unpad > 0:
            data2 = data2[:-unpad, :]
        
        return data2
    
    def process_buffer_stream(self, pos, data):
        
        assert data.shape[0]==self.chunksize
        
        data = self._check_data(data)
        
        #Online filtfilt
        event = pyopencl.enqueue_copy(self.queue,  self.input_cl, data)
        event = self.kern_stream_forward_backward_filter(self.queue,  (self.nb_channel,), (self.nb_channel,),
                            self.input_cl, self.coefficients_cl, self.zi1_cl, self.zi2_cl,
                            self.fifo_input_backward_cl, self.signals_medians_cl, self.signals_mads_cl,  self.output_backward_cl)
        event.wait()
        
         
        
        
        #~ event.wait()
        
        start = pos-self.chunksize_1pad
        if start<-self.pad_width:
            return None, None
        
        pos2 = pos-self.pad_width
        

        event = pyopencl.enqueue_copy(self.queue,  self.output_backward, self.output_backward_cl)
        if start>0:
            data2 = self.output_backward[:self.chunksize, :]
        else:
            data2 = self.output_backward[self.pad_width:self.chunksize, :]
        data2 = data2.copy()
        
        if self.common_ref_removal:
            # at the moment common_ref_removal is done on CPU
            # and so to avoid transfer normalize is also done on CPU
            #TODO implement OpenCL for removal ref
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
        assert self.pad_width<self.chunksize, 'OpenCL fifo work only for self.pad_width<self.chunksize'
        
        
        
        self.coefficients = np.ascontiguousarray(self.coefficients, dtype=self.output_dtype)
        #~ print(self.coefficients.shape)
        
        # this is for stream processing
        self.zi1 = np.zeros((self.nb_channel, self.nb_section, 2), dtype= self.output_dtype)
        self.zi2 = np.zeros((self.nb_channel, self.nb_section, 2), dtype= self.output_dtype)
        self.output_forward = np.zeros((self.chunksize, self.nb_channel), dtype= self.output_dtype)
        self.fifo_input_backward = np.zeros((self.chunksize_1pad, self.nb_channel), dtype= self.output_dtype)
        self.output_backward = np.zeros((self.chunksize_1pad, self.nb_channel), dtype= self.output_dtype)
        
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
        
        # this is for offline processing
        self.input_2pad = np.zeros((self.chunksize_2pad, self.nb_channel), dtype= self.output_dtype)
        self.output_2pad = np.zeros((self.chunksize_2pad, self.nb_channel), dtype= self.output_dtype)
        self.input_2pad_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.input_2pad)
        self.output_2pad_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.output_2pad)


        #CL prog
        if not self.common_ref_removal and  self.normalize:
            extra_code_nomalize = _extra_code_nomalize
            extra_code_nomalize2 = _extra_code_nomalize2
        else:
            extra_code_nomalize = ''
            extra_code_nomalize2 = ''
        
        kernel_formated = processor_kernel%dict(chunksize=self.chunksize, chunksize_1pad=self.chunksize_1pad,
                        chunksize_2pad=self.chunksize_2pad,
                        pad_width=self.pad_width, nb_section=self.nb_section, nb_channel=self.nb_channel, 
                        extra_code_nomalize=extra_code_nomalize, extra_code_nomalize2=extra_code_nomalize2)
        #~ print(kernel_formated)
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build(options='-cl-mad-enable')
        
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)


        self.kern_stream_forward_backward_filter = getattr(self.opencl_prg, 'stream_forward_backward_filter')
        self.kern_forward_backward_filter = getattr(self.opencl_prg, 'forward_backward_filter')

    def initialize_stream(self):
        self.output_forward[:] = 0
        event = pyopencl.enqueue_copy(self.queue,  self.output_backward_cl, self.output_backward)
        event.wait()
        
        self.zi1[:] = 0
        event = pyopencl.enqueue_copy(self.queue,  self.zi1_cl, self.zi1)
        event.wait()

        self.zi2[:] = 0
        event = pyopencl.enqueue_copy(self.queue,  self.zi2_cl, self.zi2)
        event.wait()


processor_kernel = """
#define chunksize %(chunksize)d
#define chunksize_1pad %(chunksize_1pad)d
#define chunksize_2pad %(chunksize_2pad)d
#define pad_width %(pad_width)d
#define nb_section %(nb_section)d
#define nb_channel %(nb_channel)d


__kernel void sos_filter(__global  float *input, __global  float *output, __constant  float *coefficients, 
                                                                        __global float *zi, int local_chunksize, int direction, int out_offset_index) {

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
        
        for (int s=0; s<local_chunksize;s++){
            
            if (direction==1) {idx = s*nb_channel+chan;}
            else if (direction==-1) {idx = (local_chunksize-s-1)*nb_channel+chan;}
            
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


__kernel void stream_forward_backward_filter(__global  float *input,
                                                            __constant  float * coefficients,
                                                            __global float * zi1,
                                                            __global float * zi2,
                                                            __global  float *fifo_input_backward,
                                                            __global  float *signals_medians,
                                                            __global  float *signals_mads,
                                                            __global  float *output_backward){

    
    int chan = get_global_id(0); //channel indice


    //roll
    for (int s=0; s<pad_width;s++){
        fifo_input_backward[(s)*nb_channel+chan] = fifo_input_backward[(s+chunksize)*nb_channel+chan];
    }

    int out_offset_index = pad_width*nb_channel;
    sos_filter(input, fifo_input_backward, coefficients, zi1, chunksize, 1, out_offset_index);
    
    //set zi2 to zeros
    for (int s=0; s<nb_section;s++){
        zi2[chan*nb_section*2+s] = 0;
        zi2[chan*nb_section*2+s+1] = 0;
    }
    
    //filter backward
    sos_filter(fifo_input_backward, output_backward, coefficients, zi2, chunksize_1pad, -1, 0);
    
    // nomalize optional
    %(extra_code_nomalize)s

}

__kernel void forward_backward_filter(__global  float *input,
                                                            __constant  float * coefficients,
                                                            __global float * zi1,
                                                            __global float * zi2,
                                                            __global  float *signals_medians,
                                                            __global  float *signals_mads,
                                                            __global  float *output){

    
    int chan = get_global_id(0); //channel indice


    sos_filter(input, input, coefficients, zi1, chunksize_2pad, 1, 0);
    
    //filter backward
    sos_filter(input, output, coefficients, zi2, chunksize_2pad, -1, 0);
    
    // nomalize optional
    %(extra_code_nomalize2)s

}



"""

_extra_code_nomalize = """
    float v;
    for (int s=0; s<chunksize;s++){
        v = output_backward[(s)*nb_channel+chan];
        output_backward[(s)*nb_channel+chan] = (v - signals_medians[chan]) / signals_mads[chan];
    }
"""

_extra_code_nomalize2 = """
    float v;
    for (int s=0; s<chunksize_2pad;s++){
        v = output[(s)*nb_channel+chan];
        output[(s)*nb_channel+chan] = (v - signals_medians[chan]) / signals_mads[chan];
    }
"""



signalpreprocessor_engines = { 'numpy' : SignalPreprocessor_Numpy,
                                                'opencl' : SignalPreprocessor_OpenCL}
