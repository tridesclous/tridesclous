import scipy.signal
import numpy as np


#~ from pyacq.dsp.overlapfiltfilt import SosFiltfilt_Scipy
from .tools import FifoBuffer

class SignalPreprocessor_Numpy:
    """
    This apply chunk by chunk on a multi signal:
       * baseline removal
       * hight pass filtfilt
       * normalize (optional)
    
    """
    def __init__(self,sample_rate, nb_channel, chunksize, input_dtype):
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.input_dtype = input_dtype
        
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
        start = pos-self.backward_chunksize
        if start<-self.overlapsize:
            return None, None
        if start>0:
            backward_chunk = self.forward_buffer.get_data(start,pos)
        else:
            backward_chunk = self.forward_buffer.get_data(0,pos)
        backward_filtered = scipy.signal.sosfilt(self.coefficients, backward_chunk[::-1, :], zi=None, axis=0)
        backward_filtered = backward_filtered[::-1, :]
        backward_filtered = backward_filtered.astype(self.output_dtype)
        if start>0:
            backward_filtered = backward_filtered[:self.chunksize]
        else:
            backward_filtered = backward_filtered[:-self.overlapsize]
        data2 = backward_filtered
        pos2 = pos-self.overlapsize
        #~ print('pos', pos, 'pos2', pos2, data2.shape)
        
        
        # removal ref
        if self.common_ref_removal:
            data2 -= np.median(data2, axis=1)[:, None]
        
        #normalize
        if self.normalize:
            data2 -= self.signals_medians
            data2 /= self.signals_mads
        
        return pos2, data2
    
    
    def change_params(self, common_ref_removal=True,
                                            highpass_freq=300.,
                                            output_dtype='float32', 
                                            normalize=True,
                                            backward_chunksize=None,
                                            signals_medians=None, signals_mads=None):
        self.signals_medians = signals_medians
        self.signals_mads = signals_mads
        
        self.common_ref_removal = common_ref_removal
        self.highpass_freq = highpass_freq
        self.output_dtype = np.dtype(output_dtype)
        self.normalize = normalize
        self.backward_chunksize = backward_chunksize
        
        assert self.backward_chunksize>self.chunksize
        
        self.coefficients = scipy.signal.iirfilter(5, highpass_freq/self.sample_rate*2, analog=False,
                                    btype = 'highpass', ftype = 'butter', output = 'sos')
        
        self.overlapsize = self.backward_chunksize - self.chunksize
        #~ self.filtfilt_engine = SosFiltfilt_Scipy(self.coefficients, self.nb_channel, output_dtype, self.chunksize, overlapsize)
        self.nb_section =self. coefficients.shape[0]
        self.forward_buffer = FifoBuffer((self.backward_chunksize, self.nb_channel), self.output_dtype)
        self.zi = np.zeros((self.nb_section, 2, self.nb_channel), dtype= self.output_dtype)
        
        
        



class SignalPreprocessor_OpenCL:
    def __init__(self):
        pass
    def process_data(self, pos, newbuf):
        pass
    def change_params(self,):
        pass


signalpreprocessor_engines = { 'signalpreprocessor_numpy' : SignalPreprocessor_Numpy,
                                                'signalpreprocessor_opencl' : SignalPreprocessor_OpenCL}
