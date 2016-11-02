import scipy.signal
import numpy as np


from pyacq.dsp.overlapfiltfilt import SosFiltfilt_Scipy


class SignalPreprocessor_Numpy:
    def __init__(self,sample_rate, nb_channel, chunksize, input_dtype):
        self.sample_rate = sample_rate
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        self.input_dtype = input_dtype
        
    def process_data(self, pos, data):
        pos2, data2 = self.filtfilt_engine.compute_one_chunk(pos, data)
        
        if pos2 is None:
            return None, None
        
        data2 -= self.medians
        data2 /= self.mads
        
        return pos2, data2
    
    
    def change_params(self, highpass_freq=300., output_dtype='float32', 
                                            backward_chunksize=None,
                                             medians=None, mads=None):
        self.medians = medians
        self.mads = mads
        self.highpass_freq = highpass_freq
        self.output_dtype = output_dtype
        self.backward_chunksize = backward_chunksize
        
        assert self.backward_chunksize>self.chunksize
        
        self.coefficients = scipy.signal.iirfilter(5, highpass_freq/self.sample_rate*2, analog=False,
                                    btype = 'highpass', ftype = 'butter', output = 'sos')
        
        overlapsize = self.backward_chunksize - self.chunksize
        self.filtfilt_engine = SosFiltfilt_Scipy(self.coefficients, self.nb_channel, output_dtype, self.chunksize, overlapsize)
        
        



class SignalPreprocessor_OpenCL:
    def __init__(self):
        pass
    def process_data(self, pos, newbuf):
        pass
    def change_params(self,):
        pass


signalpreprocessor_engines = { 'signalpreprocessor_numpy' : SignalPreprocessor_Numpy,
                                                'signalpreprocessor_opencl' : SignalPreprocessor_OpenCL}
