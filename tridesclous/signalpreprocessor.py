import scipy.signal
import numpy as np


from pyacq.dsp.overlapfiltfilt import SosFiltfilt_Scipy


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
        data = data.astype(self.output_dtype)
        
        
        pos2, data2 = self.filtfilt_engine.compute_one_chunk(pos, data)
        #TODO this cause problem for peakdetector_opencl
        # because pos is not multiple  chunksize

        
        if pos2 is None:
            return None, None

        if self.common_ref_removal:
            data2 -= np.median(data2, axis=1)[:, None]
        
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
        self.output_dtype = output_dtype
        self.normalize = normalize
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
