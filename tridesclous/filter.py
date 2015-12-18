import numpy as np
import pandas as pd
import scipy.signal


class SignalFilter:
    def __init__(self, signals, highpass_freq = 300., sampling_rate = None):
        """
        Class for hight filter.
        
        """
        self.signals = signals
        self.highpass_freq = float(highpass_freq)
        if sampling_rate is None:
            sampling_rate = 1./np.median(np.diff(signals.index[:1000]))
        self.sampling_rate = sampling_rate
        
        self.construct_coefficents()
    
    def construct_coefficents(self):
        
        self.coefficients = scipy.signal.iirfilter(7, self.highpass_freq/self.sampling_rate*2, analog=False,
                                btype = 'highpass', ftype = 'butter', output = 'sos')
        self.zi = scipy.signal.sosfilt_zi(self.coefficients)
        self.zi = np.repeat(self.zi[:,:,None], self.signals.shape[1], axis=2)
        
                                
    
    def get_filtered_data(self):
        data = self.signals.values
        data_filtered, self.zi = scipy.signal.sosfilt(self.coefficients, data, zi = self.zi, axis = 0)
        
        return pd.DataFrame(data_filtered, index = self.signals.index, columns = self.signals.columns)
        