import numpy as np
import pandas as pd
import scipy.signal


class SignalFilter:
    def __init__(self, signals, highpass_freq = 300., box_smooth = 1, sampling_rate = None):
        """
        Class for hight filter.
        
        """
        self.signals = signals
        self.highpass_freq = float(highpass_freq)
        if sampling_rate is None:
            sampling_rate = 1./np.median(np.diff(signals.index[:1000]))
        self.sampling_rate = sampling_rate
        
        self.box_smooth = int(box_smooth)
        assert self.box_smooth%2==1, 'box_smooth must be odd'
        
        self.construct_coefficents()
    
    def construct_coefficents(self):
        if self.highpass_freq>0.:
            self.coefficients = scipy.signal.iirfilter(5, self.highpass_freq/self.sampling_rate*2, analog=False,
                                    btype = 'highpass', ftype = 'butter', output = 'sos')
            self.zi = scipy.signal.sosfilt_zi(self.coefficients)
            self.zi = np.repeat(self.zi[:,:,None], self.signals.shape[1], axis=2)
        
                                
    
    def get_filtered_data(self):
        #~ data = self.signals.values
        #~ data_filtered, self.zi = scipy.signal.sosfilt(self.coefficients, data, zi = self.zi, axis = 0)
        
        data_filtered = self.signals.values.copy()
        if self.highpass_freq>0.:
            for s in range(self.coefficients.shape[0]):
                b = self.coefficients[s, :3]
                a = self.coefficients[s, 3:]
                data_filtered = scipy.signal.filtfilt(b, a, data_filtered, axis = 0)
        
        if self.box_smooth>1:
            kernel = np.ones(self.box_smooth)/self.box_smooth
            kernel = kernel[:, None]
            data_filtered =  scipy.signal.fftconvolve(data_filtered,kernel,'same')
        
        
        return pd.DataFrame(data_filtered, index = self.signals.index, columns = self.signals.columns)
        