import numpy as np
import pandas as pd
import scipy.signal


"""
Some function for estimation of the noise and detection of peak.


"""


def normalize_signals(signals, med = None, mad = None):
    """
    Normalize signals = zscore but with median and mad.
    
    Arguments
    --------------
    signals: pandas.DataFrame
        Signals: index is time columns are channels
    med: optional
        Can give the median to avoid recomputation.
    mad: mad
        Can give the mad to avoid recomputation.
    
    Returns
    ----------
        pandas.DataFrame
        
    
    """
    if med is None:
        med = signals.median(axis=0)
    if mad is None:
        mad = np.median(np.abs(signals-med),axis=0)*1.4826
    normed_sigs = (signals - med)/mad
    return normed_sigs
    

def derivative_signals(signals):
    """
    Apply a derivate along time axis for each channel.
    
    """
    kernel = np.array([1,0,-1])/2.
    kernel = kernel[:,None]
    np_array = scipy.signal.fftconvolve(signals,kernel,'same')
    return pd.DataFrame(np_array, index = signals.index, columns = signals.columns)


def rectify_signals(normed_sigs, threshold, copy = True):
    """
    Return signals with 0 under threshold.
    Need normed_sigs as input
    """
    if copy:
        normed_sigs = normed_sigs.copy()
    if threshold<0.:
        normed_sigs[normed_sigs>threshold] = 0.
    else:
        normed_sigs[normed_sigs<threshold] = 0.
    return normed_sigs


def detect_peak_method_span(rectified_signals, peak_sign='-', n_span = 2):
    """
    Detect peak on rectified signals.
    When there are several peak it take the best ones in a short neighborhood.
    
    Argument
    --------------
    rectified_signals: pandas.dataFrame
        rectified signals see normalize_signals and rectify_signals
    peak_sign: '+' or '-'
        sign of the peak
    n_span : int
        Number of sample arround each side of the peak that exclude other smaller peak.
    
    Return
    ----------
    peaks_pos: np.array
        position in sample of peaks.
    """
    k = n_span
    sig = rectified_signals.sum(axis=1).values #np.array
    sig_center = sig[k:-k]
    if peak_sign == '+':
        peaks = sig_center>1.
        for i in range(k):
            peaks &= sig_center>sig[i:i+sig_center.size]
            peaks &= sig_center>=sig[k+i:k+i+sig_center.size]
    elif peak_sign == '-':
        peaks = sig_center<-1.
        for i in range(k):
            peaks &= sig_center<sig[i:i+sig_center.size]
            peaks &= sig_center<=sig[k+i:k+i+sig_center.size]
    peaks_pos,  = np.where(peaks)
    peaks_pos += k
    return peaks_pos





class PeakDetector:
    """
    This is helper to estimated noise and threshold and detect peak on signals.
    It take as entry a DataFrame with signals given by DataManager.get_signals(...).    
    """
    def __init__(self, signals):
        self.sigs = signals
        
        self.estimate_noise()
        self.normed_sigs = normalize_signals(self.sigs, med = self.med, mad = self.mad)
        #self.rectified_sigs = rectify_signals(self.normed_sigs, threshold, copy = True)
    
    def estimate_noise(self):
        """
        This compute median and mad of each channel.
        """
        self.med = self.sigs.median(axis=0)
        self.mad = np.median(np.abs(self.sigs-self.med),axis=0)*1.4826
    
    def detect_peaks(self, threshold = -5, peak_sign = '-', n_span = 2):
        self.threhold = threshold
        self.rectified_sigs = rectify_signals(self.normed_sigs, threshold, copy = True)
        
        self.peak_pos = detect_peak_method_span(self.rectified_sigs, peak_sign=peak_sign, n_span = n_span)
        self.peak_index = self.sigs.index[self.peak_pos]
        
        return self.peak_pos

