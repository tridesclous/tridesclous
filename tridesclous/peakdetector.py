import numpy as np
import pandas as pd
import scipy.signal

"""


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
    norm_data = (signals - med)/mad
    return norm_data
    

def derivative_signals(signals):
    """
    Apply a derivate along time axis for each channel.
    
    """
    kernel = np.array([1,0,-1])/2.
    kernel = kernel[:,None]
    np_array = scipy.signal.fftconvolve(signals,kernel,'same')
    return pd.DataFrame(np_array, index = signals.index, columns = signals.columns)
     



class PeakDetector:
    """
    This is helper to estimated noise and threshold and detect peak on signals.
    It take as entry a DataFrame with signals given by DataManager.get_signals(...).    
    """
    def __init__(self, signals):
        self.signals = signals
        
