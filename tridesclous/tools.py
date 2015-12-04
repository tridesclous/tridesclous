import pandas as pd
import numpy as np


def median_mad(df, axis=0):
    """
    Compute along axis the median and the med.
    Note: median is already included in pandas (df.median()) but not the mad
    This take care of constructing a Series for the mad.
    
    Arguments
    ----------------
    df : pandas.DataFrame
    
    
    Returns
    -----------
    med: pandas.Series
    mad: pandas.Series
    
    
    """
    med = df.median(axis=axis)
    mad = np.median(np.abs(df-med),axis=axis)*1.4826
    mad = pd.Series(mad, index = med.index)
    return med, mad
 
    