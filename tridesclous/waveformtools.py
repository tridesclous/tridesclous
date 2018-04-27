import numpy as np


def extract_chunks(signals, indexes, width, chunks=None):
    """
    This cut small chunks on signals and return concatenate them.
    This use numpy.array for input/output.
    
    
    Arguments
    ---------------
    signals: np.ndarray 
        shape (nb_campleXnb_channel)
    indexes: np.ndarray
        sample postion of the first sample
    width: int
        Width in sample of chunks.
    Returns
    -----------
    chunks : np.ndarray
        shape = (indexes.size, width, signals.shape[1], )
    
    """
    if chunks is None:
        chunks = np.empty((indexes.size, width, signals.shape[1]), dtype = signals.dtype)
    for i, ind in enumerate(indexes):
        chunks[i,:,:] = signals[ind:ind+width,:]
    return chunks

