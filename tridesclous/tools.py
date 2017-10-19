import numpy as np


def median_mad(data, axis=0):
    """
    Compute along axis the median and the mad.
    
    Arguments
    ----------------
    data : np.ndarray
    
    
    Returns
    -----------
    med: np.ndarray
    mad: np.ndarray
    
    
    """
    med = np.median(data, axis=axis)
    mad = np.median(np.abs(data-med),axis=axis)*1.4826
    return med, mad


def get_pairs_over_threshold(m, labels, threshold):
    """
    detect pairs over threhold in a similarity matrice
    """
    m = np.triu(m)
    ind0, ind1 = np.nonzero(m>threshold)
    
    #remove diag
    keep = ind0!=ind1
    ind0 = ind0[keep]
    ind1 = ind1[keep]
    
    pairs = list(zip(labels[ind0], labels[ind1]))
    
    return pairs
    


class FifoBuffer:
    """
    Kind of fifo on axis 0 than ensure to have the buffer and partial of previous buffer
    continuous in memory.
    
    This is not efficient if shape[0]is lot greater than chunksize.
    But if shape[0]=chunksize+smallsize, it should be OK.
    
    
    """
    def __init__(self, shape, dtype):
        self.buffer = np.zeros(shape, dtype=dtype)
        self.last_index = None
        
    def new_chunk(self, data, index):
        if self.last_index is not None:
            assert self.last_index+data.shape[0]==index
        
        n = self.buffer.shape[0]-data.shape[0]
        #roll the end
        
        self.buffer[:n] = self.buffer[-n:]
        self.buffer[n:] = data
        self.last_index = index
    
    def get_data(self, start, stop):
        start = start - self.last_index + self.buffer .shape[0]
        stop = stop - self.last_index + self.buffer .shape[0]
        assert start>=0
        assert stop<=self.buffer.shape[0]
        return self.buffer[start:stop]
