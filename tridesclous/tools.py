import numpy as np
import sklearn.metrics.pairwise
import re

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


def get_neighborhood(geometry, radius_um):
    """
    get neighborhood given geometry array and radius
    
    params
    -----
    geometry: numpy array (nb_channel, 2) intresect units ar micro meter (um)
    
    radius_um: radius in micro meter
    
    returns
    ----
    
    neighborhood: boolean numpy array (nb_channel, nb_channel)
    
    """
    d = sklearn.metrics.pairwise.euclidean_distances(geometry)
    return d<=radius_um
    

def fix_prb_file_py2(probe_filename):
    """
    prb file can be define in python2
    unfortunatly some of them are done with python
    and this is not working in python3
    range(0, 17) + range(18, 128)
    
    This script tryp to change range(...) by list(range(...)))
    """
    with open(probe_filename, 'rb') as f:
        prb = f.read()
    

    pattern = b"list\(range\([^()]*\)\)"
    already_ok = re.findall( pattern, prb)
    if len(already_ok)>0:
        return
    #~ print(already_ok)


    pattern = b"range\([^()]*\)"
    changes = re.findall( pattern, prb)
    for change in changes:
        prb = prb.replace(change, b'list('+change+b')')

    with open(probe_filename, 'wb') as f:
        f.write(prb)
