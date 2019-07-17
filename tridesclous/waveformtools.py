import numpy as np
import joblib


def extract_chunks(signals, indexes, width, chunks=None, n_jobs=0):
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
    
    if n_jobs == 0:
        for i, ind in enumerate(indexes):
            chunks[i,:,:] = signals[ind:ind+width,:]
    else:
        assert n_jobs >=1 
        n = indexes.size // n_jobs
        
        items = []
        for i in range(n_jobs):
            if i < n_jobs - 1:
                sl = slice(i*n, (i+1)*n)
            else:
                sl = slice(i*n, None)
            inds = indexes[sl]
            small_chunks = chunks[sl, :, :]
            args = (signals, inds, width)
            kargs = {'chunks': small_chunks, 'n_jobs' :0}
            items.append((args, kargs))
        
        joblib.Parallel(n_jobs=n_jobs, backend='threading', prefer='threads')(joblib.delayed(extract_chunks)(*args, **kargs) for args, kargs in items)

    return chunks

