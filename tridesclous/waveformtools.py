import numpy as np
import joblib





def extract_chunks(signals, sample_indexes, width, channel_adjacency=None,  channel_indexes=None, chunks=None, n_jobs=0):
    """
    This cut small chunks on signals and return concatenate them.
    This use numpy.array for input/output.
    
    
    Arguments
    ---------------
    signals: np.ndarray 
        shape (nb_campleXnb_channel)
    sample_indexes: np.ndarray
        sample postion of the first sample
    width: int
        Width in sample of chunks.
    channel_adjacency: dict
        dict of channel adjacency
        If not None, then the wavzform extraction is sprase given adjency.
        In that case channel_indexes must be provide
    channel_indexes: np.array None
        Position in channels space
        
    Returns
    -----------
    chunks : np.ndarray
        shape = (sample_indexes.size, width, signals.shape[1], )
    
    """
    if chunks is None:
        chunks = np.empty((sample_indexes.size, width, signals.shape[1]), dtype = signals.dtype)
        
    if n_jobs == 0:
        keep = (sample_indexes>=0) & (sample_indexes<(signals.shape[0] - width))
        sample_indexes2 = sample_indexes[keep]
        if channel_indexes is not None:
            channel_indexes2 = channel_indexes[keep]
        
        if channel_adjacency is None:
            # all channels
            for i, ind in enumerate(sample_indexes2):
                chunks[i,:,:] = signals[ind:ind+width,:]
        else:
            # sparse
            assert channel_indexes is not None, 'For sparse eaxtraction channel_indexes must be provide'
            for i, ind in enumerate(sample_indexes2):
                chan = channel_indexes2[i]
                chans = channel_adjacency[chan]
                chunks[i,:,:][:, chans] = signals[ind:ind+width,:][:, chans]
    else:
        # this is totally useless because not faster....
        assert n_jobs >=1 
        n = sample_indexes.size // n_jobs
        
        items = []
        for i in range(n_jobs):
            if i < n_jobs - 1:
                sl = slice(i*n, (i+1)*n)
            else:
                sl = slice(i*n, None)
            inds = sample_indexes[sl]
            small_chunks = chunks[sl, :, :]
            args = (signals, inds, width)
            kargs = {'chunks': small_chunks, 'n_jobs' :0, 'channel_adjacency':channel_adjacency, 'channel_indexes':channel_indexes}
            items.append((args, kargs))
        
        joblib.Parallel(n_jobs=n_jobs, backend='threading', prefer='threads')(joblib.delayed(extract_chunks)(*args, **kargs) for args, kargs in items)

    return chunks

