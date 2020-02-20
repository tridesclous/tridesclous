import numpy as np
import joblib





def extract_chunks(signals, peak_sample_indexes, width, 
                channel_indexes=None, channel_adjacency=None, 
                peak_channel_indexes=None, chunks=None):
    """
    This cut small chunks on signals and return concatenate them.
    This use numpy.array for input/output.
    
    
    Arguments
    ---------------
    signals: np.ndarray 
        shape (nb_campleXnb_channel)
    peak_sample_indexes: np.ndarray
        sample postion of the first sample
    width: int
        Width in sample of chunks.
    channel_indexes: 
        Force a limited number of channels. If None then all channel.
    
    channel_adjacency: dict
        dict of channel adjacency
        If not None, then the wavzform extraction is sprase given adjency.
        In that case peak_channel_indexes must be provide and channel_indexes 
        must be None.
    peak_channel_indexes: np.array None
        Position of the peak on channel space used only when channel_adjacency not None.
        
    Returns
    -----------
    chunks : np.ndarray
        shape = (peak_sample_indexes.size, width, signals.shape[1], )
    
    """
    if channel_adjacency is not None:
        assert channel_indexes is None
        assert peak_channel_indexes is not None, 'For sparse eaxtraction peak_channel_indexes must be provide'
    
    if chunks is None:
        if channel_indexes is None:
            chunks = np.empty((peak_sample_indexes.size, width, signals.shape[1]), dtype = signals.dtype)
        else:
            chunks = np.empty((peak_sample_indexes.size, width, len(channel_indexes)), dtype = signals.dtype)
        
    keep = (peak_sample_indexes>=0) & (peak_sample_indexes<(signals.shape[0] - width))
    peak_sample_indexes2 = peak_sample_indexes[keep]
    if peak_channel_indexes is not None:
        peak_channel_indexes2 = peak_channel_indexes[keep]
    
    if channel_indexes is None and channel_adjacency is None:
        # all channels
        for i, ind in enumerate(peak_sample_indexes2):
            chunks[i,:,:] = signals[ind:ind+width,:]
    elif channel_indexes is not None:
        for i, ind in enumerate(peak_sample_indexes2):
            chunks[i,:,:] = signals[ind:ind+width,:][:, channel_indexes]
            
    elif channel_adjacency is not None:
        # sparse
        assert peak_channel_indexes is not None
        for i, ind in enumerate(peak_sample_indexes2):
            chan = peak_channel_indexes2[i]
            chans = channel_adjacency[chan]
            chunks[i,:,:][:, chans] = signals[ind:ind+width,:][:, chans]

    return chunks

