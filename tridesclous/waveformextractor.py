import numpy as np




class WaveformExtractor:
    """
    Class for extracting waveforms around peak on the fly.
    This class deal with peaks that are near the border.
    
    
    """
    
    def __init__(self,  nb_channel, chunksize):
        self.nb_channel = nb_channel
        self.chunksize = chunksize
        
        
    
    def new_peaks(self, pos, sigs_chunk, chunk_peak_pos):
        
        if pos==self.chunksize and chunk_peak_pos is not None:
            # remove the very first peak is too near the border the left of first chunk
            chunk_peak_pos = chunk_peak_pos[chunk_peak_pos>-self.n_left+1]
        
        if len(self.peak_at_right_border) >0 and self.last_sigs_chunk is not None:
            #~ print('pos', pos)
            #~ print('self.peak_at_right_border', self.peak_at_right_border)
            #~ print(self.peak_at_right_border-pos+self.chunksize*2+self.n_left)
            chunk_waveforms = cut_on_border(self.last_sigs_chunk, sigs_chunk, 
                                self.peak_at_right_border-pos+self.chunksize*2+self.n_left, self.peak_width)
            #~ print('right border')
            yield self.peak_at_right_border, chunk_waveforms
        
        pos2 = chunk_peak_pos - pos + self.chunksize
        on_left = pos2<=-self.n_left
        on_right = pos2>=self.chunksize - self.n_right
        keep = ~on_left & ~on_right
        
        # peak on left border
        if np.any(on_left) and self.last_sigs_chunk is not None:
            chunk_waveforms = cut_on_border(self.last_sigs_chunk, sigs_chunk, 
                                    pos2[on_left]+self.chunksize+self.n_left, self.peak_width)
            #~ print('left border')
            yield chunk_peak_pos[on_left], chunk_waveforms
        
        # for next call
        self.peak_at_right_border = chunk_peak_pos[on_right]
        self.last_sigs_chunk = sigs_chunk
        
        # peak not near border
        chunk_waveforms = cut_full(sigs_chunk, pos2[keep]+self.n_left, self.peak_width)
        #~ print('no border')
        yield chunk_peak_pos[keep] , chunk_waveforms
    
    
    def change_params(self, n_left=-20, n_right=30):
        self.n_left = n_left
        self.n_right = n_right
        self.peak_width = - n_left + n_right
        
        self.last_sigs_chunk = None
        self.peak_at_right_border = []




def cut_full(signals, indexes, width):
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
        shape = (indexes.size, signals.shape[1], width)
    
    """
    chunks = np.empty((indexes.size, signals.shape[1], width), dtype = signals.dtype)
    for i, ind in enumerate(indexes):
        chunks[i,:,:] = signals[ind:ind+width,:].transpose()
    return chunks


def cut_on_border(left_sigs, right_sigs, indexes, width):
    chunks = np.empty((indexes.size, left_sigs.shape[1], width), dtype = left_sigs.dtype)
    for i, ind in enumerate(indexes):
        #left 
        
        l1 = left_sigs.shape[0] - ind
        l2 = width - l1
        chunks[i,:,:l1] = left_sigs[ind:ind+l1,:].transpose()
        #right
        chunks[i,:,l1:] = right_sigs[:l2,:].transpose()
    return chunks
    
