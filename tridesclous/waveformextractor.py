import numpy as np
import pandas as pd

from .tools import median_mad

def cut_chunks(signals, indexes, width):
    """
    This cut small chunks on signals and return concatenate them.
    This use numpy.array for input/output.
    
    extract_peak_waveforms and extract_noise_waveforms ar higher level fonction
    than use dataFrame as Input/Ouput
    
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


def extract_peak_waveforms(signals, peak_pos, peak_index, n_left, n_right):
    """
    Extract waveforms around peak given signals and peak position (in sample).
    Note that peak_pos near border are eliminated, 
    so waveforms.index is or is NOT peak_pos depending if there are spike to near borders.
    
    Arguments
    ---------------
    signals : pandas.DataFrame
        Signals
    peak_pos : np.ndarray
        Vector cthat contains peak position in index.
    peak_index : pandas.Index
        Index of peak for output can be depending of the needs:
            * multiindex 2 levels = (seg_num, peak_time) 
            * or peak_time
            * or peak_pos
    n_left, n_right: int, int
        Nb of sample arround the peak to extract.
        The waveform length is = - n_left + n_right.
        n_left is negative, n_right is positve
    
    Output
    ----------
    waveforms: pandas.dataFrame
        Waveforms extract. 
        index = peak_index
        columns = multindex 2 level = (channel,sample_pos in [n_left:n_right])
    
    """
    assert n_left<0
    assert n_right>0
    
    keep = (peak_pos>-n_left+1) & (peak_pos<signals.shape[0] -n_right - 1)
    peak_pos_clean = peak_pos[keep]
    sample_index =  np.arange(n_left, n_right, dtype = 'int64')
    columns = pd.MultiIndex.from_product([signals.columns,sample_index], 
                                                    names = ['channel', 'sample'])
    
    chunks = cut_chunks(signals.values, peak_pos_clean+n_left, - n_left + n_right)
    chunks= chunks.reshape(chunks.shape[0], -1)
    
    waveforms = pd.DataFrame(chunks, index = peak_index[keep], columns = columns)
    
    # this solution is slower
    #~ waveforms = pd.DataFrame(index = peak_pos_clean, columns = columns)
    #~ for p in peak_pos_clean:
        #~ for chan in signals.columns:
            #~ waveforms.loc[p, (chan, slice(None))] = signals.loc[:,chan].iloc[p+n_left:p+n_right+1].values
    
    # this solution is even more slower
    #~ waveforms2 = pd.DataFrame(index = peak_pos_clean, columns = columns)
    #~ for p in peak_pos_clean:
        #~ wf = signals.iloc[p+n_left:p+n_right+1,:]
        #~ wf.index = sample_index
        #~ waveforms2.loc[p, :] = wf.stack().swaplevel(0,1)
    
    
    return waveforms

def extract_noise_waveforms(signals, peak_pos, n_left, n_right, size=1000, safety_factor=2): 
    """
    Extract waveform in between peaks (in the 'noise' ).
    Similar to extract_peak_waveforms.
    You must provide the number of events.
    event position are randomly selected with no overlap with peaks.
    
    
    Arguments
    ---------------
    signals : pandas.DataFrame
        Signals
    peak_pos : np.ndarray
        Vector cthat contains peak position in index.
    n_left, n_right: int, int
        Nb of sample arround the peak to extract.
        The waveform length is =  - n_left + n_right.
        n_left is negative, n_right is positve
    size: int
        Nb of events.
    
    Output
    ----------
    waveforms: pandas.dataFrame
        Waveforms extract. 
        index = peak_pos (cleaned)
        columns = multindex 2 level = channel X sample_pos in [n_left:n_right]

    """
    assert n_left<0
    assert n_right>0
    
    limit1 = peak_pos + n_left
    limit2 = peak_pos + n_right + 1
    
    positions = np.random.randint(low = -n_left, high = signals.shape[0]-n_right-1, size = size)
    for i, pos in enumerate(positions):
        while  np.any((pos+n_right + 1 + safety_factor>=limit1) & (pos<=limit2)|(pos>=limit1) & (pos+n_left-safety_factor<=limit2)):
            pos = np.random.randint(low = -n_left, high = signals.shape[0]-n_right-1)
            positions[i] = pos
    
    sample_index =  np.arange(n_left, n_right, dtype = 'int64')
    columns = pd.MultiIndex.from_product([signals.columns,sample_index], 
                                                    names = ['channel', 'sample'])
    chunks = cut_chunks(signals.values, positions+n_left,  - n_left + n_right)
    chunks= chunks.reshape(chunks.shape[0], -1)
    waveforms = pd.DataFrame(chunks, index = positions, columns = columns)
    
    return waveforms


def good_events(waveforms,  upper_thr=6.,lower_thr=-8., med = None, mad = None):
    """
    Are individual events 'close enough' to the median event?
    
    Parameters
    ----------
    waveforms: pandas.DataFrame
        waveforms
    upper_thr   a positive number, by how many time the MAD is the event allow to
                deviate from the median in the positive direction?
    lower_thr   a negative number, by how many time the MAD is the event allow to
                deviate from the median in the negative direction? If None (default)
                lower_thr is set to -upper_thr    
    med: None or np.array
        Already precomptued median (avoid recomputation)
    mad :None or np.array
        Already precomptued mad (avoid recomputation)
    
    Returns
    -------
        A Boolean vector whose elements are True if the event is 'good' and False otherwise.
    
    """
    if lower_thr is None:
        lower_thr = -upper_thr
    assert upper_thr>=0, 'upper_thr must be positive'
    assert lower_thr<=0, 'lower_thr must be negative'

    if med is None:
        med = waveforms.median(axis=0)
    if mad is None:
        mad = np.median(np.abs(waveforms-med),axis=0)*1.4826
    
    normed = (waveforms-med)/mad
    
    # any is faster that all
    #keep = np.all((normed>lower_thr)& (normed<upper_thr), axis=1)
    keep = ~(np.any(normed<lower_thr, axis=1) | np.any(normed>upper_thr, axis=1))
    
    return keep



def find_good_limits(normed_mad, mad_threshold = 1.1):
    """
    Find goods limits for the waveform.
    Where the MAD is above noise level (=1.)
    
    The technics constists in finding continuous samples above 10% of backgroud noise.
    
    Parameters
    ----------
    normed_mad: np.ndarray
        MAD for each waveform by channel.
        Already normed by channel.
        shape = (nb_channel, nb_sample)
    
    Returns
    ----------
    limit1, limi2: int
        Left and rigth limits
        Both are positive. They are relative to chunk (not peak).
    """
    # any channel above MAD mad_threshold
    above = np.any(normed_mad>=mad_threshold, axis=0)
    #find max consequitive point that are True
    
    up, = np.where(np.diff(above.astype(int))==1)
    down, = np.where(np.diff(above.astype(int))==-1)
    up = up[up<max(down)]
    down = down[down>min(up)]
    
    best = np.argmax(down-up)
    
    return up[best], down[best]+1




class WaveformExtractor_:
    def __init__(self, peakdetector, n_left=30, n_right=45):
        """
        
        """
        assert hasattr(peakdetector, 'peak_pos'), 'peakdetector must execute first PeakDetector.detect_peaks(...)'
        
        self.peakdetector = peakdetector
        self.n_left = n_left
        self.n_right = n_right
        self.seg_num = self.peakdetector.seg_num

        #work on normed signals
        self.signals = self.peakdetector.normed_sigs
        self.nb_channel = self.signals.shape[1]

        
        #Initial waveform extraction with bigger chunk
        self.long_waveforms = extract_peak_waveforms(self.signals, self.peakdetector.peak_pos,self.peakdetector.peak_index, n_left, n_right)
        self.med, self.mad = median_mad(self.long_waveforms, axis=0)
        
        #~ self.med = self.long_waveforms.median(axis=0)
        #~ self.mad = np.median(np.abs(self.long_waveforms-self.med),axis=0)*1.4826
    
    def good_events(self, upper_thr=6.,lower_thr=-8.,):
        self.keep = good_events(self.long_waveforms,  upper_thr=upper_thr,lower_thr=lower_thr, med = self.med, mad = self.mad)
        return self.keep
    
    def extract_noise(self, n_left, n_right, size=1000, safety_factor=2):
        # take some noise
        self.noises = extract_noise_waveforms(self.signals, self.peakdetector .peak_pos, n_left, n_right, size=size, safety_factor=safety_factor)
        return self.noises
    
    def find_good_limits(self, mad_threshold = 1.1):
        l1, l2 = find_good_limits(self.mad.values.reshape(self.nb_channel,-1), mad_threshold = mad_threshold)
        self.limit_left = self.long_waveforms.columns.levels[1][l1]
        self.limit_right = self.long_waveforms.columns.levels[1][l2]
        return self.limit_left, self.limit_right
    
    def get_ajusted_waveforms(self, margin=2):
        """
        Get ajusted waveform : between limit_left-margin and limit_right+margin.
        The margin of 2 sample is to get first and second derivative waveform to construct the catalogue.
        """
        sub = np.arange(self.limit_left-2, self.limit_right+2)
        short_waveforms = self.long_waveforms.loc[:, (slice(None), sub)]
        #reconstruct the real sub indexing
        # see http://pandas.pydata.org/pandas-docs/stable/advanced.html   (Basic indexing on axis with MultiIndex)
        short_waveforms.columns = pd.MultiIndex.from_tuples(short_waveforms.columns.values)
        return short_waveforms


from .mpl_plot import WaveformExtractorPlot
class WaveformExtractor(WaveformExtractor_, WaveformExtractorPlot):
    pass



