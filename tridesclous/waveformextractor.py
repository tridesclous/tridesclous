import numpy as np
import pandas as pd


def extract_waveforms(signals, peak_pos, n_left, n_right):
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
    n_left, n_right: int, int
        Nb of sample arround the peak to extract.
        The waveform length is = 1 + n_left + n_rigth.
    
    Output
    ----------
    waveforms: pandas.dataFrame
        Waveforms extract. 
        index = peak_pos (cleaned)
        columns = multindex 2 level = channel X sample_pos in [-n_left:n_right]
    
    """
    peak_pos_clean = peak_pos[(peak_pos>n_left+1) & (peak_pos<signals.shape[0] -n_right - 1)]
    sample_index =  np.arange(-n_left, n_right+1, dtype = 'int64')
    columns = pd.MultiIndex.from_product([signals.columns,sample_index], 
                                                    names = ['channel', 'sample'])
    
    #~ peak_pos_clean = peak_pos_clean[0:10]

    waveforms = pd.DataFrame(index = peak_pos_clean, columns = columns)
    for p in peak_pos_clean:
        for chan in signals.columns:
            waveforms.loc[p, (chan, slice(None))] = signals.loc[:,chan].iloc[p-n_left:p+n_right+1].values
    
    # this solution is slower
    #~ waveforms2 = pd.DataFrame(index = peak_pos_clean, columns = columns)
    #~ for p in peak_pos_clean:
        #~ wf = signals.iloc[p-n_left:p+n_right+1,:]
        #~ wf.index = sample_index
        #~ waveforms2.loc[p, :] = wf.stack().swaplevel(0,1)

    return waveforms

