import os
import pandas as pd
import numpy as np
import json


class DataIO:
    """
    This class handle data for each step of spike sorting: raw data, filtered data, peaks position,
    waveform, ....
    
    All data are copy in a directory in several hdf5 (pandas+pytables) files so a spike sorting
    can be continued/verifiyed later on without need to recompute all steps.
    
    Internally the hdf5 (pytables formated) contains:
    'info' : a Series that contains sampling_rate, nb_channels, ...
    ''segment_0/unfiltered_signals' : non filetred signals of segment 0
    ''segment_0/signals' : filetred signals of segment 0
    
    Usage:
    
    dataio = DataIO(dirname = 'test', complib = 'blosc', complevel= 9)
    
    
    """
    def __init__(self, dirname = 'test', complib = 'blosc', complevel= 9):
        self.dirname = dirname
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        self.data_filename = os.path.join(self.dirname, 'data.h5')
        if not os.path.exists(self.data_filename):
            #TODO some initilisation_list
            pass
        
        self.store = pd.HDFStore(self.data_filename, complib = complib, complevel = complevel,  mode = 'a')
        
        if 'info' in self.store:
            self.info = self.store['info']
        else:
            self.info = None
        
        if 'segments' in self.store:
            self.segments = self.store['segments']
        else:
            self.segments = None
    
    @property
    def sampling_rate(self):
        if self.info is not None:
            return float(self.info['sampling_rate'])

    @property
    def nb_channel(self):
        if self.info is not None:
            return int(self.info['nb_channel'])

    @property
    def nb_segments(self):
        if self.segments is not None:
            return len(self.segments)
    
    def summary(self, level=1):
        t = """DataIO <{}>
Workdir: {}""".format(id(self), self.data_filename)
        if self.info is None:
            t  += "\nNot initialized"
            return t
        t += """
sampling_rate: {}
nb_channel: {}
nb_segments: {}""".format(self.sampling_rate, self.nb_channel, self.nb_segments)
        if level==0:
            return t
        
        if level==1:
            t += "\n"
            for seg_num in self.segments.index:
                t+= self.summary_segment(seg_num)
                
        return t
    
    def summary_segment(self, seg_num):
        t_start, t_stop = self.segments.loc[seg_num, 't_start'], self.segments.loc[seg_num, 't_stop']
        t = """Segment {}
    duration : {}s.
    times range : {} - {}
""".format(seg_num, t_stop-t_start, t_start, t_stop)

        path = 'segment_{}/peaks'.format(seg_num)
        if path in self.store:
            t+= "    nb_peaks : {}\n".format(self.store[path].shape[0])
        
        return t
    
    def __repr__(self):
        return self.summary(level=0)
    
    def initialize(self, sampling_rate = None, channels = None):
        """
        Initialise the Datamanager.
        
        Arguments
        -----------------
        sampling_rate: float
            The sampling rate in Hz
        channels: list of str
            Channel labels
        """
        assert sampling_rate is not None, 'You must provide a sampling rate'
        assert channels is not None, 'You provide a list of channels'
        
        if self.info is None:
            self.info = pd.Series()
        self.info['sampling_rate'] = sampling_rate
        self.info['channels'] = np.array(channels, dtype = 'S')
        self.info['nb_channel'] = len(channels)
        
        if self.segments is None:
            self.segments = pd.DataFrame(columns = ['t_start', 't_stop'], dtype = 'float64')
        
        self.flush_info()

    def flush_info(self):
        print('flush_info')
        print(self.info)
        self.store['info'] = self.info
        self.store['segments'] = self.segments
        self.store.flush()
        
    def append_signals(self, signals, seg_num=0, sampling_rate = None, t_start = 0., already_hp_filtered = False, channels = None):
        """
        Append signals segment in the store.
        If the segment do not exist it is created in the store.
        Else the signals chunk is append to the previsous one.
        
        Arguments
        -----------------
        signals: np.ndarray
            Signals 2D array shape (nb_sampleXnb_channel).
        seg_num: int
            The segment num.
        t_start: float 
            Time stamp of the first sample.
        channels : list of str
            Channels labels
        
        """
        if signals.ndim==1:
            signals = signals[:, None]
        
        if self.info is None:
            self.initialize(sampling_rate = sampling_rate, channels = channels)

        assert signals.shape[1]==self.info['nb_channel'], 'Wrong shape {} ({} chans)'.format(signals.shape, self.info['nb_channel'])
        assert sampling_rate == self.info['sampling_rate'], 'Wrong sampling_rate {} {}'.format(sampling_rate, self.info['sampling_rate'])
        
        times = np.arange(signals.shape[0], dtype = 'float64')/self.sampling_rate + t_start
        df = pd.DataFrame(signals, index = times, columns = self.info['channels'])
        
        path = 'segment_{}'.format(seg_num)
        if already_hp_filtered:
            path += '/signals'
        else:
            path += '/unfiltered_signals'
        
        if seg_num in self.segments.index:
            #this check overlap if trying to write an already exisiting chunk
            # theorically this should work but the index will unefficient when self.store.select
            assert np.all(~((times>=self.segments.loc[seg_num, 't_start']) & (times<=self.segments.loc[seg_num, 't_stop']))), 'data already in store for seg_num {}'.format(seg_num)
            
        df.to_hdf(self.store, path, format = 'table', append=True)
        
        if seg_num in self.segments.index:
            self.segments.loc[seg_num, 't_start'] = min(self.segments.loc[seg_num, 't_start'], times[0])
            self.segments.loc[seg_num, 't_stop'] =  max(times[-1], self.segments.loc[seg_num, 't_stop'])
        else:
            self.segments.loc[seg_num, 't_start'] = times[0]
            self.segments.loc[seg_num, 't_stop'] = times[-1]
        self.flush_info()
        
        self.store.create_table_index(path, optlevel=9, kind='full')

    def get_signals(self, seg_num=0, t_start = None, t_stop = None, filtered = True):
        """
        Get a chunk of signals in the dataset.
        This internally use self.store.select from pandas.
        This should be efficient... and avoid loading everything in memory.
        
        Arguments
        -----------------
        seg_num: int
        
        """
        path = 'segment_{}'.format(seg_num)
        if filtered:
            path += '/signals'
        else:
            path += '/unfiltered_signals'
        
        if t_start is None and t_stop is None:
            query = None
        elif t_start is not None and t_stop is None:
            query = 'index>=t_start'
        elif t_start is None and t_stop is not None:
            query = 'index<t_stop'
        elif t_start is not None and t_stop is not None:
            query = 'index>=t_start & index<t_stop'
        
        return self.store.select(path, query)
    
    def append_peaks(self, peaks, seg_num=0, append = False):
        """
        Append detected peaks in the store.
        
        Arguments
        -----------------
        peaks: pd.DataFrame
            DataFrame of peaks:
                * index : MultiIndex (seg_num, peak_times)
                * columns : 'peak_index', 'labels'
                *
        seg_num: int
            The segment num.
        append: bool, default True
            If True append to existing peaks for the segment.
            If False overwrite.
        """
        
        path = 'segment_{}/peaks'.format(seg_num)
        peaks.to_hdf(self.store, path, format = 'table', append=append)
        
    
    def get_peaks(self, seg_num=0):
        path = 'segment_{}/peaks'.format(seg_num)
        return self.store[path]
        
        
    
