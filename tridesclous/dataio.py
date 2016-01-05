import os
import pandas as pd
import numpy as np
import json
from collections import OrderedDict

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


_signal_types = ['filtered', 'unfiltered']

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
        
        if 'segments_range' in self.store:
            self.segments_range = self.store['segments_range']
        else:
            columns = pd.MultiIndex.from_product([_signal_types, ['t_start', 't_stop']])
            self.segments_range = pd.DataFrame(columns = columns, dtype = 'float64')
        
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
        if self.segments_range is not None:
            return len(self.segments_range)
    
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
            for seg_num in self.segments_range.index:
                t += 'Segment {}\n'.format(seg_num)
                for signal_type in _signal_types:
                    t+= self.summary_segment(seg_num, signal_type = signal_type)
                
        return t
    
    def summary_segment(self, seg_num, signal_type = 'filtered'):
        t_start, t_stop = self.segments_range.loc[seg_num, (signal_type,'t_start')], self.segments_range.loc[seg_num, (signal_type,'t_stop')]
        t = """  {}
    duration : {}s.
    times range : {} - {}
""".format(signal_type, t_stop-t_start, t_start, t_stop)

        path = 'segment_{}/peaks'.format(seg_num)
        if path in self.store and self.store[path] is not None:
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
        
        self.flush_info()

    def flush_info(self):
        self.store['info'] = self.info
        self.store['segments_range'] = self.segments_range
        self.store.flush()
    
    def append_signals(self, signals,  seg_num=0, signal_type = 'filtered'):
        """
        Append signals in the store. The input is pd.dataFrame
        If the segment do not exist it is created in the store.
        Else the signals chunk is append to the previsous one.
        
        
        """
        assert isinstance(signals, pd.DataFrame), 'Signals must a DataFrame'
        
        path = 'segment_{}/{}_signals'.format(seg_num, signal_type)
        
        times = signals.index.values

        
        if seg_num in self.segments_range.index:
            #this check overlap if trying to write an already exisiting chunk
            # theorically this should work but the index will unefficient when self.store.select
            t_start = self.segments_range.loc[seg_num, (signal_type, 't_start')]
            t_stop = self.segments_range.loc[seg_num, (signal_type, 't_stop')]
            assert np.all(~((times>=t_start) & (times<=t_stop))), 'data already in store for seg_num {}'.format(seg_num)

        
        if self.info is None:
            sampling_rate = 1./np.median(np.diff(times[:1000]))
            self.initialize(sampling_rate = sampling_rate, channels = signals.columns.values)


        signals.to_hdf(self.store, path, format = 'table', append=True)
        
        if seg_num in self.segments_range.index:
            self.segments_range.loc[seg_num, (signal_type, 't_start')] = min( times[0], self.segments_range.loc[seg_num, (signal_type, 't_start')])
            self.segments_range.loc[seg_num,  (signal_type, 't_stop')] =  max(times[-1], self.segments_range.loc[seg_num,  (signal_type, 't_stop')])
        else:
            self.segments_range.loc[seg_num,  (signal_type, 't_start')] = times[0]
            self.segments_range.loc[seg_num,  (signal_type, 't_stop')] = times[-1]
        self.flush_info()
        
        self.store.create_table_index(path, optlevel=9, kind='full')
        
        
    
    def append_signals_from_numpy(self, signals, seg_num=0, sampling_rate = None, t_start = 0., signal_type = 'filtered', channels = None):
        """
        Append numpy.ndarray signals segment in the store.
        
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
        else:
            assert sampling_rate==self.info['sampling_rate']

        assert signals.shape[1]==self.info['nb_channel'], 'Wrong shape {} ({} chans)'.format(signals.shape, self.info['nb_channel'])
        assert sampling_rate == self.info['sampling_rate'], 'Wrong sampling_rate {} {}'.format(sampling_rate, self.info['sampling_rate'])
        
        times = np.arange(signals.shape[0], dtype = 'float64')/self.sampling_rate + t_start
        df = pd.DataFrame(signals, index = times, columns = self.info['channels'])
        
        self.append_signals(df,  seg_num=seg_num, signal_type = signal_type)
    
    def append_signals_from_neo(self, blocks, channel_indexes = None, signal_type = 'filtered'):
        """
        Append signals from a list of neo.Block.
        So all format readable by neo can be used.
        
        This loop over all neo.Segment inside neo.Block and append
        AnalogSignal into the datasets.
        
        
        Arguments
        ---------------
        blocks: list of neo.Block (or neo.Block
            data to append
        channel_indexes: list or None
            list of channel if None all channel are taken.
        
        """
        assert HAVE_NEO, 'neo must be installed'
        if isinstance(blocks, neo.Block):
            blocks = [blocks]
        
        if self.segments_range.index.size==0:
            seg_num=0
        else:
            seg_num= np.max(self.segments_range.index.values)+1
        
        for block in blocks:
            for seg in block.segments:
                if channel_indexes is not None:
                    analogsignals = []
                    for anasig in seg.analogsignals:
                        if anasig.channel_index in channel_indexes:
                            analogsignals.append(anasig)
                else:
                    analogsignals = seg.analogsignals
                
                if analogsignals[0].name is not None:
                    channels = [anasig.name for anasig in analogsignals]
                else:
                    channels = [ 'ch{}'.format(anasig.channel_index) for anasig in analogsignals]
                
                all_sr = [float(anasig.sampling_rate.rescale('Hz').magnitude) for anasig in analogsignals]
                all_t_start = [float(anasig.t_start.rescale('S').magnitude) for anasig in analogsignals]
                assert np.unique(all_sr).size == 1, 'Analogsignal do not have the same sampling_rate'
                assert np.unique(all_t_start).size == 1, 'Analogsignal do not have the same t_start'
                
                sigs = np.concatenate([anasig.magnitude[:, None] for anasig in analogsignals], axis = 1)
                
                sampling_rate = np.unique(all_sr)
                t_start = np.unique(all_t_start)
                
                self.append_signals_from_numpy(sigs, seg_num=seg_num, sampling_rate = sampling_rate, t_start = t_start,
                        signal_type = signal_type, channels = channels)
                
                seg_num += 1
    
    def get_signals(self, seg_num=0, t_start = None, t_stop = None, signal_type = 'filtered'):
        """
        Get a chunk of signals in the dataset.
        This internally use self.store.select from pandas.
        This should be efficient... and avoid loading everything in memory.
        
        Arguments
        -----------------
        seg_num: int
        
        """
        path = 'segment_{}/{}_signals'.format(seg_num, signal_type)
        
        if t_start is None and t_stop is None:
            query = None
        elif t_start is not None and t_stop is None:
            query = 'index>=t_start'
        elif t_start is None and t_stop is not None:
            query = 'index<t_stop'
        elif t_start is not None and t_stop is not None:
            query = 'index>=t_start & index<t_stop'
        
        return self.store.select(path, query)
    
    #~ def append_peaks(self, peaks, seg_num=0, append = False):
        #~ """
        #~ Append detected peaks in the store.
        
        #~ Arguments
        #~ -----------------
        #~ peaks: pd.DataFrame
            #~ DataFrame of peaks:
                #~ * index : MultiIndex (seg_num, peak_times)
                #~ * columns : 'peak_index', 'labels'
                #~ *
        #~ seg_num: int
            #~ The segment num.
        #~ append: bool, default True
            #~ If True append to existing peaks for the segment.
            #~ If False overwrite.
        #~ """
        
        #~ path = 'segment_{}/peaks'.format(seg_num)
        #~ peaks.to_hdf(self.store, path, format = 'table', append=append)
        
    
    #~ def get_peaks(self, seg_num=0):
        #~ path = 'segment_{}/peaks'.format(seg_num)
        #~ if path in self.store:
            #~ return self.store[path]
    
    def save_catalogue(self, catalogue, limit_left, limit_right):
        assert len(catalogue)>0, 'empty catalogue'
        index = list(catalogue.keys())
        columns = list(catalogue[index[0]].keys())
        catalogue2 = pd.DataFrame(index = index, columns = columns)
        
        for k in index:
            for col in columns:
                catalogue2.loc[k, col] = catalogue[k][col]
        
        path = 'catalogue'
        catalogue2.to_hdf(self.store, path, append = False)
        
        self.info['limit_left'] = limit_left
        self.info['limit_right'] = limit_right
        self.flush_info()
        
    def get_catalogue(self):
        path = 'catalogue'
        if path in self.store:
            catalogue = OrderedDict()
            catalogue2 = self.store['catalogue']
            for k in catalogue2.index:
                catalogue[k] = {}
                for col in catalogue2.columns:
                    catalogue[k][col] = catalogue2.loc[k, col]
            return catalogue, self.info['limit_left'], self.info['limit_right']
    
    def save_spiketrains(self, spiketrains, seg_num = 0):
        path = 'segment_{}/spiketrains'.format(seg_num)
        spiketrains.to_hdf(self.store, path, format = 'table', append=False)

    def get_spiketrains(self, seg_num = 0):
        path = 'segment_{}/spiketrains'.format(seg_num)
        if path in self.store:
            return self.store[path]


