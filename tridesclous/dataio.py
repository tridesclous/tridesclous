import os
import json
from collections import OrderedDict
import numpy as np
import pandas as pd


_signal_types = ['initial', 'processed']


class BaseDataIO:
    def __init__(self, dirname='test'):
        self.dirname = dirname
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        
        self.info_filename = os.path.join(self.dirname, 'info.json')
        if not os.path.exists(self.info_filename):
            self.info = {}
            self.flush_info()
        else:
            with open(self.info_filename, 'w', encoding='utf8') as f:
                self.info = json.load(f)
        
        
    def flush_info(self):
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f)
    
    def iter_over_chunk(self, seg_nums='all', ind_stop=None, chunksize=1024):
        if seg_nums=='all':
            seg_nums = range(self.nb_segments)
        
        for seg_num in seg_nums:
            get_segment_shape
            
    
    #~ def get_segment_shape(self, seg_num):
        #~ pass

    #~ @property
    #~ def dtype(self):
        #~ pass

    #~ @property
    #~ def sample_rate(self):
        #~ pass
        
    #~ @property
    #~ def nb_channel(self):
        #~ pass
    
    #~ @property
    #~ def nb_segments(self):
        #~ pass



class RawDataIO(BaseDataIO):
    """
    Each file is one segment in raw binary format.
    Each file is memmaped for simplicity.
    
    """
    def __init__(self, dirname='test', filenames=[], dtype='int16', nb_channel=0,
                        sample_rate=0.):
        BaseDataIO.__init__(self, dirname=dirname)

        self.dtype = np.dtype(dtype)
        self.nb_channel= int(nb_channel)
        self.sample_rate = float(sample_rate)
        
        self.filenames = filenames
        if isinstance(self.filenames, str):
            self.filenames = [self.filenames]
        assert all([os.path.exists(f) for f in self.filenames]), 'files does not exist'
        
        self.nb_segment = len(self.filenames)
        
        self.initial_data = []
        for filename in self.filenames:
            data = np.memmap(filename, dtype=self.dtype, mode='r', shape=(-1, self.nb_channel))
            self.initial_data.append(data)
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None, signal_type='initial',
                channels=None, return_type='raw_numpy'):
        
        if signal_type=='initial':
            data = self.initial_data[seg_num][i_start:i_stop]
        elif signal_type=='processed':
            raise(NotImplementedError)
        else:
            raise(ValueError, 'signal_type is not valide')
        
        
        data = data[:, channels]
        if return_type=='raw_numpy':
            return data
        elif return_type=='on_scale_numpy':
            raise(NotImplementedError)
        elif return_type=='pandas':
            raise(NotImplementedError)
    
    def get_segment_shape(self, seg_num):
        return self.initial_data[seg_num].shape



