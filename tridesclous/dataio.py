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
            #first init
            self.info = {}
            self.flush_info()
        else:
            with open(self.info_filename, 'r', encoding='utf8') as f:
                self.info = json.load(f)
            self.reload_existing()
            
        
        
    def flush_info(self):
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f, indent=4)
    
    def iter_over_chunk(self, seg_num=0, i_stop=None, chunksize=1024, **kargs):
        
        if i_stop is not None:
            length = min(self.get_segment_shape(seg_num)[0], i_stop)
        else:
            length = self.get_segment_shape(seg_num)[0]
        
        nloop = length//chunksize
        for i in range(nloop):
            i_stop = (i+1)*chunksize
            i_start = i_stop - chunksize
            sigs_chunk = self.get_signals_chunk(seg_num=seg_num, i_start=i_start, i_stop=i_stop, **kargs)
            yield  i_stop, sigs_chunk
    
    
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
    def __init__(self, dirname='test'):
        BaseDataIO.__init__(self, dirname=dirname)
    
    def set_initial_signals(self, filenames=[], dtype='int16', nb_channel=0,
                        sample_rate=0.):
    
        self.dtype = np.dtype(dtype)
        self.nb_channel= int(nb_channel)
        self.sample_rate = float(sample_rate)
        
        self.filenames = filenames
        if isinstance(self.filenames, str):
            self.filenames = [self.filenames]
        assert all([os.path.exists(f) for f in self.filenames]), 'files does not exist'
        
        self.info['filenames'] = self.filenames
        self.info['dtype'] = self.dtype.name
        self.info['nb_channel'] = self.nb_channel
        self.info['sample_rate'] = self.sample_rate
        self.flush_info()
        
        self._open_inital_data()
    
    def reload_existing(self):
        self.filenames = self.info['filenames']
        self.dtype = np.dtype(self.info['dtype'])
        self.nb_channel = self.info['nb_channel']
        self.sample_rate = self.info['sample_rate']
        
        self._open_inital_data()
    
    def _open_inital_data(self):
        self.nb_segment = len(self.filenames)
        
        self.initial_data = []
        for filename in self.filenames:
            data = np.memmap(filename, dtype=self.dtype, mode='r').reshape(-1, self.nb_channel)
            self.initial_data.append(data)
        
        self.segments_path = []
        for i in range(self.nb_segment):
            segment_path = os.path.join(self.dirname, 'segment_{}'.format(i))
            if not os.path.exists(segment_path):
                os.mkdir(segment_path)
                with open(os.path.join(segment_path, 'info.txt'), 'w', encoding='utf8') as f:
                    f.write(self.filenames[i])
            self.segments_path.append(segment_path)
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None, signal_type='initial',
                channels=None, return_type='raw_numpy'):
        
        if signal_type=='initial':
            data = self.initial_data[seg_num][i_start:i_stop, :]
        elif signal_type=='processed':
            raise(NotImplementedError)
        else:
            raise(ValueError, 'signal_type is not valide')
        
        if channels is not None:
            data = data[:, channels]
        
        if return_type=='raw_numpy':
            return data
        elif return_type=='on_scale_numpy':
            raise(NotImplementedError)
        elif return_type=='pandas':
            raise(NotImplementedError)
    
    def get_segment_shape(self, seg_num):
        return self.initial_data[seg_num].shape



