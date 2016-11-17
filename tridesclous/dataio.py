import os
import json
from collections import OrderedDict
import numpy as np
import pandas as pd

from .iotools import ArrayCollection

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
            try:
                self.reload_info()
                self.reload_existing()
            except:
                self.info = {}
                self.flush_info()
                
    
    def flush_info(self):
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f, indent=4)
    
    def reload_info(self):
        self.channel_group = self.info['channel_group']
        self.nb_channel  = len(self.channel_group)
    
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
    
    def set_channel_group(self, channel_group=[]):
        if isinstance(channel_group, np.ndarray):
            channel_group =channel_group.tolist()
        self.channel_group = list(channel_group)
        self.info['channel_group'] = self.channel_group
        self.nb_channel  = len(self.channel_group)
        self.info['nb_channel'] = self.nb_channel
        self.flush_info()
    
    #reload_existing
    #get_signals_chunk
    #reset_signals
    #flush_signals
    #get_segment_shape
    #reset_spikes
    #append_spikes
    #flush_spikes
    #get_spikes

    

class RawDataIO(BaseDataIO):
    """
    Each file is one segment in raw binary format.
    Each file is memmaped for simplicity.
    
    """
    def __init__(self, dirname='test'):
        BaseDataIO.__init__(self, dirname=dirname)
    
    def set_initial_signals(self, filenames=[], dtype='int16', total_channel=0,
                        sample_rate=0.):
    
        self.dtype = np.dtype(dtype)
        self.total_channel= int(total_channel)
        self.sample_rate = float(sample_rate)
        
        self.filenames = filenames
        if isinstance(self.filenames, str):
            self.filenames = [self.filenames]
        assert all([os.path.exists(f) for f in self.filenames]), 'files does not exist'
        
        self.info['filenames'] = self.filenames
        self.info['dtype'] = self.dtype.name
        self.info['total_channel'] = self.total_channel
        self.info['sample_rate'] = self.sample_rate
        #~ self.info['dtype_processed'] = None
        self.flush_info()
        
        self.set_channel_group(np.arange(self.total_channel))
        
        self._open_inital_data()
        self.processed_data = []
    
    def reload_existing(self):
        self.filenames = self.info['filenames']
        self.dtype = np.dtype(self.info['dtype'])
        self.total_channel = self.info['total_channel']
        self.sample_rate = self.info['sample_rate']
        
        
        self._open_inital_data()
        self._open_processed_data()
    
    def _open_inital_data(self):
        self.nb_segment = len(self.filenames)
        
        self.initial_data = []
        for filename in self.filenames:
            data = np.memmap(filename, dtype=self.dtype, mode='r').reshape(-1, self.total_channel)
            self.initial_data.append(data)
        
        self.segments_path = []
        for i in range(self.nb_segment):
            segment_path = os.path.join(self.dirname, 'segment_{}'.format(i))
            if not os.path.exists(segment_path):
                os.mkdir(segment_path)
                with open(os.path.join(segment_path, 'info.txt'), 'w', encoding='utf8') as f:
                    f.write('initial filename: '.format(self.filenames[i]))
            self.segments_path.append(segment_path)

        self.arrays_by_seg = []
        for i in range(self.nb_segment):
            arrays = ArrayCollection(parent=None, dirname=self.segments_path[i])
            self.arrays_by_seg.append(arrays)

    
    def _open_processed_data(self):
        for i in range(self.nb_segment):
            for name in ['processed_signals', 'spikes']:
                self.arrays_by_seg[i].load_if_exists(name)
        
    def get_segment_shape(self, seg_num):
        shape = self.initial_data[seg_num].shape
        shape = (shape[0],self.nb_channel,)
        return shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None, signal_type='initial',
                return_type='raw_numpy'):
        
        if signal_type=='initial':
            data = self.initial_data[seg_num][i_start:i_stop, :]
            data = data[:, self.channel_group]
        elif signal_type=='processed':
            if seg_num>=self.nb_segment:
                return None
            else:
                data = self.arrays_by_seg[seg_num].get('processed_signals')[i_start:i_stop, :]
        else:
            raise(ValueError, 'signal_type is not valide')
        
        if return_type=='raw_numpy':
            return data
        elif return_type=='on_scale_numpy':
            raise(NotImplementedError)
        elif return_type=='pandas':
            raise(NotImplementedError)
    
    def reset_processed_signals(self, seg_num=0, dtype='float32'):
        self.arrays_by_seg[seg_num].create_array('processed_signals', dtype, self.get_segment_shape(seg_num), 'memmap')

    
    def set_signals_chunk(self,sigs_chunk, seg_num=0, i_start=None, i_stop=None, signal_type='processed'):
        assert signal_type != 'initial'
        
        if signal_type=='processed':
            data = self.arrays_by_seg[seg_num].get('processed_signals')
            data[i_start:i_stop, :] = sigs_chunk
        
    def flush_processed_signals(self, seg_num=0):
        self.arrays_by_seg[seg_num].flush_array('processed_signals')
        
    def reset_spikes(self, seg_num=0,  dtype=None):
        assert dtype is not None
        self.arrays_by_seg[seg_num].initialize_array('spikes', 'memmap', dtype, (-1,))
        
    def append_spikes(self, seg_num=0, spikes=None):
        if spikes is None: return
        self.arrays_by_seg[seg_num].append_chunk('spikes', spikes)
        
    def flush_spikes(self, seg_num=0):
        self.arrays_by_seg[seg_num].finalize_array('spikes')
    
    def get_spikes(self, seg_num=0, i_start=None, i_stop=None):
        spikes = self.arrays_by_seg[seg_num].get('spikes')
        return spikes[i_start:i_stop]
        
