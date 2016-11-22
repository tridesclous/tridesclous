import os
import json
from collections import OrderedDict
import numpy as np
import pandas as pd

from .iotools import ArrayCollection

_signal_types = ['initial', 'processed']


class DataIO:
    """
    
    """
    def __init__(self, dirname='test'):
        self.dirname = dirname
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        self.info_filename = os.path.join(self.dirname, 'info.json')
        if not os.path.exists(self.info_filename):
            #first init
            self.info = {}
            self.flush_info()
            self.datasource = None
        else:
            with open(self.info_filename, 'r', encoding='utf8') as f:
                self.info = json.load(f)
            try:
                self.reload_info()
                self._reload_data_source()
                self._open_processed_data()
            except:
                self.info = {}
                self.flush_info()
                self.datasource = None
 
    def __repr__(self):
        t = "DataIO <id: {}> \n  workdir: {}\n".format(id(self), self.dirname)
        if len(self.info) ==0 or self.datasource is None:
            t  += "\n  Not datasource is set yet"
            return t
        
        t += "  sample_rate: {}\n".format(self.sample_rate)
        t += "  nb_segment: {}\n".format(self.nb_segment)
        t += "  total_channel: {}\n".format(self.total_channel)
        t += "  nb_channel: {}\n".format(self.nb_channel)
        if self.nb_channel<12:
            t += "  channel_group: {}\n".format(self.channel_group)
        else:
            t += "  channel_group: [{} ... {}]\n".format(' '.join(str(e) for e in self.channel_group[:4]),
                                                                                        ' '.join(str(e) for e in self.channel_group[-4:]))
        if self.nb_segment<5:
            lengths = [ self.get_segment_shape(i)[0] for i in range(self.nb_segment)]
            t += '  length: '+' '.join('{}'.format(l) for l in lengths)+'\n'
            t += '  durations: '+' '.join('{:0.1f}'.format(l/self.sample_rate) for l in lengths)+' s.\n'
        
        return t

 
    def flush_info(self):
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f, indent=4)
    
    def reload_info(self):
        if 'channel_group' in self.info:
            self.channel_group = self.info['channel_group']
        
        self.nb_channel  = len(self.channel_group)
    
    def set_data_source(self, type='RawData', **kargs):
        assert type in data_source_classes, 'this source type do not exists yet!!'
        assert 'datasource_type' not in self.info, 'datasource is already set'
        self.info['datasource_type'] = type
        self.info['datasource_kargs'] = kargs
        self._reload_data_source()
        # be default chennel group all channels
        self.set_channel_group(np.arange(self.total_channel))
        self.flush_info()
        # this create segment path
        self._open_processed_data()
    
    def _reload_data_source(self):
        assert 'datasource_type' in self.info
        kargs = self.info['datasource_kargs']
        
        self.datasource = data_source_classes[self.info['datasource_type']](**kargs)
        self.total_channel = self.datasource.total_channel
        self.nb_segment = self.datasource.nb_segment
        self.sample_rate = self.datasource.sample_rate
        self.source_dtype = self.datasource.dtype
    
    def set_channel_group(self, channel_group=[]):
        if isinstance(channel_group, np.ndarray):
            channel_group =channel_group.tolist()
        self.channel_group = list(channel_group)
        self.info['channel_group'] = self.channel_group
        self.nb_channel  = len(self.channel_group)
        self.info['nb_channel'] = self.nb_channel
        self.flush_info()

    def _open_processed_data(self):
        self.segments_path = []
        for i in range(self.nb_segment):
            segment_path = os.path.join(self.dirname, 'segment_{}'.format(i))
            if not os.path.exists(segment_path):
                os.mkdir(segment_path)
                #~ with open(os.path.join(segment_path, 'info.txt'), 'w', encoding='utf8') as f:
                    #~ f.write('initial filename: '.format(self.filenames[i]))
            self.segments_path.append(segment_path)
        self.arrays_by_seg = []
        for i in range(self.nb_segment):
            arrays = ArrayCollection(parent=None, dirname=self.segments_path[i])
            self.arrays_by_seg.append(arrays)
            
            for name in ['processed_signals', 'spikes']:
                self.arrays_by_seg[i].load_if_exists(name)
        
    def get_segment_shape(self, seg_num):
        full_shape =  self.datasource.get_segment_shape(seg_num)
        #~ shape = self.array_sources[seg_num].shape
        shape = (full_shape[0],self.nb_channel,)
        return shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None, signal_type='initial',
                return_type='raw_numpy'):
        
        if signal_type=='initial':
            data = self.datasource.get_signals_chunk(seg_num=seg_num, i_start=i_start, i_stop=i_stop)
            data = data[:, self.channel_group]
        elif signal_type=='processed':
            data = self.arrays_by_seg[seg_num].get('processed_signals')[i_start:i_stop, :]
        else:
            raise(ValueError, 'signal_type is not valide')
        
        if return_type=='raw_numpy':
            return data
        elif return_type=='on_scale_numpy':
            raise(NotImplementedError)
        elif return_type=='pandas':
            raise(NotImplementedError)

    def iter_over_chunk(self, seg_num=0, i_stop=None, chunksize=1024, **kargs):
        
        if i_stop is not None:
            length = min(self.get_segment_shape(seg_num)[0], i_stop)
        else:
            
            length = self.get_segment_shape(seg_num)[0]
        
        #TODO for last chunk append some zeros: maybe: ????
        nloop = length//chunksize
        for i in range(nloop):
            i_stop = (i+1)*chunksize
            i_start = i_stop - chunksize
            sigs_chunk = self.get_signals_chunk(seg_num=seg_num, i_start=i_start, i_stop=i_stop, **kargs)
            yield  i_stop, sigs_chunk
    
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





class DataSourceBase:
    def __init__(self):
        # important total_channel != nb_channel becausenb_channel=len(channel_group)
        self.total_channel = None 
        self.sample_rate = None
        self.nb_segment = None
    
    def get_segment_shape(self, seg_num):
        raise NotImplementedError
    
    
class InMemoryDataSource(DataSourceBase):
    """
    DataSource in memory numpy array.
    This is for debugging  or fast testing.
    """
    def __init__(self, nparrays=[], sample_rate=None):
        DataSourceBase.__init__(self)
        
        self.nparrays = nparrays
        self.nb_segment = len(self.nparrays)
        self.total_channel = self.nparrays[0].shape[1]
        self.sample_rate = sample_rate
        self.dtype = self.nparrays[0].dtype
    
    def get_segment_shape(self, seg_num):
        full_shape = self.nparrays[seg_num].shape
        return full_shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
            data = self.nparrays[seg_num][i_start:i_stop, :]
            return data        

    
class RawDataSource(DataSourceBase):
    """
    DataSource from raw binary file. Easy case.
    """
    def __init__(self, filenames=[], dtype='int16', total_channel=0,
                        sample_rate=0.):
        DataSourceBase.__init__(self)
        
        self.filenames = filenames
        if isinstance(self.filenames, str):
            self.filenames = [self.filenames]
        assert all([os.path.exists(f) for f in self.filenames]), 'files does not exist'
        self.nb_segment = len(self.filenames)

        self.total_channel = total_channel
        self.sample_rate = sample_rate
        self.dtype = np.dtype(dtype)

        self.array_sources = []
        for filename in self.filenames:
            data = np.memmap(filename, dtype=self.dtype, mode='r').reshape(-1, self.total_channel)
            self.array_sources.append(data)
    
    def get_segment_shape(self, seg_num):
        full_shape = self.array_sources[seg_num].shape
        return full_shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
            data = self.array_sources[seg_num][i_start:i_stop, :]
            return data


    
data_source_classes = {'InMemory':InMemoryDataSource, 'RawData':RawDataSource}

