import os
import numpy as np

from collections import OrderedDict





class DataSourceBase:
    def __init__(self):
        # important total_channel != nb_channel because nb_channel=len(channels)
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
                        sample_rate=0., offset=0):
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
            data = np.memmap(filename, dtype=self.dtype, mode='r', offset=offset).reshape(-1, self.total_channel)
            self.array_sources.append(data)
    
    def get_segment_shape(self, seg_num):
        full_shape = self.array_sources[seg_num].shape
        return full_shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
            data = self.array_sources[seg_num][i_start:i_stop, :]
            return data


try:
    import neo
    import distutils.version
    assert distutils.version.LooseVersion(neo.__version__)>='0.5'
    HAS_NEO_0_5 = True
except:
    HAS_NEO_0_5 = False




class MyBlackrockIO(neo.BlackrockIO):
    def read_analogsignals(self):
        nsx_nb = 5
        spec = self._BlackrockIO__nsx_spec[nsx_nb]
        nsx_data = self._BlackrockIO__nsx_data_reader[spec](nsx_nb)
        sampling_rate = self._BlackrockIO__nsx_params[spec]('sampling_rate', nsx_nb)        
        return nsx_data, sampling_rate.rescale('Hz').magnitude
    
    @property
    def ids_to_labels(self):
        return self._BlackrockIO__nev_params('channel_labels')
    
    @property        
    def channel_ids(self):
        return self._BlackrockIO__nsx_ext_header[5]['electrode_id']
    
    

class BlackrockDataSource(DataSourceBase):
    """
    DataSource for blackrock files ns5.
    based on neo v0.5
    Not very efficient for the moment because everything if in float.
    
    """
    def __init__(self, filename=[]):
        DataSourceBase.__init__(self)
        assert HAS_NEO_0_5, 'neo version 0.5.x is not installed'
        
        self.filename = filename

        assert os.path.exists(self.filename), 'files does not exist'
        
        
        for ext in ['.nev', '.ns5']:
            if self.filename.endswith(ext):
                self.filename = self.filename.strip(ext)
        self.reader = MyBlackrockIO(filename=self.filename, verbose=False)

        self.channel_ids = self.reader.channel_ids
        self.labels_to_ids = {self.reader.ids_to_labels[k]:k for k in self.reader.channel_ids}
        
        self.nsx_data, self.sample_rate = self.reader.read_analogsignals()
        self.all_segment_idx = np.unique(list(self.nsx_data.keys()))
        self.nb_segment = len(self.all_segment_idx)
        
        seg_id = self.all_segment_idx[0]
        data0 = self.nsx_data[seg_id]
        
        self.total_channel = data0.shape[1]
        self.dtype = data0.dtype

    
    def get_segment_shape(self, seg_num):
        seg_id = self.all_segment_idx[seg_num]
        data = self.nsx_data[seg_id]
        return data.shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
            seg_id = self.all_segment_idx[seg_num]
            data = self.nsx_data[seg_id]
            return data[i_start:i_stop, :]
    
    def label_to_nums(self, channel_labels):
        # channel_labels > channel_id > channel_num
        chan_ids = self.channel_ids.tolist()
        channel_ids = [self.labels_to_ids[l] for l in channel_labels]
        channel_nums = [ chan_ids.index(chan_id) for chan_id in channel_ids]
        return channel_nums
        


#TODO implement KWIK and OpenEphys
#https://open-ephys.atlassian.net/wiki/display/OEW/Data+format
# https://github.com/open-ephys/analysis-tools/tree/master/Python3


#OrderedDict    
data_source_classes = {'InMemory':InMemoryDataSource, 'RawData':RawDataSource,
                                        'Blackrock': BlackrockDataSource,
                                        }
