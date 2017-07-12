import os
import numpy as np
import re
from collections import OrderedDict


possible_modes = ['one-file', 'multi-file', 'one-dir', 'multi-dir', 'other']


class DataSourceBase:
    gui_params = None
    def __init__(self):
        # important total_channel != nb_channel because nb_channel=len(channels)
        self.total_channel = None 
        self.sample_rate = None
        self.nb_segment = None
        self.dtype = None
        self.bit_to_microVolt = None
    
    def get_segment_shape(self, seg_num):
        raise NotImplementedError
    
    def get_channel_names(self):
        raise NotImplementedError
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
        raise NotImplementedError
    
    
class InMemoryDataSource(DataSourceBase):
    """
    DataSource in memory numpy array.
    This is for debugging  or fast testing.
    """
    mode = 'other'
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
            
    def get_channel_names(self):
        return ['ch{}'.format(i) for i in range(self.total_channel)]

    
class RawDataSource(DataSourceBase):
    """
    DataSource from raw binary file. Easy case.
    """
    mode = 'multi-file'
    gui_params = [
        {'name': 'dtype', 'type': 'list', 'values':['int16', 'uint16', 'float32', 'float64']},
        {'name': 'total_channel', 'type': 'int', 'value':1},
        {'name': 'sample_rate', 'type': 'float', 'value':10000., 'step': 1000., 'suffix': 'Hz', 'siPrefix': True},
        {'name': 'offset', 'type': 'int', 'value':0},
    ]
    
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

    def get_channel_names(self):
        return ['ch{}'.format(i) for i in range(self.total_channel)]



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
    
    """
    mode = 'multi-file'
    
    def __init__(self, filename=None, filenames=None):
        DataSourceBase.__init__(self)
        assert HAS_NEO_0_5, 'neo version 0.5.x is not installed'
        # TODO make it multifile
        
        if filename is not None:
            self.filenames = [filename]
        else:
            self.filenames = filenames
            
        print(self.filenames)
        assert len(self.filenames)==1, 'Support only one file should be improved'
        self.filename = self.filenames[0]
        
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
    
    def get_channel_names(self):
        return ['ch{} id{} {}'.format(i, k, self.reader.ids_to_labels[k]) for i, k, in enumerate(self.reader.channel_ids) ]




###

def read_ncs_header(ncs_filename):
    with  open(ncs_filename, 'rb') as f:
        header = f.read(2**14)
    header = header.strip(b'\x00').decode('latin-1')
    
    keys = [('ADBitVolts', 'bit_to_microVolt', float),
                ('AcqEntName', 'channel_name', None),
                ('ADChannel', 'channel_id', int),
                ('SamplingFrequency', 'sample_rate', float),
                ('InputInverted', 'input_inverted', bool),
        ]
    info = {}
    for k1, k2, type_ in keys:
        pattern = '-'+k1+' (\S+)'
        r = re.findall(pattern, header)
        if len(r) == 1:
            info[k2] = r[0]
            if type_ is not None:
                info[k2] = type_(info[k2])
    
    info['bit_to_microVolt'] = info['bit_to_microVolt']*1e6
    return info

ncs_dtype = np.dtype([('timestamp', 'uint64'),
                ('channel', 'uint32'),
                ('sample_rate', 'uint32'),
                ('nb_valid', 'uint32'),
                ('samples', 'int16', (512,))
                ])

def explore_neuralynx_dir(dirname):
    datas = []
    channel_names = []
    invert_input = []
    sample_rate = None
    length = None
    bit_to_microVolt = None
    for filename in os.listdir(dirname):
        if not filename.endswith('.ncs'): continue
        
        fullname = os.path.join(dirname, filename)
        info = read_ncs_header(fullname)
        if sample_rate is None:
            sample_rate = info['sample_rate']
        else:
            assert sample_rate == info['sample_rate'], 'Some channel do not have the same sample rate'
        
        if bit_to_microVolt is None:
            bit_to_microVolt = info['bit_to_microVolt']
        else:
            assert bit_to_microVolt==info['bit_to_microVolt'], 'Some channel do not have the same bit_to_microVolt'
        
        if info['input_inverted']:
            invert_input.append(-1)
        else:
            invert_input.append(-1)
        
        
        data = np.memmap(fullname, dtype=ncs_dtype, mode='r', offset=2**14)
        #~ print(data.shape)
        
        if length is None:
            length = data['samples'].size
        else:
            assert data['samples'].size==length, 'Some channel do not have the same size'
        
        datas.append(data)
        channel_names.append(info['channel_name'])
    
    invert_input = np.array(invert_input, dtype='int16')
    
    return datas, channel_names, sample_rate, invert_input, bit_to_microVolt
    
    

class NeuralynxDataSource(DataSourceBase):
    mode = 'multi-dir'
    def __init__(self, dirnames=[]):
        DataSourceBase.__init__(self)

        self.all_datas = []
        self.channel_names = None
        self.total_channel = None
        self.invert_input = None
        self.bit_to_microVolt = None
        for dirname in dirnames:
            datas, channel_names, sample_rate, invert_input, bit_to_microVolt = explore_neuralynx_dir(dirname)
            
            if self.channel_names is None:
                self.channel_names = channel_names
                self.total_channel = n = len(channel_names)
            else:
                assert all([self.channel_names[i]==channel_names[i] for i in range(n)]), 'channel do not match between segments'
            
            if self.sample_rate is None:
                self.sample_rate = sample_rate
            else:
                assert self.sample_rate == sample_rate, 'Sample rate do not match between segments'
            
            if self.invert_input is None:
                self.invert_input = invert_input
            else:
                assert np.all(invert_input==self.invert_input), 'Invert input do not match between segments'
            
            if self.bit_to_microVolt is None:
                self.bit_to_microVolt = bit_to_microVolt
            else:
                assert bit_to_microVolt==self.bit_to_microVolt,  'bit_to_microVolt do not match between segments'
            
            self.all_datas.append(datas)
        
        self.invert_input = self.invert_input[None, :]
        
        self.nb_segment = len(dirnames)
        self.dtype = np.dtype('int16')
        
        
    
    def get_segment_shape(self, seg_num):
        return self.all_datas[seg_num][0]['samples'].size, self.total_channel
    
    def get_channel_names(self):
        return self.channel_names
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
        if i_start is None: i_start=0
        if i_stop is None: i_stop=self.get_segment_shape(seg_num)
        
        block_start = i_start//512
        block_stop = i_stop//512+1
        sl0 = i_start % 512
        sl1 = sl0 + (i_stop-i_start)
        
        sigs_chunk = np.zeros((i_stop-i_start, self.total_channel, ), dtype=self.dtype)
        sigs = []
        for i, data in enumerate(self.all_datas[seg_num]):
            sub = data[block_start:block_stop]
            sigs_chunk[:, i] = sub['samples'].flatten()[sl0:sl1]
        sigs_chunk *= self.invert_input
        
        return sigs_chunk

    

#TODO implement KWIK and OpenEphys
#https://open-ephys.atlassian.net/wiki/display/OEW/Data+format
# https://github.com/open-ephys/analysis-tools/tree/master/Python3


#OrderedDict    
data_source_classes = {'InMemory':InMemoryDataSource, 'RawData':RawDataSource,
                                        'Blackrock': BlackrockDataSource, 'Neuralynx': NeuralynxDataSource,
                                        }
