import os
import numpy as np
import re
from collections import OrderedDict


data_source_classes = OrderedDict()

possible_modes = ['one-file', 'multi-file', 'one-dir', 'multi-dir', 'other']


import neo



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

data_source_classes['InMemory'] = InMemoryDataSource





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
        {'name': 'bit_to_microVolt', 'type': 'float', 'value':0.5 },        
    ]
    
    def __init__(self, filenames=[], dtype='int16', total_channel=0,
                        sample_rate=0., offset=0, bit_to_microVolt=None, channel_names=None):
        DataSourceBase.__init__(self)
        
        self.filenames = filenames
        if isinstance(self.filenames, str):
            self.filenames = [self.filenames]
        assert all([os.path.exists(f) for f in self.filenames]), 'files does not exist'
        self.nb_segment = len(self.filenames)

        self.total_channel = total_channel
        self.sample_rate = sample_rate
        self.dtype = np.dtype(dtype)
        
        if bit_to_microVolt == 0.:
            bit_to_microVolt = None
        self.bit_to_microVolt = bit_to_microVolt
        
        if channel_names is None:
            channel_names = ['ch{}'.format(i) for i in range(self.total_channel)]
        self.channel_names = channel_names
        

        self.array_sources = []
        for filename in self.filenames:
            data = np.memmap(filename, dtype=self.dtype, mode='r', offset=offset)
            #~ data = data[:-(data.size%self.total_channel)]
            data = data.reshape(-1, self.total_channel)
            self.array_sources.append(data)
    
    def get_segment_shape(self, seg_num):
        full_shape = self.array_sources[seg_num].shape
        return full_shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
            data = self.array_sources[seg_num][i_start:i_stop, :]
            return data

    def get_channel_names(self):
        return self.channel_names

data_source_classes['RawData'] = RawDataSource






import neo.rawio

io_gui_params = {
    'RawBinarySignal':[
                {'name': 'dtype', 'type': 'list', 'values':['int16', 'uint16', 'float32', 'float64']},
                {'name': 'nb_channel', 'type': 'int', 'value':1},
                {'name': 'sampling_rate', 'type': 'float', 'value':10000., 'step': 1000., 'suffix': 'Hz', 'siPrefix': True},
                {'name': 'bytesoffset', 'type': 'int', 'value':0},
    ],
}


# hook for some neo.rawio that have problem with TDC (multi sampling rate or default params)
neo_rawio_hooks = {}

class Intan(neo.rawio.IntanRawIO):
    def _parse_header(self):
        neo.rawio.IntanRawIO._parse_header(self)
        sig_channels = self.header['signal_channels']
        sig_channels = sig_channels[sig_channels['group_id']==0]
        self.header['signal_channels'] = sig_channels

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        assert np.unique(self.header['signal_channels'][channel_indexes]['group_id']).size == 1
        channel_names = self.header['signal_channels'][channel_indexes]['name']
        chan_name = channel_names[0]
        size = self._raw_data[chan_name].size
        return size

neo_rawio_hooks['Intan'] = Intan



class NeoRawIOAggregator(DataSourceBase):
    """
    wrappe and agregate several neo.rawio in the class.
    """
    gui_params = None
    rawio_class = None
    def __init__(self, **kargs):
        DataSourceBase.__init__(self)
        
        self.rawios = []
        if  'filenames' in kargs:
            filenames= kargs.pop('filenames') 
            self.rawios = [self.rawio_class(filename=f, **kargs) for f in filenames]
        elif 'dirnames' in kargs:
            dirnames= kargs.pop('dirnames') 
            self.rawios = [self.rawio_class(dirname=d, **kargs) for d in dirnames]
        else:
            raise(ValueError('Must have filenames or dirnames'))
            
        
        self.sample_rate = None
        self.total_channel = None
        self.sig_channels = None
        nb_seg = 0
        self.segments = {}
        for rawio in self.rawios:
            rawio.parse_header()
            assert not rawio._several_channel_groups, 'several sample rate for signals'
            assert rawio.block_count() ==1, 'Multi block RawIO not implemented'
            for s in range(rawio.segment_count(0)):
                #nb_seg = absolut seg index and s= local seg index
                self.segments[nb_seg] = (rawio, s)
                nb_seg += 1
            
            if self.sample_rate is None:
                self.sample_rate = rawio.get_signal_sampling_rate()
            else:
                assert self.sample_rate == rawio.get_signal_sampling_rate(), 'bad joke different sample rate!!'
            
            sig_channels = rawio.header['signal_channels']
            if self.sig_channels is None:
                self.sig_channels = sig_channels
                self.total_channel = len(sig_channels)
            else:
                assert np.all(sig_channels==self.sig_channels), 'bad joke different channels!'
            
        self.nb_segment = len(self.segments)
        
        self.dtype = np.dtype(self.sig_channels['dtype'][0])
        units = sig_channels['units'][0]
        #~ assert 'V' in units, 'Units are not V, mV or uV'
        if units =='V':
            self.bit_to_microVolt = self.sig_channels['gain'][0]*1e-6
        elif units =='mV':
            self.bit_to_microVolt = self.sig_channels['gain'][0]*1e-3
        elif units =='uV':
            self.bit_to_microVolt = self.sig_channels['gain'][0]
        else:
            self.bit_to_microVolt = None
        
    def get_segment_shape(self, seg_num):
        rawio, s = self.segments[seg_num]
        l = rawio.get_signal_size(0, s)
        return l, self.total_channel
    
    def get_channel_names(self):
        return self.sig_channels['name'].tolist()
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
        rawio, s = self.segments[seg_num]
        return rawio.get_analogsignal_chunk(block_index=0, seg_index=s, 
                        i_start=i_start, i_stop=i_stop)

#Construct the list with taking local class with hooks dict
rawiolist = []
for rawio_class in neo.rawio.rawiolist:
    name = rawio_class.__name__.replace('RawIO', '')
    if name in neo_rawio_hooks:
        rawio_class = neo_rawio_hooks[name]
    rawiolist.append(rawio_class)

if neo.rawio.RawBinarySignalRawIO in rawiolist:
    # to avoid bug in readthe doc with moc
    RawBinarySignalRawIO = rawiolist.pop(rawiolist.index(neo.rawio.RawBinarySignalRawIO))
#~ rawiolist.insert(0, RawBinarySignalRawIO)

for rawio_class in rawiolist:
    name = rawio_class.__name__.replace('RawIO', '')
    class_name = name+'DataSource'
    datasource_class = type(class_name,(NeoRawIOAggregator,), { })
    datasource_class.rawio_class = rawio_class
    if rawio_class.rawmode in ('multi-file', 'one-file'):
        #multi file in neo have another meaning
        datasource_class.mode = 'multi-file'
    elif rawio_class.rawmode in ('one-dir', ):
        datasource_class.mode = 'multi-dir'
    else:
        continue
    
    #gui stuffs
    if name in io_gui_params:
        datasource_class.gui_params = io_gui_params[name]
        
    data_source_classes[name] = datasource_class
    #~ print(datasource_class, datasource_class.mode )


    
#TODO implement KWIK and OpenEphys
#https://open-ephys.atlassian.net/wiki/display/OEW/Data+format
# https://github.com/open-ephys/analysis-tools/tree/master/Python3


