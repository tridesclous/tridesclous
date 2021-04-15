from ..peeler import Peeler


from pyqtgraph.util.mutex import Mutex

import pyacq
from pyacq import Node, register_node_type, ThreadPollInput

from ..peeler import _dtype_spike




class PeelerThread(ThreadPollInput):
    def __init__(self, input_stream, output_streams, peeler,in_group_channels, geometry,
                        timeout = 200, parent = None):
        
        ThreadPollInput.__init__(self, input_stream,  timeout=timeout, return_data=True, parent = parent)
        self.output_streams = output_streams
        self.peeler = peeler
        self.in_group_channels = in_group_channels
        self.geometry = geometry
        
        self.sample_rate = input_stream.params['sample_rate']
        self.total_channel = self.input_stream().params['shape'][1]
        
        self.mutex = Mutex()
    
    def process_data(self, pos, sigs_chunk):
        #TODO maybe remove this
        #~ print('process_data', sigs_chunk.shape[0], self.peeler.chunksize)
        assert sigs_chunk.shape[0] == self.peeler.chunksize, 'PeelerThread chunksize is BAD!! {} {}'.format(sigs_chunk.shape[0], self.peeler.chunksize)
        
        with self.mutex:
            #take only channels concerned
            sigs_chunk = sigs_chunk[:, self.in_group_channels]
            #~ print('pos', pos)
            #~ print(sigs_chunk.shape, sigs_chunk.dtype)
            #~ print('signals_medians', self.peeler.signalpreprocessor.signals_medians)
            sig_index, preprocessed_chunk, total_spike, spikes  = self.peeler.process_one_chunk(pos, sigs_chunk)
            #~ print('sig_index', sig_index)
            
            #~ print('total_spike', total_spike, len(spikes))
            #~ print('sig_index', sig_index, preprocessed_chunk.shape)
            
            self.output_streams['signals'].send(preprocessed_chunk, index=sig_index)
            #~ if spikes is not None and spikes.size>0:
            if spikes.size>0:
                self.output_streams['spikes'].send(spikes, index=total_spike)
        
    
    def change_params(self, **peeler_params):
        #~ print('PeelerThread.change_params')
        with self.mutex:
            # complicated hack pad_width can be changed on the fly
            # because it break the index stream with advance or delay
            # so we always forec the prvious one!!!!
            if hasattr(self.peeler, 'peeler_engine') and hasattr(self.peeler.peeler_engine, 'signalpreprocessor'):
                #~ print(self.peeler.peeler_engine.signalpreprocessor.pad_width)
                prev_pad_width = self.peeler.peeler_engine.signalpreprocessor.pad_width
                #~ if 'pad_width' in catalogue['signal_preprocessor_params']:
                peeler_params['catalogue']['signal_preprocessor_params']['pad_width'] = prev_pad_width
                #~ print('*'*50)
                #~ print('Force prev_pad_width', prev_pad_width)
                #~ print('*'*50)
            else:
                prev_pad_width = None

            self.peeler.change_params(**peeler_params)
            
            buffer_spike_index = self.output_streams['spikes'].last_index
            #~ print('buffer_spike_index', buffer_spike_index)
            
            # TODO check tha pad_width have not changed bechause
            # head index will be out
            
            
            
            self.peeler.initialize_online_loop(sample_rate=self.sample_rate,
                                                nb_channel=len(self.in_group_channels),
                                                source_dtype=self.input_stream().params['dtype'],
                                                geometry=self.geometry,
                                                )
            
            #~ print('*'*50)
            #~ print(self.peeler.peeler_engine.signalpreprocessor.pad_width)
            #~ print('*'*50)
            
            #~ print('self.peeler.peeler_engine.total_spike', self.peeler.peeler_engine.total_spike)
            self.peeler.peeler_engine.total_spike = buffer_spike_index
            #~ print('self.peeler.peeler_engine.total_spike', self.peeler.peeler_engine.total_spike)
            
            
            

class OnlinePeeler(Node):
    """
    Wrapper on top of Peeler class to make a pyacq Node.
    And so to have on line spike sorting!!
    """
    _input_specs = {'signals' : dict(streamtype = 'signals')}
    _output_specs = {'signals' : dict(streamtype = 'signals'),
                                'spikes': dict(streamtype='events', shape = (-1, ),  dtype=_dtype_spike),
                                }

    def __init__(self , **kargs):
        Node.__init__(self, **kargs)
    
    def _configure(self, in_group_channels=None, catalogue=None, chunksize=None,
                                    internal_dtype='float32', 
                                    geometry=None,
                                    **peeler_params):
        
        #~ if 'engine' in peeler_engine_kargs:
            #~ peeler_engine = peeler_engine_kargs.pop('engine')
        #~ print('peeler_engine', peeler_engine)
        
        assert geometry is not None
        
        self.in_group_channels = in_group_channels
        self.catalogue = catalogue
        self.chunksize = chunksize
        self.internal_dtype = internal_dtype
        
        self.geometry = geometry
        self.peeler_params = peeler_params
        
        

    def after_input_connect(self, inputname):
        self.total_channel = self.input.params['shape'][1]
        self.sample_rate = self.input.params['sample_rate']
        
        # internal dtype (for waveforms) will also be the output dtype
        self.outputs['signals'].spec['dtype'] = self.internal_dtype
        self.outputs['signals'].spec['shape'] = (-1, len(self.in_group_channels))
        self.outputs['signals'].spec['sample_rate'] = self.input.params['sample_rate']
    
    def after_output_configure(self, inputname):
        channel_info = self.input.params.get('channel_info', None)
        if channel_info is not None:
            channel_info = [channel_info[c] for c in self.in_group_channels]
            self.outputs['signals'].params['channel_info'] = channel_info
    
    def _initialize(self):
        
        self.peeler = Peeler(dataio=None)
        #~ self.peeler.change_params(catalogue=self.catalogue, 
                                        #~ chunksize=self.chunksize, internal_dtype=self.internal_dtype,)
        
        self.thread = PeelerThread(self.input, self.outputs, self.peeler, self.in_group_channels, self.geometry)
        self.change_catalogue(self.catalogue)
        
    def _start(self):
        #~ self.peeler.initialize_online_loop(sample_rate=self.input.params['sample_rate'],
                                            #~ nb_channel=len(self.in_group_channels),
                                            #~ source_dtype=self.input.params['dtype'])
        self.thread.start()
        
    def _stop(self):
        self.thread.stop()
        self.thread.wait()

        
    def _close(self):
        pass
    
    def change_catalogue(self, catalogue):
        print('change_catalogue', )
        #~ print(catalogue['label_to_index'])
        print('self.peeler_params', self.peeler_params)
        self.catalogue = catalogue
        #~ self.thre.change_params(catalogue=self.catalogue, 
                                        #~ chunksize=self.chunksize, internal_dtype=self.internal_dtype,)
        self.thread.change_params(catalogue=catalogue, 
                                        chunksize=self.chunksize, internal_dtype=self.internal_dtype,
                                        **self.peeler_params)
                                        
                                        
        
    
    
#~ register_node_type(OnlinePeeler)
