from ..peeler import Peeler
import pyacq

import pyacq
from pyacq import Node, register_node_type, ThreadPollInput

from ..peeler import _dtype_spike



class PeelerThread(ThreadPollInput):
    def __init__(self, input_stream, output_streams, peeler,in_group_channels,
                        timeout = 200, parent = None):
        
        ThreadPollInput.__init__(self, input_stream,  timeout=timeout, return_data=True, parent = parent)
        self.output_streams = output_streams
        self.peeler = peeler
        self.in_group_channels = in_group_channels
        
        self.sample_rate = input_stream.params['sample_rate']
        self.total_channel = self.input_stream().params['shape'][1]
        
        #~ self.mutex = Mutex()
    
    def process_data(self, pos, sigs_chunk):
        #TODO maybe remove this
        #~ print(sigs_chunk.shape[0], self.peeler.chunksize)
        assert sigs_chunk.shape[0] == self.peeler.chunksize, 'PeelerThread chunksize is BAD!!'
        
        
        #take only channels concerned
        sigs_chunk = sigs_chunk[:, self.in_group_channels]
        #~ print('pos', pos)
        
        sig_index, preprocessed_chunk, total_spike, spikes  = self.peeler.process_one_chunk(pos, sigs_chunk)
        #~ print('total_spike', total_spike, len(spikes))
        #~ print('sig_index', sig_index, preprocessed_chunk.shape)
        
        self.output_streams['signals'].send(preprocessed_chunk, index=sig_index)
        #~ if spikes is not None and spikes.size>0:
        if spikes.size>0:
            self.output_streams['spikes'].send(spikes, index=total_spike)
        
    
    def change_params(self, kargs):
        pass
        #~ print('PeelerThread.change_params', kargs)
        #~ with self.mutex:
            #~ self.engine.change_params(**kargs)

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
                                    signalpreprocessor_engine='numpy',
                                    peakdetector_engine='numpy',
                                    internal_dtype='float32', n_peel_level=2):
        
        self.in_group_channels = in_group_channels
        self.catalogue = catalogue
        self.chunksize = chunksize
        self.signalpreprocessor_engine = signalpreprocessor_engine
        self.peakdetector_engine = peakdetector_engine
        self.internal_dtype = internal_dtype
        self.n_peel_level = n_peel_level
        
        

    def after_input_connect(self, inputname):
        self.total_channel = self.input.params['shape'][1]
        self.sample_rate = self.input.params['sample_rate']
        
        # internal dtype (for waveforms) will also be the output dtype
        self.outputs['signals'].spec['dtype'] = self.internal_dtype
        self.outputs['signals'].spec['shape'] = (-1, len(self.in_group_channels))
        self.outputs['signals'].spec['sample_rate'] = self.input.params['sample_rate']
        
        
    
    def _initialize(self):
        
        self.peeler = Peeler(dataio=None)
        self.peeler.change_params(catalogue=self.catalogue, n_peel_level=self.n_peel_level,
                                        chunksize=self.chunksize, internal_dtype=self.internal_dtype,
                                        signalpreprocessor_engine=self.signalpreprocessor_engine,
                                        peakdetector_engine=self.peakdetector_engine)
        
        self.thread = PeelerThread(self.input, self.outputs, self.peeler, self.in_group_channels)
        
    def _start(self):
        self.peeler.initialize_online_loop(sample_rate=self.input.params['sample_rate'],
                                            nb_channel=len(self.in_group_channels),
                                            source_dtype=self.input.params['dtype'])
        self.thread.start()
        
    def _stop(self):
        self.thread.stop()
        self.thread.wait()

        
    def _close(self):
        pass

#~ register_node_type(OnlinePeeler)
