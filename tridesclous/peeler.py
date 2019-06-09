"""

.. autoclass:: Peeler
   :members:

"""

import os
import json
from collections import OrderedDict, namedtuple
import time

import numpy as np
import scipy.signal


from .peeler_tools import _dtype_spike


from tqdm import tqdm

from .peeler_engine_classic import PeelerEngineClassic
from .peeler_engine_classic_cl import PeelerEngineClassicOpenCl
#~ from .peeler_engine_strict import PeelerEngineStrict

peeler_engines = {
    'classic' : PeelerEngineClassic,
    'classic_opencl' : PeelerEngineClassicOpenCl,
    #~ 'strict' : PeelerEngineStrict,
}



class Peeler:
    """
    The peeler is core of spike sorting itself.
    It basically do a *template matching* on a signals.
    
    This class nedd a *catalogue* constructed by :class:`CatalogueConstructor`.
    Then the compting is applied chunk chunk on the raw signal itself.
    
    So this class is the same for both offline/online computing.
    
    At each chunk, the algo is basically this one:
      1. apply the processing chain (filter, normamlize, ....)
      2. Detect peaks
      3. Try to classify peak and detect the *jitter*
      4. With labeled peak create a prediction for the chunk
      5. Substract the prediction from the processed signals.
      6. Go back to **2** until there is no peak or only peaks that can't be labeled.
      7. return labeld spikes from this or previous chunk and the processed signals (for display or recoding)
    
    The main difficulty in the implemtation is to deal with edge because spikes 
    waveforms can spread out in between 2 chunk.
    
    Note that the global latency depend on this é paramters:
      * lostfront_chunksize
      * chunksize

    
    """
    def __init__(self, dataio):
        #for online dataio is None
        self.dataio = dataio

    def __repr__(self):
        t = "Peeler <id: {}> \n  workdir: {}\n".format(id(self), self.dataio.dirname)
        
        return t

    def change_params(self, catalogue=None, engine='classic', internal_dtype='float32', chunksize=1024, **params):
        assert catalogue is not None
        
        self.catalogue = catalogue
        self.internal_dtype = internal_dtype
        self.chunksize = chunksize
        self.engine_name = engine
        self.peeler_engine = peeler_engines[engine]()
        self.peeler_engine.change_params(catalogue=catalogue, internal_dtype=internal_dtype, chunksize=chunksize, **params)
    
    def process_one_chunk(self,  pos, sigs_chunk):
        return self.peeler_engine.process_one_chunk(pos, sigs_chunk)
    
    def initialize_online_loop(self, sample_rate=None, nb_channel=None, source_dtype=None):
        self.peeler_engine.initialize_before_each_segment(sample_rate=sample_rate, nb_channel=nb_channel, source_dtype=source_dtype)
    
    def run_offline_loop_one_segment(self, seg_num=0, duration=None, progressbar=True):
        chan_grp = self.catalogue['chan_grp']
        
        kargs = {}
        kargs['sample_rate'] = self.dataio.sample_rate
        kargs['nb_channel'] = self.dataio.nb_channel(chan_grp)
        kargs['source_dtype'] = self.dataio.source_dtype
        self.peeler_engine.initialize_before_each_segment(**kargs)
        
        
        if duration is not None:
            length = int(duration*self.dataio.sample_rate)
        else:
            length = self.dataio.get_segment_length(seg_num)
        #~ length -= length%self.chunksize
        
        #initialize engines
        self.dataio.reset_processed_signals(seg_num=seg_num, chan_grp=chan_grp, dtype=self.internal_dtype)
        self.dataio.reset_spikes(seg_num=seg_num, chan_grp=chan_grp, dtype=_dtype_spike)

        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=chan_grp, chunksize=self.chunksize, 
                                                    i_stop=length, signal_type='initial')
        if progressbar:
            iterator = tqdm(iterable=iterator, total=length//self.chunksize)
        for pos, sigs_chunk in iterator:
            
            #~ sig_index, preprocessed_chunk, total_spike, spikes = self.process_one_chunk(pos, sigs_chunk)
            sig_index, preprocessed_chunk, total_spike, spikes = self.peeler_engine.process_one_chunk(pos, sigs_chunk)
            
            
            if sig_index<=0:
                continue
            
            # save preprocessed_chunk to file
            self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num,chan_grp=chan_grp,
                        i_start=sig_index-preprocessed_chunk.shape[0], i_stop=sig_index,
                        signal_type='processed')
            
            if spikes is not None and spikes.size>0:
                self.dataio.append_spikes(seg_num=seg_num, chan_grp=chan_grp, spikes=spikes)
        
        extra_spikes = self.peeler_engine.get_remaining_spikes()
        if extra_spikes is not None:
            
            if extra_spikes.size>0:
                self.dataio.append_spikes(seg_num=seg_num, chan_grp=chan_grp, spikes=extra_spikes)
        
        self.dataio.flush_processed_signals(seg_num=seg_num, chan_grp=chan_grp)
        self.dataio.flush_spikes(seg_num=seg_num, chan_grp=chan_grp)

    def run_offline_all_segment(self, **kargs):
        assert hasattr(self, 'catalogue'), 'So peeler.change_params first'
        
        for seg_num in range(self.dataio.nb_segment):
            self.run_offline_loop_one_segment(seg_num=seg_num, **kargs)
    
    run = run_offline_all_segment
        


    
