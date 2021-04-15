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

#~ from .peeler_engine_testing import PeelerEngineTesting
from .peeler_engine_geometry import PeelerEngineGeometrical
from .peeler_engine_geometry_cl import PeelerEngineGeometricalCl




peeler_engines = {
    #~ 'testing' : PeelerEngineTesting,
    'geometrical' : PeelerEngineGeometrical,
    'geometrical_opencl' : PeelerEngineGeometricalCl,
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
    
    Note that the global latency depend on this paramters:
      * pad_width
      * chunksize

    
    """
    def __init__(self, dataio):
        #for online dataio is None
        self.dataio = dataio

    def __repr__(self):
        t = "Peeler <id: {}> \n  workdir: {}\n".format(id(self), self.dataio.dirname)
        
        return t

    def change_params(self, catalogue=None, engine='geometrical', internal_dtype='float32',
                        chunksize=1024, speed_test_mode=False, **params):
        """
        
        speed_test_mode: only for offline mode create a log file with 
                run time for each buffers
        """
        assert catalogue is not None
        
        self.catalogue = catalogue
        self.engine_name = engine
        self.internal_dtype = internal_dtype
        self.chunksize = chunksize
        self.speed_test_mode = speed_test_mode
        
        self.peeler_engine = peeler_engines[engine]()
        self.peeler_engine.change_params(catalogue=catalogue, internal_dtype=internal_dtype, chunksize=chunksize, **params)
    
    def process_one_chunk(self,  pos, sigs_chunk):
        # this is for online
        return self.peeler_engine.process_one_chunk(pos, sigs_chunk)
        
        #~ abs_head_index, preprocessed_chunk, self.total_spike, all_spikes,  = self.peeler_engine.process_one_chunk(pos, sigs_chunk)
        #~ print(pos, sigs_chunk.shape, abs_head_index, preprocessed_chunk.shape)
        #~ return abs_head_index, preprocessed_chunk, self.total_spike, all_spikes
    
    def initialize_online_loop(self, sample_rate=None, nb_channel=None, source_dtype=None, geometry=None):
        # global initialize
        self.peeler_engine.initialize(sample_rate=sample_rate, nb_channel=nb_channel,
                        source_dtype=source_dtype, already_processed=False, geometry=geometry)
        self.peeler_engine.initialize_before_each_segment(already_processed=False)
    
    def run_offline_loop_one_segment(self, seg_num=0, duration=None, progressbar=True):
        chan_grp = self.catalogue['chan_grp']

        if duration is not None:
            length = int(duration*self.dataio.sample_rate)
        else:
            length = self.dataio.get_segment_length(seg_num)
        
        # check if the desired length is already computed or not for this particular segment
        already_processed = self.dataio.already_processed(seg_num=seg_num, chan_grp=chan_grp, length=length)
        
        self.peeler_engine.initialize_before_each_segment(already_processed=already_processed)
        #~ print('run_offline_loop_one_segment already_processed', already_processed)
        
        if already_processed:
            # ready from "processed'
            signal_type = 'processed'
        else:
            # read from "initial" 
            # activate signal processor
            signal_type = 'initial'
        
            #initialize engines
            self.dataio.reset_processed_signals(seg_num=seg_num, chan_grp=chan_grp, dtype=self.internal_dtype)
        
        self.dataio.reset_spikes(seg_num=seg_num, chan_grp=chan_grp, dtype=_dtype_spike)

        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chan_grp=chan_grp, chunksize=self.chunksize, 
                                                    i_stop=length, signal_type=signal_type)
        if progressbar:
            iterator = tqdm(iterable=iterator, total=length//self.chunksize)
        
        if self.speed_test_mode:
            process_run_times = []
        
        for pos, sigs_chunk in iterator:
            if self.speed_test_mode:
                t0 = time.perf_counter()
            
            sig_index, preprocessed_chunk, total_spike, spikes = self.peeler_engine.process_one_chunk(pos, sigs_chunk)
            
            if self.speed_test_mode:
                t1 = time.perf_counter()
                process_run_times.append(t1-t0)
                

            
            
            if sig_index<=0:
                continue
            
            if not already_processed:
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
        
        if not already_processed:
            self.dataio.flush_processed_signals(seg_num=seg_num, chan_grp=chan_grp, processed_length=int(sig_index))
            
        self.dataio.flush_spikes(seg_num=seg_num, chan_grp=chan_grp)
        
        if self.speed_test_mode:
            process_run_times = np.array(process_run_times, dtype='float64')
            log_path = self.dataio.get_log_path(chan_grp=chan_grp)
            filename = os.path.join(log_path, 'peeler_run_times_seg{}.npy'.format(seg_num))
            np.save(filename, process_run_times)

    def run(self, duration=None, progressbar=True):
        assert hasattr(self, 'catalogue'), 'So peeler.change_params first'
        
        chan_grp = self.catalogue['chan_grp']
        
        duration_per_segment = self.dataio.get_duration_per_segments(duration)
        
        already_processed_segs = []
        for seg_num in range(self.dataio.nb_segment):
            length = int(duration_per_segment[seg_num]*self.dataio.sample_rate)
            
            # check if the desired length is already computed or not
            already_processed = self.dataio.already_processed(seg_num=seg_num, chan_grp=chan_grp, length=length)
            already_processed_segs.append(already_processed)
        
        kargs = {}
        kargs['sample_rate'] = self.dataio.sample_rate
        kargs['nb_channel'] = self.dataio.nb_channel(chan_grp)
        if any(already_processed_segs):
            kargs['source_dtype'] = self.internal_dtype
        else:
            kargs['source_dtype'] = self.dataio.source_dtype
        kargs['geometry'] = self.dataio.get_geometry(chan_grp)
        kargs['already_processed'] =  all(already_processed_segs)
        self.peeler_engine.initialize(**kargs)
        
        for seg_num in range(self.dataio.nb_segment):
            self.run_offline_loop_one_segment(seg_num=seg_num, duration=duration_per_segment[seg_num], progressbar=progressbar)
    
    # old alias just in case
    run_offline_all_segment = run
    
    
    def get_run_times(self, chan_grp=0, seg_num=0):
        """
        need speed_test_mode=True in params
        """
        p = self.dataio.get_log_path(chan_grp=chan_grp)
        filename = os.path.join(p, 'peeler_run_times_seg{}.npy'.format(seg_num))
        run_times = np.load(filename)
        return run_times
    
