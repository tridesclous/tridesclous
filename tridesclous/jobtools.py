"""
Some help function to compute in parallel processing at some stage:

  * CatalogueConstructor.run_signalprocessor = preprocessing + peak detection

Used only for offline computation.
This is usefull mainlly when the IO are slow.


"""
import time
import os
import loky

#~ import concurrent.futures.ThreadPoolExecutor

import numpy as np

from .dataio import DataIO
from .signalpreprocessor import signalpreprocessor_engines
from .peakdetector import get_peak_detector_class
from . import labelcodes


# TODO VERY IMPORTANT MOVE THIS
_dtype_peak = [('index', 'int64'), ('cluster_label', 'int64'), ('channel', 'int64'),  ('segment', 'int64'), ('extremum_amplitude', 'float64'),]



def signalprocessor_initializer(dirname, chan_grp_, seg_num_,
                internal_dtype, chunksize, pad_width_,
                signals_medians, signals_mads,
                signal_preprocessor_params,
                peak_detector_params):
    global dataio
    global chan_grp
    global seg_num
    global pad_width
    global signalpreprocessor
    global peakdetector
    
    dataio = DataIO(dirname)
    #~ print('signalprocessor_initializer', id(dataio))
    
    chan_grp = chan_grp_
    
    seg_num = seg_num_
    
    pad_width = pad_width_
    
    
    p = dict(signal_preprocessor_params)
    engine = p.pop('engine')
    SignalPreprocessor_class = signalpreprocessor_engines[engine]
    signalpreprocessor = SignalPreprocessor_class(dataio.sample_rate, dataio.nb_channel(chan_grp), chunksize, dataio.source_dtype)
    p['normalize'] = True
    p['signals_medians'] = signals_medians
    p['signals_mads'] = signals_mads
    signalpreprocessor.change_params(**p)

    
    p = dict(peak_detector_params)
    engine = p.pop('engine')
    method = p.pop('method')
    PeakDetector_class = get_peak_detector_class(method, engine)
    geometry = dataio.get_geometry(chan_grp)
    peakdetector = PeakDetector_class(dataio.sample_rate, dataio.nb_channel(chan_grp),
                                                    chunksize, internal_dtype, geometry)
    peakdetector.change_params(**p)


def read_process_write_one_chunk(args):
    i_start, i_stop = args
    # this read process and write one chunk
    global dataio
    global chan_grp
    global seg_num
    global pad_width
    global signalpreprocessor
    global peakdetector
    
    #~ print(i_start, i_stop, id(dataio))
    #~ print(dataio)
    
    # read chunk and pad
    sigs_chunk = dataio.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp,
                i_start=i_start, i_stop=i_stop,  signal_type='initial', pad_width=pad_width)
    #~ print('pad_width', pad_width)
    #~ print('read_process_write_one_chunk', i_start, i_stop, i_stop-i_start, sigs_chunk.shape)

    # process
    preprocessed_chunk = signalpreprocessor.process_buffer(sigs_chunk)
    #~ exit()
    
    
    
    # peak detection
    n_span = peakdetector.n_span
    assert n_span < pad_width
    chunk_peak = preprocessed_chunk[pad_width-n_span:-pad_width+n_span]
    time_ind_peaks, chan_ind_peaks, peak_val_peaks = peakdetector.process_buffer(chunk_peak)
    
    peaks = np.zeros(time_ind_peaks.size, dtype=_dtype_peak)
    peaks['index'] = time_ind_peaks -  n_span+ i_start
    peaks['segment'][:] = seg_num
    peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
    if chan_ind_peaks is None:
        peaks['channel'][:] = -1
    else:
        peaks['channel'][:] = chan_ind_peaks
    if peak_val_peaks is None:
        peaks['extremum_amplitude'][:] = 0.
    else:
        peaks['extremum_amplitude'][:] = peak_val_peaks
    
    
    # remove the padding and write
    preprocessed_chunk = preprocessed_chunk[pad_width:-pad_width]
    dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num, chan_grp=chan_grp,
                    i_start=i_start, i_stop=i_stop, signal_type='processed')
    
    
    return peaks
    

def run_parallel_read_process_write(cc, seg_num, length, n_jobs):
    
    chunksize = cc.info['chunksize']
    pad_width = cc.info['preprocessor']['pad_width']
    
    initargs=(cc.dataio.dirname, cc.chan_grp, seg_num, 
                    cc.internal_dtype, chunksize, pad_width,
                    cc.signals_medians, cc.signals_mads,
                    cc.info['preprocessor'],
                    cc.info['peak_detector'],
                    )
    
    
    
    if length is None:
        length = cc.dataio.get_segment_length(seg_num)
    num_chunk = length // chunksize
    chunk_slice = [(i*chunksize, (i+1)*chunksize) for i in range(num_chunk)]
    if length % chunksize > 0:
        chunk_slice.append((num_chunk*chunksize, length))

    if n_jobs < 0:
        n_jobs = os.cpu_count() + 1 - n_jobs

    if n_jobs > 1:
        n_jobs = min(n_jobs, len(chunk_slice))
        executor = loky.get_reusable_executor(
            max_workers=n_jobs, initializer=signalprocessor_initializer,
            initargs=initargs, context="loky", timeout=20)
        
        #~ concurrent.futures.ThreadPoolExecutor
        #~ executor = 
        
        
        
        all_peaks = executor.map(read_process_write_one_chunk, chunk_slice)
        for peaks in all_peaks:
            #~ print('peaks', peaks.size)
            cc.arrays.append_chunk('all_peaks',  peaks)
    else:
        signalprocessor_initializer(*initargs)
        
        for sl in chunk_slice:
            peaks = read_process_write_one_chunk(sl)
            #~ print(peaks)
            cc.arrays.append_chunk('all_peaks',  peaks)
    
    cc.dataio.flush_processed_signals(seg_num=seg_num, chan_grp=cc.chan_grp, processed_length=int(length))


