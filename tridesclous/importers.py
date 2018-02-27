import os
import time
import configparser

import numpy as np

from .dataio import DataIO
from . catalogueconstructor import CatalogueConstructor, _dtype_peak


try:
    import h5py
    HAVE_H5PY = True
except:
    HAVE_H5PY = False

_supported_formats = ('raw_binary', 'mcs_raw_binary')

def import_from_spykingcircus(data_filename, spykingcircus_dirname, tdc_dirname):
    
    
    
    assert HAVE_H5PY, 'h5py is not installed'
    assert not os.path.exists(tdc_dirname), 'tdc already exisst please remvoe {}'.format(tdc_dirname)
    
    if spykingcircus_dirname.endswith('/'):
        spykingcircus_dirname = spykingcircus_dirname[:-1]
    

    #parse spyking circus params file
    params_filename = spykingcircus_dirname + '.params'
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(params_filename)
    
    #Set DataIO and source
    file_fmormat = config['data']['file_format']
    assert file_fmormat  in _supported_formats, 'only  {} are supported'.format(_supported_formats)

    dtype = config['data']['data_dtype']
    nb_channels = config.getint('data', 'nb_channels')
    if file_fmormat == 'mcs_raw_binary':
        import re
        data_filename = 'patch1.raw'
        with open(data_filename, 'rb') as f:
            header = f.read(5000).decode('Windows-1252')
        data_offset = re.search("EOH\r\n", header).start() + 5
    else:
        data_offset = config.getint('data', 'data_offset')
    sample_rate = config.getfloat('data', 'sampling_rate')
    
    dataio = DataIO(dirname=tdc_dirname)
    filenames = [data_filename]
    dataio.set_data_source(type='RawData', filenames=filenames, dtype=dtype,
                                     total_channel=nb_channels, sample_rate=sample_rate, offset=data_offset)
    
    #set probe file (supposed to be in same path)
    probe = config.get('data', 'mapping')
    probe_filename = os.path.join(os.path.dirname(data_filename), probe)
    dataio.set_probe_file(probe_filename)
    
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    # filter
    if config.getboolean('filtering', 'filter'):
        highpass_freq = config.getfloat('filtering', 'cut_off')
    else:
        highpass_freq = None
    common_ref_removal = config.getboolean('filtering', 'remove_median')
    
    # detection
    relative_threshold = config.getfloat('detection', 'spike_thresh')
    if config.get('detection', 'peaks') == 'negative':
        peak_sign='-'
    elif config.get('detection', 'peaks') == 'positive':
        peak_sign='+'
    else:
        raise(NotImlementedError)
    
    engine = 'numpy'
    #~ engine = 'opencl'

    catalogueconstructor.set_preprocessor_params(chunksize=1024,
            memory_mode='memmap',
            
            #signal preprocessor
            signalpreprocessor_engine=engine,
            highpass_freq=highpass_freq, 
            lowpass_freq=None,
            common_ref_removal=common_ref_removal,
            lostfront_chunksize=128,
            
            #peak detector
            peakdetector_engine=engine,
            peak_sign=peak_sign, 
            relative_threshold=relative_threshold,
            peak_span=1./sample_rate,
            )
    
    t1 = time.perf_counter()
    duration=30.
    catalogueconstructor.estimate_signals_noise(seg_num=0, duration=duration)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    duration = dataio.get_segment_length(0)/dataio.sample_rate
    t1 = time.perf_counter()
    catalogueconstructor.run_signalprocessor(duration=duration, detect_peak=False)
    t2 = time.perf_counter()
    print('run_signalprocessor', t2-t1)
    
    
    
    #read peaks from results files
    name = os.path.basename(spykingcircus_dirname)
    result_filename = os.path.join(spykingcircus_dirname, '{}.result.hdf5'.format(name))
    result = h5py.File(result_filename,'r')
    
    #~ result.visit(lambda p: print(p))
    
    all_peaks = []
    for k in result['spiketimes'].keys():
        #~ print('k', k)
        _, label = k.split('_')
        label = int(label)
        indexes = np.array(result['spiketimes'][k])
        peaks = np.zeros(indexes.shape, dtype=_dtype_peak)
        peaks['index'][:] = indexes
        peaks['cluster_label'][:] = label
        peaks['segment'][:] = 0
        
        all_peaks.append(peaks)
        
    
    all_peaks = np.concatenate(all_peaks)
    order = np.argsort(all_peaks['index'])
    all_peaks = all_peaks[order]
    
    nb_peak = all_peaks.size
    
    catalogueconstructor.arrays.create_array('all_peaks', _dtype_peak, (nb_peak,), 'memmap')
    catalogueconstructor.all_peaks[:] = all_peaks
    catalogueconstructor.on_new_cluster()
    
    
    N_t = config.getfloat('data', 'N_t')
    n_right = int(N_t*sample_rate/1000.)//2
    n_left = -n_right
    
    catalogueconstructor.extract_some_waveforms(n_left=n_left, n_right=n_right,   mode='rand', nb_max=10000)
    catalogueconstructor.project(method='peak_max')
    
    
    
    return catalogueconstructor
    








