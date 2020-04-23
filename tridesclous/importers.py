import os
import time
import configparser
from pathlib import Path

import numpy as np

from .dataio import DataIO
from . catalogueconstructor import CatalogueConstructor, _dtype_peak
from .autoparams import get_auto_params_for_catalogue

try:
    import h5py
    HAVE_H5PY = True
except:
    HAVE_H5PY = False
    
try:
    import spikeextractors as se
    import spiketoolkit as st
    HAVE_SPIKEINTERFACE = True
except:
    HAVE_SPIKEINTERFACE = True



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
    _supported_formats = ('raw_binary', 'mcs_raw_binary')
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
    
    
    cc = CatalogueConstructor(dataio=dataio)
    
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
    
    
    cc.set_global_params(
            chunksize=1024,
            memory_mode='memmap',
            mode='sparse',
            adjacency_radius_um=400,
            sparse_threshold=1.5)
    
    # params preprocessor
    cc.set_preprocessor_params(
            engine='numpy',
            highpass_freq=highpass_freq, 
            lowpass_freq=None,
            common_ref_removal=common_ref_removal,
            lostfront_chunksize=-1,
        )
    
    # params peak detector
    cc.set_peak_detector_params(
            method='global',
            engine='numpy',
            peak_sign=peak_sign, 
            relative_threshold=relative_threshold,
            peak_span=1./sample_rate,
            )

    
    t1 = time.perf_counter()
    duration=30.
    cc.estimate_signals_noise(seg_num=0, duration=duration)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    duration = dataio.get_segment_length(0)/dataio.sample_rate
    t1 = time.perf_counter()
    cc.run_signalprocessor(duration=duration, detect_peak=False)
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
    
    cc.arrays.create_array('all_peaks', _dtype_peak, (nb_peak,), 'memmap')
    cc.all_peaks[:] = all_peaks
    cc.on_new_cluster()
    
    
    N_t = config.getfloat('data', 'N_t')
    n_right = int(N_t*sample_rate/1000.)//2
    n_left = -n_right
    
    cc.set_waveform_extractor_params(n_left=n_left, n_right=n_right)
    cc.sample_some_peaks( mode='rand', nb_max=10000)

    cc.extract_some_noise()
    cc.extract_some_features(method='peak_max')
    
    self.all_peaks['cluster_label'][cc.some_peaks_index] = all_peaks[cc.some_peaks_index]['cluster_label']
    cc.on_new_cluster()
    cc.compute_all_centroid()
    cc.refresh_colors()
    
    return cc
    

def import_from_spike_interface(recording, sorting, tdc_dirname, spike_per_cluster=300, align_peak=True, catalogue_params={}):
    output_folder = Path(tdc_dirname)
    
    # save prb file:
    probe_file = output_folder / 'probe.prb'
    se.save_to_probe_file(recording, probe_file)

    # save binary file (chunk by hcunk) into a new file
    raw_filename = output_folder / 'raw_signals.raw'
    n_chan = recording.get_num_channels()
    chunksize = 2**24// n_chan
    se.write_to_binary_dat_format(recording, raw_filename, time_axis=0, dtype='float32', chunk_size=chunksize)
    dtype='float32'
    offset = 0
    sr = recording.get_sampling_frequency()

    # initialize source and probe file
    dataio = DataIO(dirname=str(output_folder))
    nb_chan = recording.get_num_channels()

    dataio.set_data_source(type='RawData', filenames=[str(raw_filename)],
                               dtype=dtype, sample_rate=sr,
                               total_channel=nb_chan, offset=offset)
    dataio.set_probe_file(str(probe_file))
    print(dataio)
    
    params = get_auto_params_for_catalogue(dataio)
    for k in params:
        if k in catalogue_params:
            if isinstance(catalogue_params[k], dict):
                params[k].update(catalogue_params[k])
            else:
                params[k] = catalogue_params[k]
    
    cc = CatalogueConstructor(dataio=dataio)

    # global params
    d = {k:params[k] for k in ('chunksize', 'mode', )}
    cc.set_global_params(**d)

    # params preprocessor
    d = dict(params['preprocessor'])
    cc.set_preprocessor_params(**d)
    
    # params peak detector
    d = dict(params['peak_detector'])
    cc.set_peak_detector_params(**d)
    
    t1 = time.perf_counter()
    noise_duration = min(10., params['duration'], dataio.get_segment_length(seg_num=0)/dataio.sample_rate*.99)
    t1 = time.perf_counter()
    cc.estimate_signals_noise(seg_num=0, duration=noise_duration)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    duration = dataio.get_segment_length(0)/dataio.sample_rate
    t1 = time.perf_counter()
    cc.run_signalprocessor(duration=duration, detect_peak=False)
    t2 = time.perf_counter()
    print('run_signalprocessor', t2-t1)
    
    wf_left_ms = params['extract_waveforms']['wf_left_ms']
    wf_right_ms = params['extract_waveforms']['wf_right_ms']
    n_left = int(wf_left_ms / 1000. * dataio.sample_rate)
    n_right = int(wf_right_ms / 1000. * dataio.sample_rate)
    
    sig_size = dataio.get_segment_length(seg_num=0)
    
    peak_shift = {}
    extremum_channel = {}

    t1 = time.perf_counter()
    # compute the shift to have the true max at -n_left
    for label in sorting.get_unit_ids():
        indexes = sorting.get_unit_spike_train(label)
        indexes = indexes[indexes<(sig_size-n_right-1)]
        indexes = indexes[indexes>(-n_left+1)]
        if indexes.size>spike_per_cluster:
            keep = np.random.choice(indexes.size, spike_per_cluster, replace=False)
            indexes = indexes[keep]
        wfs = dataio.get_some_waveforms(seg_num=0, chan_grp=0, peak_sample_indexes=indexes, n_left=n_left, n_right=n_right)
        wf0 = np.median(wfs, axis=0)
        # maybe use '-' or '+'
        chan_max = np.argmax(np.max(np.abs(wf0), axis=0))
        ind_max = np.argmax(np.abs(wf0[:, chan_max]))
        extremum_channel[label] = chan_max
        shift = ind_max + n_left
        peak_shift[label] = shift
        #~ print(label,'>', shift)
    t2 = time.perf_counter()
    print('template', t2-t1)
    
    
    all_peaks = []
    for label in sorting.get_unit_ids():
        indexes = sorting.get_unit_spike_train(label)
        indexes = indexes[indexes<(sig_size-n_right-1)]
        indexes = indexes[indexes>(-n_left+1)]
        peaks = np.zeros(indexes.shape, dtype=_dtype_peak)
        if align_peak:
            indexes = indexes + peak_shift[label]
        peaks['index'][:] = indexes
        peaks['cluster_label'][:] = label
        peaks['segment'][:] = 0
        peaks['channel'][:] = extremum_channel[label]
        
        
        
        all_peaks.append(peaks)
    
    all_peaks = np.concatenate(all_peaks)
    order = np.argsort(all_peaks['index'])
    all_peaks = all_peaks[order]

    nb_peak = all_peaks.size
    
    cc.arrays.create_array('all_peaks', _dtype_peak, (nb_peak,), 'memmap')
    cc.all_peaks[:] = all_peaks
    cc.on_new_cluster()
    #~ print(cc.clusters)
    
    #  waveform per cluster selection
    t1 = time.perf_counter()
    global_keep = np.zeros(all_peaks.size, dtype='bool')
    for label in sorting.get_unit_ids():
        inds, = np.nonzero(all_peaks['cluster_label'] == label)
        if inds.size>spike_per_cluster:
            incluster_sel = np.random.choice(inds.size, spike_per_cluster, replace=False)
            inds = inds[incluster_sel]
        global_keep[inds] = True
    index, = np.nonzero(global_keep)

    cc.set_waveform_extractor_params(n_left=n_left, n_right=n_right)
    
    t1 = time.perf_counter()
    cc.sample_some_peaks(mode='force', index=index)
    t2 = time.perf_counter()
    print('sample_some_peaks', t2-t1)
    
    cc.clean_peaks(alien_value_threshold=None)
    
    cc.extract_some_noise()
    
    cc.extract_some_features(method=params['feature_method'], **params['feature_kargs'])
    
    # put back label 
    cc.all_peaks['cluster_label'][cc.some_peaks_index] = all_peaks[cc.some_peaks_index]['cluster_label']
    t1 = time.perf_counter()
    cc.on_new_cluster()
    cc.compute_all_centroid()
    t2 = time.perf_counter()
    print('compute_all_centroid', t2-t1)
    cc.refresh_colors()
    print(cc)
    
    return cc






