from tridesclous import DataIO, PeakDetector, WaveformExtractor, Clustering, Peeler
from tridesclous import SpikeSorter

from test_dataio import download_locust
import os
import neo
import quantities as pq
import numpy as np

def test_spikesorter():
    if os.path.exists('datatest/data.h5'):
        os.remove('datatest/data.h5')
    
    spikesorter = SpikeSorter(dirname = 'datatest')
    
    sigs_by_trials, sampling_rate, ch_names = download_locust(trial_names = ['trial_01', 'trial_02', 'trial_03'])
    for seg_num in range(3):
        sigs = sigs_by_trials[seg_num]
        spikesorter.dataio.append_signals_from_numpy(sigs, seg_num = seg_num,
                    t_start = 0.+5*seg_num, sampling_rate =  sampling_rate,
                    already_hp_filtered = True, channels = ch_names)
    
    print('### after data import ###')
    print(spikesorter)
    
    spikesorter.detect_peaks_extract_waveforms(seg_nums = 'all',  threshold=-4, peak_sign = '-', n_span = 2,  n_left=-30, n_right=50)
    print('### after peak detection ###')
    print(spikesorter.summary(level=1))
    
    spikesorter.project(method = 'pca', n_components = 5)
    spikesorter.find_clusters(7)
    print('### after clustering ###')
    print(spikesorter.summary(level=1))




def test_spikesorter_neo():
    if os.path.exists('datatest_neo/data.h5'):
        os.remove('datatest_neo/data.h5')
    
    spikesorter = SpikeSorter(dirname = 'datatest_neo')
    
    filenames = ['tem16a00.IOT', 'tem16a01.IOT', 'tem16a02.IOT', ]
    
    for filename in filenames:
        blocks = neo.RawBinarySignalIO(filename).read(sampling_rate = 10.*pq.kHz,
                    t_start = 0. *pq.S, unit = pq.V, nbchannel = 16, bytesoffset = 0,
                    dtype = 'int16', rangemin = -10, rangemax = 10)
    
        channel_indexes = np.arange(14)
        spikesorter.dataio.append_signals_from_neo(blocks, channel_indexes = channel_indexes, 
                                    already_hp_filtered = False)
    
    print('### after data import ###')
    print(spikesorter.summary(level=1))

    
    
    spikesorter.apply_filter(highpass_freq = 300.)
    print('### after filtering ###')
    print(spikesorter.summary(level=1))
    
    
    spikesorter.detect_peaks_extract_waveforms(seg_nums = 'all',  threshold=-2., peak_sign = '-', n_span = 2,  n_left=-30, n_right=50)
    print('### after peak detection ###')
    print(spikesorter.summary(level=1))
    
    spikesorter.project(method = 'pca', n_components = 5)
    spikesorter.find_clusters(7)
    print('### after clustering ###')
    print(spikesorter.summary(level=1))


    
if __name__ == '__main__':
    test_spikesorter()
    #~ test_spikesorter_neo()
