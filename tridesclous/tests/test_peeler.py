from tridesclous import *

import numpy as np
import scipy.signal
import time
import os
import shutil

import  pyqtgraph as pg

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous import Peeler, Peeler_OpenCl, apply_all_catalogue_steps

from tridesclous.peakdetector import  detect_peaks_in_chunk

from tridesclous.peeler_OLD import PeelerOLD


def setup_catalogue():
    if os.path.exists('test_peeler'):
        shutil.rmtree('test_peeler')
        
    dataio = DataIO(dirname='test_peeler')
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    dataio.add_one_channel_group(channels=[5, 6, 7, 8, 9])
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    
    fullchain_kargs = {
        'duration' : 60.,
        'preprocessor' : {
            'highpass_freq' : 300.,
            'chunksize' : 1024,
            'lostfront_chunksize' : 100,
        },
        'peak_detector' : {
            'peak_sign' : '-',
            'relative_threshold' : 7.,
            'peak_span' : 0.0005,
            #~ 'peak_span' : 0.000,
        },
        'extract_waveforms' : {
            'n_left' : -25,
            'n_right' : 40,
            'nb_max' : 10000,
        },
        'clean_waveforms' : {
            'alien_value_threshold' : 60.,
        },
        'noise_snippet' : {
            'nb_snippet' : 300,
        },        
    }
    
    apply_all_catalogue_steps(catalogueconstructor,
        fullchain_kargs,
        'global_pca', {'n_components': 12},
        'kmeans', {'n_clusters': 12},
        verbose=True)
    catalogueconstructor.trash_small_cluster()
    
    catalogueconstructor.make_catalogue_for_peeler()



def open_catalogue_window():
    dataio = DataIO(dirname='test_peeler')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    app.exec_()


def test_peeler():
    dataio = DataIO(dirname='test_peeler')
    print(dataio)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    peeler = Peeler(dataio)
    
    peeler.change_params(catalogue=initial_catalogue,chunksize=1024)
    
    t1 = time.perf_counter()
    peeler.run(progressbar=False)
    t2 = time.perf_counter()
    print('peeler.run_loop', t2-t1)

    

def test_peeler_empty_catalogue():
    """
    This test peeler with empty catalogue.
    This is like a peak detector.
    Check several chunksize and compare to offline-one-buffer.
    
    """
    dataio = DataIO(dirname='test_peeler')
    #~ print(dataio)
    catalogue = dataio.load_catalogue(chan_grp=0)
    
    # empty catalogue for debug peak detection
    s = catalogue['centers0'].shape
    empty_centers = np.zeros((0, s[1], s[2]), dtype='float32')
    catalogue['centers0'] = empty_centers
    catalogue['centers1'] = empty_centers
    catalogue['centers2'] = empty_centers
    catalogue['cluster_labels'] = np.zeros(0, dtype=catalogue['cluster_labels'].dtype)
        
    
    sig_length = dataio.get_segment_length(0)
    chunksizes = [ 101, 174, 512, 1024, 1023, 10000, 150000]
    #~ chunksizes = [1024,]
    
    previous_peak = None
    
    for chunksize in chunksizes:
        print('**',  chunksize, '**')
        peeler = Peeler(dataio)
        peeler.change_params(catalogue=catalogue,chunksize=chunksize)
        t1 = time.perf_counter()
        #~ peeler.run(progressbar=False)
        peeler.run_offline_loop_one_segment(seg_num=0, progressbar=False)
        t2 = time.perf_counter()
        
        #~ print('n_side', peeler.n_side, 'n_span', peeler.n_span, 'peak_width', peeler.peak_width)
        #~ print('peeler.run_loop', t2-t1)
        
        spikes = dataio.get_spikes(seg_num=0, chan_grp=0)
        labeled_spike = spikes[spikes['cluster_label']>=0]
        unlabeled_spike = spikes[spikes['cluster_label']<0]
        assert labeled_spike.size == 0
        
        is_sorted = np.all(np.diff(unlabeled_spike['index'])>=0)
        assert is_sorted
        
        
        online_peaks = unlabeled_spike['index']

        i_stop = sig_length-catalogue['params_signalpreprocessor']['lostfront_chunksize']-peeler.n_side+peeler.n_span
        sigs = dataio.get_signals_chunk(signal_type='processed', i_stop=i_stop)
        offline_peaks  = detect_peaks_in_chunk(sigs, peeler.n_span, peeler.relative_threshold, peeler.peak_sign)
        
        offline_peaks  = offline_peaks[offline_peaks<=online_peaks[-1]]
        
        assert offline_peaks.size == online_peaks.size
        np.testing.assert_array_equal(offline_peaks, online_peaks)
        
        if previous_peak is not None:
            last = min(previous_peak[-1], online_peaks[-1])
            previous_peak = previous_peak[previous_peak<=last]
            online_peaks_cliped = online_peaks[online_peaks<=last]
            assert  previous_peak.size == online_peaks_cliped.size
            np.testing.assert_array_equal(previous_peak, online_peaks_cliped)
        
        previous_peak = online_peaks
        

def test_peeler_several_chunksize():
    
    dataio = DataIO(dirname='test_peeler')
    print(dataio)
    catalogue = dataio.load_catalogue(chan_grp=0)
    
    
    all_spikes = []
    sig_length = dataio.get_segment_length(0)
    chunksizes = [ 174, 512, 1024, 1023, 10000, 150000]
    #~ chunksizes = [512, 1024,]
    for chunksize in chunksizes:
        print('**',  chunksize, '**')
        peeler = Peeler(dataio)
        peeler.change_params(catalogue=catalogue,chunksize=chunksize)
        t1 = time.perf_counter()
        peeler.run_offline_loop_one_segment(seg_num=0, progressbar=False)
        t2 = time.perf_counter()
        print('n_side', peeler.n_side, 'n_span', peeler.n_span, 'peak_width', peeler.peak_width)
        print('peeler.run_loop', t2-t1)
        
        spikes = dataio.get_spikes(seg_num=0, chan_grp=0)
        all_spikes.append(spikes)
    
    
    # clip to last spike
    last = min([spikes[-1]['index'] for spikes in all_spikes])
    for i, chunksize in enumerate(chunksizes):
        spikes = all_spikes[i]
        all_spikes[i] = spikes[spikes['index']<=last]
    
    
    previsous_spikes = None
    for i, chunksize in enumerate(chunksizes):
        print('**',  chunksize, '**')
        spikes = all_spikes[i]
        is_sorted = np.all(np.diff(spikes['index'])>=0)
        assert is_sorted
        
        labeled_spike = spikes[spikes['cluster_label']>=0]
        unlabeled_spike = spikes[spikes['cluster_label']<0]
        print('labeled_spike.size', labeled_spike.size, 'unlabeled_spike.size', unlabeled_spike.size)
        
        
        if previsous_spikes is not None:
            assert  previsous_spikes.size == spikes.size
            np.testing.assert_array_equal(previsous_spikes['index'], spikes['index'])
            np.testing.assert_array_equal(previsous_spikes['cluster_label'], spikes['cluster_label'])
        
        previsous_spikes = spikes

def open_PeelerWindow():
    dataio = DataIO(dirname='test_peeler')
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()


def test_export_spikes():
    dataio = DataIO(dirname='test_peeler')
    dataio.export_spikes()

def test_compare_peeler():

    dataio = DataIO(dirname='test_peeler')
    print(dataio)
    
    all_spikes = []
    #~ for peeler_class in [Peeler,]:
    #~ for peeler_class in [Peeler_OpenCl,]:
    for peeler_class in [Peeler, Peeler_OpenCl]:
        print()
        print(peeler_class)
        initial_catalogue = dataio.load_catalogue(chan_grp=0)
        
        peeler = peeler_class(dataio)
        
        peeler.change_params(catalogue=initial_catalogue,  chunksize=1024)
        
        t1 = time.perf_counter()
        #~ peeler.run_offline_loop_one_segment(duration=None, progressbar=False)
        peeler.run_offline_loop_one_segment(duration=4., progressbar=False)
        t2 = time.perf_counter()
        print('peeler.run_loop', t2-t1)
        
        
        all_spikes.append(dataio.get_spikes(chan_grp=0).copy())
        
        #~ print(dataio.get_spikes(chan_grp=0).size)
    
    #~ all_spikes[0] = all_spikes[0][88+80:88+81+10]
    #~ all_spikes[1] = all_spikes[1][88+80:88+81+10]

    #~ all_spikes[0] = all_spikes[0][:88+81]
    #~ all_spikes[1] = all_spikes[1][:88+81]
    
    #~ for spikes in all_spikes:
        #~ print(spikes)
        #~ print(spikes.size)
        #~ assert all_spikes[0].size == spikes.size
        #~ assert np.all(all_spikes[0]['index'] == spikes['index'])
        #~ assert np.all(all_spikes[0]['cluster_label'] == spikes['cluster_label'])
        #~ assert np.all(np.abs(all_spikes[0]['jitter'] - spikes['jitter'])<0.0001)
    
    
if __name__ =='__main__':
    #~ setup_catalogue()
    
    #~ open_catalogue_window()
    
    test_peeler()
    
    #~ test_peeler_empty_catalogue()
    
    #~ test_peeler_several_chunksize()
    
    #~ open_PeelerWindow()
    
    #~ test_export_spikes()
    
    
    #~ test_compare_peeler()
    
    #~ open_PeelerWindow()
