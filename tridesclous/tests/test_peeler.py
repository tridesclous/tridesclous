import pytest

from tridesclous import *

import numpy as np
import scipy.signal
import time
import os
import shutil

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous import Peeler
from tridesclous.peeler_cl import Peeler_OpenCl

from tridesclous.peakdetector import  detect_peaks_in_chunk


from tridesclous.tests.testingtools import setup_catalogue
from tridesclous.tests.testingtools import ON_CI_CLOUD




def setup_module():
    setup_catalogue('test_peeler', dataset_name='olfactory_bulb')
    setup_catalogue('test_peeler2', dataset_name='olfactory_bulb', duration=16.)

def teardown_module():
    shutil.rmtree('test_peeler')
    shutil.rmtree('test_peeler2')


def open_catalogue_window():
    dataio = DataIO(dirname='test_peeler')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    app.exec_()


def test_peeler_geometry():
    dataio = DataIO(dirname='test_peeler')
    
    catalogue0 = dataio.load_catalogue(chan_grp=0)
    # catalogue1 = dataio.load_catalogue(chan_grp=0, name='with_oversampling')
    
        
        
    # for catalogue in (catalogue0, catalogue1):
    for catalogue in (catalogue0, ):
        print()
        print('engine=geometrical')
        print('inter_sample_oversampling', catalogue['inter_sample_oversampling'])
        peeler = Peeler(dataio)

        peeler.change_params(engine='geometrical',
                                    catalogue=catalogue,
                                    chunksize=1024)
                                    
                                    

        t1 = time.perf_counter()
        peeler.run(progressbar=False)
        t2 = time.perf_counter()
        print('peeler.run_loop', t2-t1)

        spikes = dataio.get_spikes(chan_grp=0).copy()
        labels = catalogue['clusters']['cluster_label']
        count_by_label = [np.sum(spikes['cluster_label'] == label) for label in labels]
        print(labels)
        print(count_by_label)


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_peeler_geometry_cl():
    dataio = DataIO(dirname='test_peeler')
    #~ catalogue = dataio.load_catalogue(chan_grp=0)
    catalogue0 = dataio.load_catalogue(chan_grp=0)
    # catalogue1 = dataio.load_catalogue(chan_grp=0, name='with_oversampling')
    
    # for catalogue in (catalogue0, catalogue1):
    for catalogue in (catalogue0, ):
        print()
        print('engine=geometrical_opencl')
        print('inter_sample_oversampling', catalogue['inter_sample_oversampling'])

        peeler = Peeler(dataio)
        
        catalogue['clean_peaks_params']['alien_value_threshold'] = None
        
        
        peeler.change_params(engine='geometrical_opencl',
                                    catalogue=catalogue,
                                    chunksize=1024,
                                    speed_test_mode=True,
                                    )

        t1 = time.perf_counter()
        peeler.run(progressbar=False)
        t2 = time.perf_counter()
        print('peeler.run_loop', t2-t1)

        spikes = dataio.get_spikes(chan_grp=0).copy()
        labels = catalogue['clusters']['cluster_label']
        count_by_label = [np.sum(spikes['cluster_label'] == label) for label in labels]
        print(labels)
        print(count_by_label)
        
        
        run_times = peeler.get_run_times(chan_grp=0, seg_num=0)
        print(run_times)


    
@pytest.mark.skipif(ON_CI_CLOUD, reason='TOO_OLD')
def test_peeler_empty_catalogue():
    """
    This test peeler with empty catalogue.
    This is like a peak detector.
    Check several chunksize and compare to offline-one-buffer.
    
    THIS TEST IS TOO OLD need to be rewritten
    
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
        peeler.change_params(engine='geometrical', catalogue=catalogue,chunksize=chunksize, 
                save_bad_label=True)
        t1 = time.perf_counter()
        #~ peeler.run(progressbar=False)
        #~ peeler.run_offline_loop_one_segment(seg_num=0, progressbar=False)
        peeler.run(progressbar=False)
        t2 = time.perf_counter()
        
        #~ print('n_side', peeler.n_side, 'n_span', peeler.n_span, 'peak_width', peeler.peak_width)
        #~ print('peeler.run_loop', t2-t1)
        
        spikes = dataio.get_spikes(seg_num=0, chan_grp=0)
        labeled_spike = spikes[spikes['cluster_label']>=0]
        unlabeled_spike = spikes[spikes['cluster_label']<0]
        assert labeled_spike.size == 0
        print(unlabeled_spike.size)
        
        is_sorted = np.all(np.diff(unlabeled_spike['index'])>=0)
        
        
        
        online_peaks = unlabeled_spike['index']
        engine = peeler.peeler_engine
        
        i_stop = sig_length-catalogue['signal_preprocessor_params']['pad_width']-engine.extra_size+engine.n_span
        sigs = dataio.get_signals_chunk(signal_type='processed', i_stop=i_stop)
        offline_peaks  = detect_peaks_in_chunk(sigs, engine.n_span, engine.relative_threshold, engine.peak_sign)
        print(offline_peaks.size)
        
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
        


@pytest.mark.skipif(ON_CI_CLOUD, reason='To hard for CI')
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
        peeler.change_params(engine='geometrical', catalogue=catalogue,chunksize=chunksize)
        t1 = time.perf_counter()
        peeler.run(progressbar=False)
        t2 = time.perf_counter()
        print('extra_size', peeler.peeler_engine.extra_size, 'n_span', peeler.peeler_engine.n_span,
                        'peak_width', peeler.peeler_engine.peak_width)
        print('peeler.run_loop', t2-t1)
        
        # copy is need because the memmap is reset at each loop
        spikes = dataio.get_spikes(seg_num=0, chan_grp=0).copy()
        all_spikes.append(spikes)
        print(spikes.size)
    
    
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
        print(spikes)
        
        # TODO: Peeler chunksize influence the number of spikes
        
        if previsous_spikes is not None:
            assert  previsous_spikes.size == spikes.size
            np.testing.assert_array_equal(previsous_spikes['index'], spikes['index'])
            np.testing.assert_array_equal(previsous_spikes['cluster_label'], spikes['cluster_label'])
        
        previsous_spikes = spikes



def test_peeler_with_and_without_preprocessor():
    
    if ON_CI_CLOUD:
        engines = ['geometrical']
    else:
        engines = ['geometrical', 'geometrical_opencl']

    
    #~ engines = ['geometrical_opencl']
    
    for engine in engines:
        for i in range(2):
        #~ for i in [1]:
        
            print()
            if i == 0:
                print(engine, 'without processing')
                dataio = DataIO(dirname='test_peeler')
            else:
                print(engine, 'with processing')
                dataio = DataIO(dirname='test_peeler2')
            
            catalogue = dataio.load_catalogue(chan_grp=0)
            
            peeler = Peeler(dataio)
            peeler.change_params(engine=engine, catalogue=catalogue, chunksize=1024)
            t1 = time.perf_counter()
            peeler.run(progressbar=False)
            t2 = time.perf_counter()
            print('peeler run_time', t2 - t1)
            spikes = dataio.get_spikes(chan_grp=0).copy()
            labels = catalogue['clusters']['cluster_label']
            count_by_label = [np.sum(spikes['cluster_label'] == label) for label in labels]
            print(labels)
            print(count_by_label)
            
    
    


def open_PeelerWindow():
    dataio = DataIO(dirname='test_peeler')
    #~ dataio = DataIO(dirname='test_peeler2')
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()


def test_export_spikes():
    dataio = DataIO(dirname='test_peeler')
    dataio.export_spikes()







def debug_compare_peeler_engines():
    # this do not work because oversampling is not handle
    dataio = DataIO(dirname='test_peeler')
    print(dataio)
    
    
    engine_list = [ 
            ('geometrical argmin opencl', 'geometrical', {}),
            ('geometrical_opencl', 'geometrical_opencl', {}),
        ]
    
    all_spikes =  []
    for name, engine, kargs in engine_list:
        #~ print()
        #~ print(name)
        # catalogue = dataio.load_catalogue(chan_grp=0, name='with_oversampling')
        catalogue = dataio.load_catalogue(chan_grp=0)


        peeler = Peeler(dataio)
        peeler.change_params(engine=engine, catalogue=catalogue,chunksize=1024, **kargs)

        t1 = time.perf_counter()
        peeler.run(progressbar=False, duration=None)
        t2 = time.perf_counter()
        print(name, 'run', t2-t1)
        
        
        spikes = dataio.get_spikes(chan_grp=0).copy()
        #~ print(spikes.size)
        all_spikes.append(spikes)
        
        
        #~ print(dataio.get_spikes(chan_grp=0).size)
    
    print()
    #~ all_spikes[0] = all_spikes[0][88+80:88+81+10]
    #~ all_spikes[1] = all_spikes[1][88+80:88+81+10]

    #~ all_spikes[0] = all_spikes[0][:88+81]
    #~ all_spikes[1] = all_spikes[1][:88+81]
    
    labels = catalogue['clusters']['cluster_label']
    
    for i, spikes in enumerate(all_spikes):
        name = engine_list[i][0]
        print()
        print(name)
        print(spikes[:10])
        print(spikes.size)
        
        count_by_label = [np.sum(spikes['cluster_label'] == label) for label in labels]
        print(count_by_label)
            
        
        #~ assert all_spikes[0].size == spikes.size
        #~ assert np.all(all_spikes[0]['index'] == spikes['index'])
        #~ assert np.all(all_spikes[0]['cluster_label'] == spikes['cluster_label'])
        #~ assert np.all(np.abs(all_spikes[0]['jitter'] - spikes['jitter'])<0.0001)
    

if __name__ =='__main__':
    #~ setup_module()
    
    #~ open_catalogue_window()
    
    
    #~ test_peeler_geometry()
    
    #~ test_peeler_geometry_cl()
    
    
    #~ test_peeler_empty_catalogue()
    
    test_peeler_several_chunksize()
    
    #~ test_peeler_with_and_without_preprocessor()
    
    #~ test_export_spikes()
    
    
    #~ debug_compare_peeler_engines()
    
    #~ open_PeelerWindow()
    
    #~ teardown_module()
