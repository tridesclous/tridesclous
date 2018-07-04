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
    
    
    #~ for seg_num in [0, 1, 2]:
        #~ print()
        #~ spikes = dataio.get_spikes(seg_num=seg_num)
        #~ print(spikes.size)
        #~ print(np.sum(spikes['cluster_label']>=0))
        #~ d = np.diff(spikes['index'])
        #~ print(np.any(d<0))
        #~ bad_spikes = spikes[spikes['cluster_label']<0]
        #~ print(bad_spikes.size)
        #~ d_bad = np.diff(bad_spikes['index'])
        #~ print(np.any(d_bad<=0))
    
    #~ print(dataio.get_spikes(seg_num=0).size)
    #~ print(dataio.get_spikes(seg_num=1).size)
    #~ print(dataio.get_spikes(seg_num=2).size)



def test_peeler_several_chunksize():
    
    dataio = DataIO(dirname='test_peeler')
    print(dataio)
    catalogue = dataio.load_catalogue(chan_grp=0)
    
    if True:
        # empty catalogue for debug peak detection
        s = catalogue['centers0'].shape
        empty_centers = np.zeros((0, s[1], s[2]), dtype='float32')
        catalogue['centers0'] = empty_centers
        catalogue['centers1'] = empty_centers
        catalogue['centers2'] = empty_centers
        catalogue['cluster_labels'] = np.zeros(0, dtype=catalogue['cluster_labels'].dtype)
        
    
    sig_length = dataio.get_segment_length(0)
    #~ chunksizes = [ 150000]
    chunksizes = [ 174, 512, 1024, 1023, 10000, 150000]
    #~ chunksizes = [ 69, 128, 512, 1024, 1023]
    #~ chunksizes = [1024,]
    for chunksize in chunksizes:
    #~ for chunksize in [ 64]:
        print('**',  chunksize, '**')
        peeler = Peeler(dataio)
        
        
        peeler.change_params(catalogue=catalogue,chunksize=chunksize)
        
        t1 = time.perf_counter()
        #~ peeler.run(progressbar=False)
        peeler.run_offline_loop_one_segment(seg_num=0, progressbar=False)
        t2 = time.perf_counter()
        print('n_side', peeler.n_side, 'n_span', peeler.n_span, 'peak_width', peeler.peak_width)
        print('peeler.run_loop', t2-t1)
        
        spikes = dataio.get_spikes(seg_num=0, chan_grp=0)
        #~ spikes = spikes[spikes['index']<(sig_length-2*chunksize)]
        #~ spikes = spikes[spikes['index']>(2*chunksize)]
        
        
        print('nb total spikes', spikes.size, 'ordered', np.all(np.diff(spikes['index'])>=0))
        if not np.all(np.diff(spikes['index'])>=0):
            ind_unordered,  = np.nonzero(np.diff(spikes['index'])<0)
            for ind in ind_unordered:
                print('  ', ind, spikes[ind-1:ind+2])
            
            
        labeled_spike = spikes[spikes['cluster_label']>=0]
        print('nb labeled spikes', labeled_spike.size, 'ordered', np.all(np.diff(labeled_spike['index'])>=0))
        unlabeled_spike = spikes[spikes['cluster_label']<0]
        print('nb unlabeled spikes', unlabeled_spike.size, 'ordered', np.all(np.diff(unlabeled_spike['index'])>=0))
        print('unique unlabeled', np.unique(unlabeled_spike['index']).size==unlabeled_spike.size)
        #~ print('unique unlabeled', np.unique(unlabeled_spike['index']).size, unlabeled_spike.size)
        
        #~ begin_spike = spikes[spikes['index']<3000]
        #~ begin_spike = spikes[(spikes['index']>10000) & (spikes['index']<15000)]
        #~ print(begin_spike)
        
        #~ print(sig_length-catalogue['params_signalpreprocessor']['lostfront_chunksize'])
        #~ i_stop=sig_length-catalogue['params_signalpreprocessor']['lostfront_chunksize']-peeler.n_side+peeler.n_span
        #~ sigs = dataio.get_signals_chunk(signal_type='processed', i_stop=i_stop)
        #~ total_peaks  = detect_peaks_in_chunk(sigs, peeler.n_span, peeler.relative_threshold, peeler.peak_sign)
        #~ total_peaks = total_peaks[total_peaks<(sig_length-2*chunksize)]
        #~ total_peaks = total_peaks[total_peaks>(2*chunksize)]
        
        #~ print('long terme total_peak', total_peaks.size)
        
        #~ if total_peaks.size==  spikes['index'].size:
            #~ print('ALL EQUAL', np.all(np.equal(total_peaks, spikes['index'])))
        
        #~ print(total_peaks[:5], total_peaks[-5:])
        #~ print(spikes['index'][:5], spikes['index'][-5:])
        

    


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
    
    #~ test_peeler()
    
    test_peeler_several_chunksize()
    
    #~ open_PeelerWindow()
    
    #~ test_export_spikes()
    
    
    #~ test_compare_peeler()
    
    #~ open_PeelerWindow()
