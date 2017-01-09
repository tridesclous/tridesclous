from tridesclous import *

import numpy as np
import scipy.signal
import time
import os
import shutil

import  pyqtgraph as pg

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.peeler import Peeler



def setup_catalogue():
    if os.path.exists('test_peeler'):
        shutil.rmtree('test_peeler')
        
    dataio = DataIO(dirname='test_peeler')
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    dataio.add_one_channel_group(channels=[5, 6, 7, 8, 9])
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    #~ print(catalogueconstructor)

    catalogueconstructor.set_preprocessor_params(chunksize=1024,
            memory_mode='memmap',
            
            #signal preprocessor
            highpass_freq=300,
            backward_chunksize=1280,
            
            #peak detector
            peak_sign='-', relative_threshold=7, peak_span=0.0005,
            )
    
    t1 = time.perf_counter()
    catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    
    t1 = time.perf_counter()
    catalogueconstructor.run_signalprocessor()
    t2 = time.perf_counter()
    print('run_signalprocessor', t2-t1)
    
    print(catalogueconstructor)
    
    
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(n_left=-25, n_right=40,  nb_max=10000)
    t2 = time.perf_counter()
    print('extract_some_waveforms', t2-t1)

    t1 = time.perf_counter()
    n_left, n_right = catalogueconstructor.find_good_limits()
    t2 = time.perf_counter()
    print('find_good_limits', t2-t1)
    print(n_left, n_right)
    print(catalogueconstructor.some_waveforms.shape)
    

    # PCA
    t1 = time.perf_counter()
    catalogueconstructor.project(method='pca', n_components=12, batch_size=16384)
    t2 = time.perf_counter()
    print('project', t2-t1)
    
    
    
    # cluster
    t1 = time.perf_counter()
    catalogueconstructor.find_clusters(method='kmeans', n_clusters=12)
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)
    
    # trash_small_cluster
    catalogueconstructor.trash_small_cluster()


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
    initial_catalogue = dataio.load_catalogue()

    peeler = Peeler(dataio)
    
    peeler.change_params(catalogue=initial_catalogue, n_peel_level=2, chunksize=1024)
    
    t1 = time.perf_counter()
    peeler.run()
    t2 = time.perf_counter()
    print('peeler.run_loop', t2-t1)


def open_PeelerWindow():
    dataio = DataIO(dirname='test_peeler')
    initial_catalogue = dataio.load_catalogue()

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()
    
    
    
    
if __name__ =='__main__':
    #~ setup_catalogue()
    
    #~ open_catalogue_window()
    
    #~ test_peeler()
    
    open_PeelerWindow()
