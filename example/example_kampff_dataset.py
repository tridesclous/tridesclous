"""
The dataset is here http://www.kampff-lab.org/validating-electrodes/

You have to download 2015_09_03_Cell9.0 and put files
somewhere on your machine.

Them change working_dir = ... to the correct path

"""
from tridesclous import *
import pyqtgraph as pg

import os
from urllib.request import urlretrieve
import time

from matplotlib import pyplot


# !!!!!!!! change the working dir here
#working_dir = '/home/samuel/Documents/projet/DataSpikeSorting/kampff/'
working_dir = '/media/samuel/dataspikesorting/DataSpikeSortingHD2/kampff/'


dirname= working_dir+'tdc_2015_09_09_Pair_6_0'


 
def initialize_catalogueconstructor():
    #setup file source
    filename = working_dir+'2015_09_09_Pair_6_0/'+'amplifier2015-09-09T17_46_43.bin'
    print(filename)
    dataio = DataIO(dirname=dirname)
    dataio.set_data_source(type='RawData', filenames=[filename], dtype='uint16',
                                     total_channel=128, sample_rate=30000.)
    print(dataio)#check
    
    #setup probe file
    # Pierre Yger have done the PRB file in spyking-circus lets download
    # it with the build-in dataio.download_probe
    dataio.download_probe('kampff_128', origin='spyking-circus')
    
    #initiailize catalogue
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print(catalogueconstructor)


def preprocess_signals_and_peaks():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print(dataio)


    catalogueconstructor.set_preprocessor_params(chunksize=1024,
            memory_mode='memmap',
            
            #signal preprocessor
            #~ signalpreprocessor_engine='numpy',
            signalpreprocessor_engine='opencl',
            highpass_freq=300, 
            lowpass_freq=6000., 
            smooth_size=1,
            
            common_ref_removal=True,
            lostfront_chunksize=64,
            
            #peak detector
            #~ peakdetector_engine='numpy',
            peakdetector_engine='opencl',
            peak_sign='-', 
            relative_threshold=6,
            peak_span_ms=0.2,
            )
            
    t1 = time.perf_counter()
    catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    t1 = time.perf_counter()
    catalogueconstructor.run_signalprocessor(duration=60.)
    t2 = time.perf_counter()
    print('run_signalprocessor', t2-t1)
    
    print(catalogueconstructor)

def extract_waveforms_pca_cluster():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(mode='rand', n_left=-45, n_right=60,  nb_max=10000)
    #catalogueconstructor.extract_some_waveforms(mode='all', n_left=-45, n_right=60)
    t2 = time.perf_counter()
    print('extract_some_waveforms', t2-t1)

    t1 = time.perf_counter()
    catalogueconstructor.clean_waveforms(alien_value_threshold=100.)
    t2 = time.perf_counter()
    print('clean_waveforms', t2-t1)


    #extract_some_noise
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_noise(nb_snippet=400)
    t2 = time.perf_counter()
    print('extract_some_noise', t2-t1)


    t1 = time.perf_counter()
    catalogueconstructor.extract_some_features(method='peak_max')
    t2 = time.perf_counter()
    print('project', t2-t1)
    
    t1 = time.perf_counter()
    #catalogueconstructor.find_clusters(method='kmeans', n_clusters=100)#this is faster 
    catalogueconstructor.find_clusters(method='sawchaincut')
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)
    
    print(catalogueconstructor)
    
    
    catalogueconstructor.order_clusters(by='waveforms_rms')
    



def open_cataloguewindow():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()    


def clean_catalogue():
    # the catalogue need strong attention with teh catalogue windows.
    # here a dirty way a cleaning is to take on the first 20 bigger cells
    # the peeler will only detect them
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    #re order by rms
    cc.order_clusters(by='waveforms_rms')

    #re label >20 to trash (-1)
    mask = cc.all_peaks['cluster_label']>20
    cc.all_peaks['cluster_label'][mask] = -1
    cc.on_new_cluster()
    
    #save catalogue before peeler
    cc.make_catalogue_for_peeler()
    

def run_peeler():
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=1)
    
    print(dataio)
    peeler = Peeler(dataio)
    peeler.change_params(catalogue=initial_catalogue)
    
    t1 = time.perf_counter()
    peeler.run(duration=1.)
    t2 = time.perf_counter()
    print('peeler.run_loop', t2-t1)


def open_PeelerWindow():
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=1)

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()



if __name__ =='__main__':
    #~ initialize_catalogueconstructor()
    #~ preprocess_signals_and_peaks()
    #~ extract_waveforms_pca_cluster()
    #~ open_cataloguewindow()

    #~ clean_catalogue()
    #~ run_peeler()
    open_PeelerWindow()

    

    
