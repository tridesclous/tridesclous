"""
This script is equivalent of the jupyter notebook example_locust_dataset.ipynb
but in a standard python script.

"""

from tridesclous import *
import pyqtgraph as pg


from matplotlib import pyplot
import time


dirname = 'tridesclous_locust'

def initialize_catalogueconstructor():
    #download dataset
    localdir, filenames, params = download_dataset(name='locust')
    print(filenames)
    print(params)

    print()
    #create a DataIO
    import os, shutil
    
    if os.path.exists(dirname):
        #remove is already exists
        shutil.rmtree(dirname)    
    dataio = DataIO(dirname=dirname)

    # feed DataIO
    dataio.set_data_source(type='RawData', filenames=filenames, **params)

    #The dataset contains 4 channels : we use them all
    dataio.add_one_channel_group(channels=[0, 1, 2, 3])

    print(dataio)



def preprocess_signals_and_peaks():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    catalogueconstructor.set_global_params(chunksize=1024,mode='dense')
    
    catalogueconstructor.set_preprocessor_params(
            highpass_freq=None,
            lowpass_freq=None,
            smooth_size=1,
            common_ref_removal=False,
            lostfront_chunksize=64)
    
    catalogueconstructor.set_peak_detector_params(
            engine='numpy',
            peak_sign='-', 
            relative_threshold=4,
            peak_span_ms=0.9)
    
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
    print(catalogueconstructor)
    
    
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(n_left=-35, n_right=60,  nb_max=10000)
    t2 = time.perf_counter()
    print('extract_some_waveforms', t2-t1)
    #~ print(catalogueconstructor.some_waveforms.shape)
    print(catalogueconstructor)
    
    t1 = time.perf_counter()
    n_left, n_right = catalogueconstructor.find_good_limits(mad_threshold = 1.1,)
    t2 = time.perf_counter()
    #~ print('n_left', n_left, 'n_right', n_right)
    #~ print(catalogueconstructor.some_waveforms.shape)
    print(catalogueconstructor)


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
    catalogueconstructor.project(method='global_pca', n_components=7)
    t2 = time.perf_counter()
    print('project', t2-t1)
    print(catalogueconstructor)
    
    t1 = time.perf_counter()
    #~ catalogueconstructor.find_clusters(method='kmeans', n_clusters=10)
    catalogueconstructor.find_clusters(method='sawchaincut')
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)
    print(catalogueconstructor)


    


def open_cataloguewindow():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print(catalogueconstructor)
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()    


def run_peeler():
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    peeler = Peeler(dataio)
    peeler.change_params(catalogue=initial_catalogue)
    
    t1 = time.perf_counter()
    peeler.run()
    t2 = time.perf_counter()
    print('peeler.run', t2-t1)
    
    
    
def open_PeelerWindow():
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()



if __name__ =='__main__':
    #~ initialize_catalogueconstructor()
    #~ preprocess_signals_and_peaks()
    #~ extract_waveforms_pca_cluster()
    #~ open_cataloguewindow()
    #~ run_peeler()
    open_PeelerWindow()
    
