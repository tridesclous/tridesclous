"""
This script is equivalent of the jupyter notebook example_locust_dataset.ipynb
but in a standard python script.

"""

from tridesclous import *
import pyqtgraph as pg


from matplotlib import pyplot
import time


dirname = 'tridesclous_olfactory_bulb'

def initialize_catalogueconstructor():
    #download dataset
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
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

    #The dataset contains 16 channels but 14 and 15 are respiration and trigs.
    dataio.add_one_channel_group(channels=range(14), chan_grp=0)

    print(dataio)



def preprocess_signals_and_peaks():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    catalogueconstructor.set_preprocessor_params(chunksize=1024,
            
            #signal preprocessor
            highpass_freq=300, 
            common_ref_removal=True,
            backward_chunksize=1280,
            
            #peak detector
            peak_sign='-', 
            relative_threshold=4,
            peak_span=0.0005,
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
    print(catalogueconstructor)
    
    
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(n_left=-25, n_right=40,  nb_max=20000)
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
    catalogueconstructor.project(method='pca', n_components=5)
    t2 = time.perf_counter()
    print('project', t2-t1)
    print(catalogueconstructor)
    
    t1 = time.perf_counter()
    catalogueconstructor.find_clusters(method='kmeans', n_clusters=12)
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)
    print(catalogueconstructor)
    
    catalogueconstructor.trash_small_cluster(n=5)
    
    catalogueconstructor.order_clusters()




def open_cataloguewindow():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()    


def run_peeler():
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    peeler = Peeler(dataio)
    peeler.change_params(catalogue=initial_catalogue, n_peel_level=2)
    
    t1 = time.perf_counter()
    peeler.run()
    t2 = time.perf_counter()
    print('peeler.run', t2-t1)
    
    
    
def open_PeelerWindow():
    dataio = DataIO(dirname=dirname)
    print(dataio)
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
    run_peeler()
    open_PeelerWindow()
    
