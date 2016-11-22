from tridesclous import *
import pyqtgraph as pg


from matplotlib import pyplot
import time


def initialize_catalogueconstructor():
    #~ filenames = ['kampff/2015_09_09_Pair_6_0/amplifier2015-09-09T17_46_43.bin']
    filenames = ['/home/samuel/Documents/projet/Data SpikeSorting/kampff/2015_09_09_Pair_6_0/amplifier2015-09-09T17_46_43.bin']
    dataio = DataIO(dirname='tridesclous_kampff')
    dataio.set_data_source(type='RawData', filenames=filenames, dtype='uint16',
                                     total_channel=128, sample_rate=30000.)    
    dataio.set_channel_group(range(50,90))
    print(dataio)
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)

def preprocess_signals_and_peaks():
    dataio = DataIO(dirname='tridesclous_kampff')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print('shape', dataio.get_segment_shape(seg_num=0))
    print('duration', dataio.get_segment_shape(seg_num=0)[0]/dataio.sample_rate)


    catalogueconstructor.initialize_signalprocessor_loop(chunksize=1024,
            memory_mode='memmap',
            
            #signal preprocessor
            highpass_freq=300, 
            common_ref_removal=True,
            backward_chunksize=1280,
            
            #peak detector
            peakdetector_engine='peakdetector_numpy',
            peak_sign='-', 
            relative_threshold=7,
            peak_span=0.0005,
            )
            
    t1 = time.perf_counter()
    catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    t1 = time.perf_counter()
    catalogueconstructor.run_signalprocessor_loop(seg_num=0, duration=60.)
    t2 = time.perf_counter()
    print('run_signalprocessor_loop', t2-t1)

    t1 = time.perf_counter()
    catalogueconstructor.finalize_signalprocessor_loop()
    t2 = time.perf_counter()
    print('finalize_signalprocessor_loop', t2-t1)
    
    print('nb_peak', catalogueconstructor.nb_peak)

def extract_waveforms_pca_cluster():
    dataio = DataIO(dirname='tridesclous_kampff')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print('nb_peak', catalogueconstructor.nb_peak)
    #~ exit()
    
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(n_left=-20, n_right=30,  nb_max=5000)
    t2 = time.perf_counter()
    print('extract_some_waveforms', t2-t1)
    print(catalogueconstructor.peak_waveforms.shape)
    
    t1 = time.perf_counter()
    catalogueconstructor.project(method='pca', n_components=20)
    t2 = time.perf_counter()
    print('project', t2-t1)
    
    t1 = time.perf_counter()
    catalogueconstructor.find_clusters(method='gmm', n_clusters=12)
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)




def open_cataloguewindow():
    dataio = DataIO(dirname='tridesclous_kampff')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()    


def run_peeler():
    dataio = DataIO(dirname='tridesclous_kampff')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    initial_catalogue = catalogueconstructor.load_catalogue()

    peeler = Peeler(dataio)
    peeler.change_params(catalogue=initial_catalogue, n_peel_level=2)
    
    t1 = time.perf_counter()
    peeler.run()
    t2 = time.perf_counter()
    print('peeler.run_loop', t2-t1)
    
def open_PeelerWindow():
    dataio = DataIO(dirname='tridesclous_kampff')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    initial_catalogue = catalogueconstructor.load_catalogue()

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

    
