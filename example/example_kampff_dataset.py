"""
The dataset is here http://www.kampff-lab.org/validating-electrodes/

"""
from tridesclous import *
import pyqtgraph as pg


from matplotlib import pyplot
import time


#~ p = '/media/samuel/SamCNRS/DataSpikeSorting/kampff/'
p = '/home/samuel/Documents/projet/DataSpikeSorting/kampff/'
dirname= p+'tdc_2014_11_25_Pair_3_0'
#~ dirname=p+'tdc_2015_09_03_Cell9.0'
#~ dirname=p+'tdc_2015_09_09_Pair_6_0'



def initialize_catalogueconstructor():
    filenames = p+'2014_11_25_Pair_3_0/'+'amplifier2014-11-25T23_00_08.bin'
    #~ filenames = p+'2015_09_09_Pair_6_0/'+'amplifier2015-09-09T17_46_43.bin'
    #~ filenames = p+'2015_09_03_Cell9.0/'+'amplifier2015-09-03T21_18_47.bin'
    
    dataio = DataIO(dirname=dirname)
    dataio.set_data_source(type='RawData', filenames=filenames, dtype='uint16',
                                     total_channel=32, sample_rate=30000.)    
    dataio.set_probe_file(p+'probe 32.prb')
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)


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
            relative_threshold=7,
            peak_span=0.0002,
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
    #~ catalogueconstructor.extract_some_waveforms(mode='all', n_left=-45, n_right=60)
    t2 = time.perf_counter()
    print('extract_some_waveforms', t2-t1)


    #extract_some_noise
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_noise(nb_snipet=400)
    t2 = time.perf_counter()
    print('extract_some_noise', t2-t1)


    t1 = time.perf_counter()
    catalogueconstructor.project(method='peak_max')
    t2 = time.perf_counter()
    print('project', t2-t1)
    
    t1 = time.perf_counter()
    catalogueconstructor.find_clusters(method='dirtycut')
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)
    
    print(catalogueconstructor)
    
    

    



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
    peeler.run(duration=60)
    t2 = time.perf_counter()
    print('peeler.run_loop', t2-t1)
    
def open_PeelerWindow():
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()



if __name__ =='__main__':
    #~ initialize_catalogueconstructor()
    preprocess_signals_and_peaks()
    extract_waveforms_pca_cluster()
    #~ open_cataloguewindow()
    #~ run_peeler()
    #~ open_PeelerWindow()

    
