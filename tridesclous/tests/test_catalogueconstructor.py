import numpy as np
import time

from tridesclous import get_dataset
from tridesclous.dataio import RawDataIO
from tridesclous.catalogueconstructor import CatalogueConstructor

from matplotlib import pyplot

def test_catalogue_constructor():
    filenames = ['Tem06c06.IOT', 'Tem06c07.IOT', 'Tem06c08.IOT']
    #~ filenames = ['Tem06c06.IOT']
    dataio = RawDataIO(dirname='test_catalogueconstructor')
    dataio.set_initial_signals(filenames=filenames, dtype='int16',
                                     nb_channel=16, sample_rate=10000.)    
    print(dataio.segments_path)
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    for memory_mode in ['ram', 'memmap']:
        print()
        print(memory_mode)
        catalogueconstructor.initialize(chunksize=1024,
                memory_mode=memory_mode,
                
                #signal preprocessor
                highpass_freq=300, backward_chunksize=1280,
                
                #peak detector
                peakdetector_engine='peakdetector_numpy',
                peak_sign='-', relative_threshold=5, peak_span=0.0005,
                
                #waveformextractor
                n_left=-20, n_right=30, 
                
                )
        t1 = time.perf_counter()
        catalogueconstructor.estimate_noise(seg_num=0, duration=10.)
        t2 = time.perf_counter()
        print('estimate_noise', t2-t1)
        
        t1 = time.perf_counter()
        for seg_num in range(dataio.nb_segment):
            #~ print('seg_num', seg_num)
            catalogueconstructor.loop_extract_waveforms(seg_num=seg_num)
        t2 = time.perf_counter()
        print('loop_extract_waveforms', t2-t1)

        t1 = time.perf_counter()
        catalogueconstructor.finalize_extract_waveforms()
        t2 = time.perf_counter()
        print('finalize_extract_waveforms', t2-t1)

        for seg_num in range(dataio.nb_segment):
            mask = catalogueconstructor.peak_segment==seg_num
            print('seg_num', seg_num, np.sum(mask))
        
        # PCA
        t1 = time.perf_counter()
        catalogueconstructor.project(method='IncrementalPCA', n_components=7, batch_size=16384)
        t2 = time.perf_counter()
        print('project', t2-t1)
        
        # cluster
        t1 = time.perf_counter()
        catalogueconstructor.find_clusters(method='kmeans', n_clusters=11)
        #~ catalogueconstructor.find_clusters(method='gmm', n_clusters=11)
        t2 = time.perf_counter()
        print('find_clusters', t2-t1)
        
        
        
        #plot
        #~ wf = catalogueconstructor.peak_waveforms
        #~ wf = wf.reshape(wf.shape[0], -1)
        
        #~ fig, ax = pyplot.subplots()
        #~ ax.plot(np.median(wf, axis=0), color='b')
        
        #~ pyplot.show()




    
if __name__ == '__main__':
    test_catalogue_constructor()
