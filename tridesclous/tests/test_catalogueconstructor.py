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
                                     total_channel=16, sample_rate=10000.)    
    dataio.set_channel_group(range(14))
    print(dataio.segments_path)
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    for memory_mode in ['ram', 'memmap']:
    #~ for memory_mode in ['memmap']:
    
        print()
        print(memory_mode)
        catalogueconstructor.initialize_signalprocessor_loop(chunksize=1024,
                memory_mode=memory_mode,
                
                #signal preprocessor
                highpass_freq=300,
                backward_chunksize=1280,
                #~ backward_chunksize=1024*2,
                
                #peak detector
                peakdetector_engine='peakdetector_numpy',
                peak_sign='-', relative_threshold=7, peak_span=0.0005,
                
                #waveformextractor
                #~ n_left=-20, n_right=30, 
                
                )
        t1 = time.perf_counter()
        catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.)
        t2 = time.perf_counter()
        print('estimate_signals_noise', t2-t1)
        
        t1 = time.perf_counter()
        for seg_num in range(dataio.nb_segment):
            #~ print('seg_num', seg_num)
            catalogueconstructor.run_signalprocessor_loop(seg_num=seg_num)
        t2 = time.perf_counter()
        print('run_signalprocessor_loop', t2-t1)

        t1 = time.perf_counter()
        catalogueconstructor.finalize_signalprocessor_loop()
        t2 = time.perf_counter()
        print('finalize_signalprocessor_loop', t2-t1)

        for seg_num in range(dataio.nb_segment):
            mask = catalogueconstructor.peak_segment==seg_num
            print('seg_num', seg_num, np.sum(mask))
        
        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=-20, n_right=30,  nb_max=5000)
        t2 = time.perf_counter()
        print('extract_some_waveforms', t2-t1)
        print(catalogueconstructor.peak_waveforms.shape)
        

        
        #~ # PCA
        #~ t1 = time.perf_counter()
        #~ catalogueconstructor.project(method='IncrementalPCA', n_components=7, batch_size=16384)
        #~ t2 = time.perf_counter()
        #~ print('project', t2-t1)
        
        #~ # cluster
        #~ t1 = time.perf_counter()
        #~ catalogueconstructor.find_clusters(method='kmeans', n_clusters=11)
        #~ t2 = time.perf_counter()
        #~ print('find_clusters', t2-t1)
        
        
        
        #plot
        #~ wf = catalogueconstructor.peak_waveforms
        #~ wf = wf.reshape(wf.shape[0], -1)
        
        #~ fig, ax = pyplot.subplots()
        #~ ax.plot(np.median(wf, axis=0), color='b')
        
        #~ pyplot.show()
    
    
def compare_nb_waveforms():
    filenames = ['Tem06c06.IOT', 'Tem06c07.IOT', 'Tem06c08.IOT']
    dataio = RawDataIO(dirname='test_catalogueconstructor')
    dataio.set_initial_signals(filenames=filenames, dtype='int16',
                                     total_channel=16, sample_rate=10000.)    
    dataio.set_channel_group(range(14))
    #~ print(dataio.segments_path)
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    memory_mode ='memmap'

    catalogueconstructor.initialize_signalprocessor_loop(chunksize=1024,
            memory_mode=memory_mode,
            
            #signal preprocessor
            highpass_freq=300,
            backward_chunksize=1280,
            
            #peak detector
            peakdetector_engine='peakdetector_numpy',
            peak_sign='-', relative_threshold=7, peak_span=0.0005,
            )
    
    t1 = time.perf_counter()
    catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    t1 = time.perf_counter()
    for seg_num in range(dataio.nb_segment):
        #~ print('seg_num', seg_num)
        catalogueconstructor.run_signalprocessor_loop(seg_num=seg_num)
    t2 = time.perf_counter()
    print('run_signalprocessor_loop', t2-t1)

    t1 = time.perf_counter()
    catalogueconstructor.finalize_signalprocessor_loop()
    t2 = time.perf_counter()
    print('finalize_signalprocessor_loop', t2-t1)

    for seg_num in range(dataio.nb_segment):
        mask = catalogueconstructor.peak_segment==seg_num
        print('seg_num', seg_num, np.sum(mask))
    
    
    fig, axs = pyplot.subplots(nrows=2)
    
    colors = ['r', 'g', 'b']
    for i, nb_max in enumerate([100, 1000, 10000]):
        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=-20, n_right=30,  nb_max=nb_max)
        t2 = time.perf_counter()
        print('extract_some_waveforms', nb_max,  t2-t1)
        print(catalogueconstructor.peak_waveforms.shape)
        wf = catalogueconstructor.peak_waveforms
        wf = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
        axs[0].plot(np.median(wf, axis=0), color=colors[i], label='nb_max {}'.format(nb_max))
        
        axs[1].plot(np.mean(wf, axis=0), color=colors[i], label='nb_max {}'.format(nb_max))
    
    axs[0].legend()
    axs[0].set_title('median')
    axs[1].set_title('mean')
    pyplot.show()        
    


    
if __name__ == '__main__':
    #~ test_catalogue_constructor()
    
    compare_nb_waveforms()