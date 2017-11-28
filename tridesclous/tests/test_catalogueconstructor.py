import numpy as np
import time
import os
import shutil

from tridesclous import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.tools import median_mad

from matplotlib import pyplot

def test_catalogue_constructor():
    if os.path.exists('test_catalogueconstructor'):
        shutil.rmtree('test_catalogueconstructor')
        
    dataio = DataIO(dirname='test_catalogueconstructor')
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    #~ localdir, filenames, params = download_dataset(name='locust')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    
    channels=range(14)
    #~ channels=list(range(4))
    dataio.add_one_channel_group(channels=channels, chan_grp=0)
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    for memory_mode in ['ram', 'memmap']:
    #~ for memory_mode in ['memmap']:
    
        print()
        print(memory_mode)
        catalogueconstructor.set_preprocessor_params(chunksize=1024,
                memory_mode=memory_mode,
                
                #signal preprocessor
                highpass_freq=300,
                lowpass_freq=5000.,
                common_ref_removal=False,
                smooth_size=0,
                
                lostfront_chunksize = 128,
                
                #peak detector
                peakdetector_engine='numpy',
                peak_sign='-', relative_threshold=7, peak_span=0.0005,
                
                #waveformextractor
                #~ n_left=-20, n_right=30, 
                
                )
        t1 = time.perf_counter()
        catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.)
        t2 = time.perf_counter()
        print('estimate_signals_noise', t2-t1)
        
        #~ t1 = time.perf_counter()
        #~ for seg_num in range(dataio.nb_segment):
            #~ print('seg_num', seg_num)
            #~ catalogueconstructor.run_signalprocessor_loop_one_segment(seg_num=seg_num, duration=10.)
        catalogueconstructor.run_signalprocessor(duration=10., detect_peak=True)
        t2 = time.perf_counter()
        print('run_signalprocessor_loop', t2-t1)

        for seg_num in range(dataio.nb_segment):
            mask = catalogueconstructor.all_peaks['segment']==seg_num
            print('seg_num', seg_num, 'nb peak',  np.sum(mask))
        
        #redetect peak
        catalogueconstructor.re_detect_peak(peakdetector_engine='numpy',
                                            peak_sign='-', relative_threshold=5, peak_span=0.0002)
        for seg_num in range(dataio.nb_segment):
            mask = catalogueconstructor.all_peaks['segment']==seg_num
            print('seg_num', seg_num, 'nb peak',  np.sum(mask))

        
        
        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=-25, n_right=40, mode='rand', nb_max=5000)
        t2 = time.perf_counter()
        print('extract_some_waveforms rand', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)

        t1 = time.perf_counter()
        catalogueconstructor.find_good_limits()
        t2 = time.perf_counter()
        print('find_good_limits', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)

        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=None, n_right=None, mode='rand', nb_max=20000)
        t2 = time.perf_counter()
        print('extract_some_waveforms rand', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)

        #extract_some_noise
        t1 = time.perf_counter()
        catalogueconstructor.extract_some_noise(nb_snippet=400)
        t2 = time.perf_counter()
        print('extract_some_noise', t2-t1)
        
        # PCA
        t1 = time.perf_counter()
        catalogueconstructor.project(method='global_pca', n_components=7, batch_size=16384)
        t2 = time.perf_counter()
        print('project pca', t2-t1)

        # peak_max
        #~ t1 = time.perf_counter()
        #~ catalogueconstructor.project(method='peak_max')
        #~ t2 = time.perf_counter()
        #~ print('project peak_max', t2-t1)
        #~ print(catalogueconstructor.some_features.shape)

        #~ t1 = time.perf_counter()
        #~ catalogueconstructor.extract_some_waveforms(index=np.arange(1000))
        #~ t2 = time.perf_counter()
        #~ print('extract_some_waveforms others', t2-t1)
        #~ print(catalogueconstructor.some_waveforms.shape)

        
        # cluster
        t1 = time.perf_counter()
        catalogueconstructor.find_clusters(method='kmeans', n_clusters=11)
        t2 = time.perf_counter()
        print('find_clusters', t2-t1)
        
        
        # similarity
        #~ catalogueconstructor.compute_centroid()
        #~ similarity, cluster_labels = catalogueconstructor.compute_similarity()
        #~ print(cluster_labels)
        #~ fig, ax = pyplot.subplots()
        #~ ax.matshow(similarity)
        #~ pyplot.show()
        
        #plot
        #~ wf = catalogueconstructor.some_waveforms
        #~ wf = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
        
        #~ fig, ax = pyplot.subplots()
        #~ ax.plot(np.median(wf, axis=0), color='b')
        
        #~ pyplot.show()

#~ def show_similarity():
    
    
    
def compare_nb_waveforms():
    if os.path.exists('test_catalogueconstructor'):
        shutil.rmtree('test_catalogueconstructor')
        
    dataio = DataIO(dirname='test_catalogueconstructor')
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    dataio.add_one_channel_group(channels=range(14), chan_grp=0)


    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    catalogueconstructor.set_preprocessor_params(chunksize=1024,
            
                                #signal preprocessor
                                highpass_freq=300.,
                                lowpass_freq=5000.,
                                lostfront_chunksize=128,
                                
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
    
    fig, axs = pyplot.subplots(nrows=2)
    
    colors = ['r', 'g', 'b']
    for i, nb_max in enumerate([100, 1000, 10000]):
        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=-20, n_right=30,  nb_max=nb_max)
        t2 = time.perf_counter()
        print('extract_some_waveforms', nb_max,  t2-t1)
        print(catalogueconstructor.some_waveforms.shape)
        wf = catalogueconstructor.some_waveforms
        wf = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
        axs[0].plot(np.median(wf, axis=0), color=colors[i], label='nb_max {}'.format(nb_max))
        
        axs[1].plot(np.mean(wf, axis=0), color=colors[i], label='nb_max {}'.format(nb_max))
    
    axs[0].legend()
    axs[0].set_title('median')
    axs[1].set_title('mean')
    pyplot.show()        
    

def test_make_catalogue():
    if os.path.exists('test_catalogueconstructor'):
        shutil.rmtree('test_catalogueconstructor')
        
    dataio = DataIO(dirname='test_catalogueconstructor')
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    #~ dataio.add_one_channel_group(channels=range(14), chan_grp=0)
    dataio.add_one_channel_group(channels=[5, 6, 7, 8, 9], chan_grp=0)
    

    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    catalogueconstructor.set_preprocessor_params(chunksize=1024,
            
                                    #signal preprocessor
                                    highpass_freq=300.,
                                    lowpass_freq=5000.,
                                    lostfront_chunksize=128,
                                    
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
    catalogueconstructor.extract_some_waveforms(n_left=-12, n_right=15,  nb_max=10000)
    t2 = time.perf_counter()
    print('extract_some_waveforms', t2-t1)


    # PCA
    t1 = time.perf_counter()
    catalogueconstructor.project(method='pca', n_components=12, batch_size=16384)
    t2 = time.perf_counter()
    print('project', t2-t1)
    
    # cluster
    t1 = time.perf_counter()
    catalogueconstructor.find_clusters(method='kmeans', n_clusters=13)
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)
    
    # trash_small_cluster
    catalogueconstructor.trash_small_cluster()
    
    catalogueconstructor.make_catalogue()
    


def test_ratio_amplitude():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    pairs = catalogueconstructor.detect_similar_waveform_ratio(0.5)
    print(pairs)


def test_create_copy_catalogue_constructor():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    catalogueconstructor.create_copy()


    
if __name__ == '__main__':
    #~ test_catalogue_constructor()
    
    #~ compare_nb_waveforms()
    
    #~ test_make_catalogue()
    #~ test_ratio_amplitude()
    
    test_create_copy_catalogue_constructor()


