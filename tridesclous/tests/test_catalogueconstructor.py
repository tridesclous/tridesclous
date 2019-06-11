import numpy as np
import time
import os
import shutil

from tridesclous import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.tools import median_mad

from matplotlib import pyplot as plt

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
                peak_sign='-', relative_threshold=7, peak_span_ms=0.5,
                
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
                                            peak_sign='-', relative_threshold=5, peak_span_ms=0.2)
        for seg_num in range(dataio.nb_segment):
            mask = catalogueconstructor.all_peaks['segment']==seg_num
            print('seg_num', seg_num, 'nb peak',  np.sum(mask))

        
        
        t1 = time.perf_counter()
        #~ catalogueconstructor.extract_some_waveforms(n_left=-25, n_right=40, mode='rand', nb_max=5000)
        catalogueconstructor.extract_some_waveforms(wf_left_ms=-2.5, wf_right_ms=4.0, mode='rand', nb_max=5000)
        t2 = time.perf_counter()
        print('extract_some_waveforms rand', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)

        t1 = time.perf_counter()
        catalogueconstructor.find_good_limits()
        t2 = time.perf_counter()
        print('find_good_limits', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)
        

        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=None, n_right=None, mode='rand', nb_max=5000)
        t2 = time.perf_counter()
        print('extract_some_waveforms rand', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)
        
        t1 = time.perf_counter()
        catalogueconstructor.clean_waveforms(alien_value_threshold=60.)
        t2 = time.perf_counter()
        print('clean_waveforms', t2-t1)
        
        print(catalogueconstructor)
        

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
        
        print(catalogueconstructor)
        
        # similarity
        #~ catalogueconstructor.compute_centroid()
        #~ similarity, cluster_labels = catalogueconstructor.compute_similarity()
        #~ print(cluster_labels)
        #~ fig, ax = plt.subplots()
        #~ ax.matshow(similarity)
        #~ plt.show()
        
        #plot
        #~ wf = catalogueconstructor.some_waveforms
        #~ wf = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
        
        #~ fig, ax = plt.subplots()
        #~ ax.plot(np.median(wf, axis=0), color='b')
        
        #~ plt.show()

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
                                peak_sign='-', relative_threshold=7, peak_span_ms=0.5,
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
    
    fig, axs = plt.subplots(nrows=2)
    
    colors = ['r', 'g', 'b']
    for i, nb_max in enumerate([100, 1000, 10000]):
        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(wf_left_ms=-2.0, wf_right_ms=3.0,  nb_max=nb_max)
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
    plt.show()        
    

def test_make_catalogue():
    dataio = DataIO(dirname='test_catalogueconstructor')

    cc = CatalogueConstructor(dataio=dataio)

    #~ cc.make_catalogue()
    cc.make_catalogue_for_peeler()
    
    #~ for i, k in cc.catalogue['label_to_index'].items():
    
        #~ fig, ax = plt.subplots()
        #~ ax.plot(cc.catalogue['centers0'][i, :, :].T.flatten(), color='b')
        #~ ax.plot(cc.catalogue['centers1'][i, :, :].T.flatten(), color='g')
        #~ ax.plot(cc.catalogue['centers2'][i, :, :].T.flatten(), color='r')
        
    
    #~ plt.show()
    


def test_ratio_amplitude():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    pairs = catalogueconstructor.detect_similar_waveform_ratio(0.5)
    print(pairs)


def test_create_savepoint_catalogue_constructor():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    copy_path = catalogueconstructor.create_savepoint()
    print(copy_path)


    
if __name__ == '__main__':
    test_catalogue_constructor()
    
    #~ compare_nb_waveforms()
    
    #~ test_make_catalogue()
    #~ test_ratio_amplitude()
    
    #~ test_create_savepoint_catalogue_constructor()


