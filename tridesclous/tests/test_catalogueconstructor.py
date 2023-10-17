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
    
    cc = CatalogueConstructor(dataio=dataio)
    
    for memory_mode in ['ram', 'memmap']:
        for mode in ['dense', 'sparse']:
        
            print('*'*5)
            print('memory_mode', memory_mode, 'mode', mode)

            if mode == 'dense':
                peak_engine = 'numpy'
                peak_method = 'global'
                adjacency_radius_um = None
            elif mode == 'sparse':
                peak_engine = 'numpy'
                peak_method = 'geometrical'
                adjacency_radius_um = 450.
            
            cc.set_global_params(chunksize=1024,
                                            memory_mode=memory_mode,
                                            mode=mode,
                                            n_jobs=1,
                                            #~ adjacency_radius_um=adjacency_radius_um,
                                            )
            
            cc.set_preprocessor_params(
                    #signal preprocessor
                    highpass_freq=300,
                    lowpass_freq=5000.,
                    common_ref_removal=False,
                    smooth_size=0,
                    pad_width = None)
            
            
            cc.set_peak_detector_params(
                    #peak detector
                    method=peak_method, engine=peak_engine,
                    peak_sign='-', relative_threshold=7, peak_span_ms=0.5,
                    adjacency_radius_um=adjacency_radius_um,
                    )
            
            t1 = time.perf_counter()
            cc.estimate_signals_noise(seg_num=0, duration=10.)
            t2 = time.perf_counter()
            print('estimate_signals_noise', t2-t1)
            
            t1 = time.perf_counter()
            cc.run_signalprocessor(duration=10., detect_peak=True)
            t2 = time.perf_counter()
            print('run_signalprocessor_loop', t2-t1)

            for seg_num in range(dataio.nb_segment):
                mask = cc.all_peaks['segment']==seg_num
                print('seg_num', seg_num, 'nb peak',  np.sum(mask))
            
            # redetect peak
            cc.re_detect_peak(method=peak_method, engine=peak_engine,
                                                peak_sign='-', relative_threshold=5, peak_span_ms=0.7,
                                                adjacency_radius_um=adjacency_radius_um)
            for seg_num in range(dataio.nb_segment):
                mask = cc.all_peaks['segment']==seg_num
                print('seg_num', seg_num, 'nb peak',  np.sum(mask))

            cc.set_waveform_extractor_params(n_left=-25, n_right=40)

            t1 = time.perf_counter()
            cc.clean_peaks(alien_value_threshold=100, mode='extremum_amplitude')
            t2 = time.perf_counter()
            print('clean_peaks extremum_amplitude', t2-t1)

            t1 = time.perf_counter()
            cc.clean_peaks(alien_value_threshold=100, mode='full_waveform')
            t2 = time.perf_counter()
            print('clean_peaks full_waveforms', t2-t1)
            
            t1 = time.perf_counter()
            cc.sample_some_peaks( mode='rand', nb_max=5000)
            t2 = time.perf_counter()
            print('sample_some_peaks', t2-t1)
            
            
            
            print(cc)
            
            #extract_some_noise
            t1 = time.perf_counter()
            cc.extract_some_noise(nb_snippet=400)
            t2 = time.perf_counter()
            print('extract_some_noise', t2-t1)
            
            
            if mode == 'dense':
                # PCA
                t1 = time.perf_counter()
                cc.extract_some_features(method='global_pca', n_components=12)
                t2 = time.perf_counter()
                print('project pca', t2-t1)

                # cluster
                t1 = time.perf_counter()
                cc.find_clusters(method='kmeans', n_clusters=11)
                t2 = time.perf_counter()
                print('find_clusters', t2-t1)
            
            elif mode == 'sparse':
            
                # PCA
                t1 = time.perf_counter()
                cc.extract_some_features(method='pca_by_channel', n_components_by_channel=3)
                t2 = time.perf_counter()
                print('project pca', t2-t1)

                # cluster
                t1 = time.perf_counter()
                cc.find_clusters(method='pruningshears')
                t2 = time.perf_counter()
                print('find_clusters', t2-t1)
            
            print(cc)
        
        
        t1 = time.perf_counter()
        cc.cache_some_waveforms()
        t2 = time.perf_counter()
        print('cache_some_waveforms', t2-t1)
        
        t1 = time.perf_counter()
        cc.auto_split_cluster()
        t2 = time.perf_counter()
        print('auto_split_cluster', t2-t1)

        t1 = time.perf_counter()
        cc.trash_not_aligned()
        t2 = time.perf_counter()
        print('trash_not_aligned', t2-t1)

        t1 = time.perf_counter()
        cc.auto_merge_cluster()
        t2 = time.perf_counter()
        print('auto_merge_cluster', t2-t1)

        t1 = time.perf_counter()
        cc.trash_low_extremum()
        t2 = time.perf_counter()
        print('trash_low_extremum', t2-t1)


        t1 = time.perf_counter()
        cc.trash_small_cluster()
        t2 = time.perf_counter()
        print('trash_small_cluster', t2-t1)
        
        
            # similarity
            #~ cc.compute_centroid()
            #~ similarity, cluster_labels = cc.compute_similarity()
            #~ print(cluster_labels)
            #~ fig, ax = plt.subplots()
            #~ ax.matshow(similarity)
            #~ plt.show()
            
            #plot
            #~ wf = cc.some_waveforms
            #~ wf = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(np.median(wf, axis=0), color='b')
            
            #~ plt.show()

    
    
    
def compare_nb_waveforms():
    if os.path.exists('test_catalogueconstructor'):
        shutil.rmtree('test_catalogueconstructor')
        
    dataio = DataIO(dirname='test_catalogueconstructor')
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    dataio.add_one_channel_group(channels=range(14), chan_grp=0)


    cc = CatalogueConstructor(dataio=dataio)

    cc.set_global_params(chunksize=1024,
                                    memory_mode='ram',
                                    mode='dense',
                                    n_jobs=1,
                                    #~ adjacency_radius_um=None,
                                    )
    
    cc.set_preprocessor_params(
            #signal preprocessor
            highpass_freq=300,
            lowpass_freq=5000.,
            common_ref_removal=False,
            smooth_size=0,
            pad_width = None)
    
    
    cc.set_peak_detector_params(
            #peak detector
            method='global', engine='numpy',
            peak_sign='-', relative_threshold=7, peak_span_ms=0.5,
            #~ adjacency_radius_um=None,
            )
    
    t1 = time.perf_counter()
    cc.estimate_signals_noise(seg_num=0, duration=10.)
    t2 = time.perf_counter()
    print('estimate_signals_noise', t2-t1)
    
    t1 = time.perf_counter()
    cc.run_signalprocessor()
    t2 = time.perf_counter()
    print('run_signalprocessor', t2-t1)
    
    print(cc)
    
    fig, axs = plt.subplots(nrows=2)


    cc.set_waveform_extractor_params(wf_left_ms=-2.0, wf_right_ms=3.0)
    
    t1 = time.perf_counter()
    cc.sample_some_peaks( mode='rand', nb_max=5000)
    t2 = time.perf_counter()
    print('sample_some_peaks', t2-t1)
    
    colors = ['r', 'g', 'b', 'y']
    for i, nb_max in enumerate([100, 500, 1000, 2000]):
        
        cc.sample_some_peaks( mode='rand', nb_max=nb_max)
        #~ catalogueconstructor.extract_some_waveforms(wf_left_ms=-2.0, wf_right_ms=3.0,  nb_max=nb_max)
        #~ print(catalogueconstructor.some_waveforms.shape)
        t1 = time.perf_counter()
        wf = cc.get_some_waveforms()
        t2 = time.perf_counter()
        print('get_some_waveforms', nb_max,  t2-t1)
        
        #~ wf = catalogueconstructor.some_waveforms
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


def debug_interp_centers0():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    catalogue = catalogueconstructor.make_catalogue()
    centers = catalogue['centers0']
    interp_centers = catalogue['interp_centers0']
    subsample_ratio = catalogue['subsample_ratio']
    
    #~ fig, ax = plt.subplots()
    #~ ax.plot(centers[0])
    #~ ax.plot(np.arange(interp_centers[0].shape[0])/subsample_ratio - 0.5, interp_centers[0])
    
    #~ plt.show()

def test_feature_with_lda_selection():
    dataio = DataIO(dirname='test_catalogueconstructor')
    cc = CatalogueConstructor(dataio=dataio)
    print(cc)
    
    selection = np.isin(cc.all_peaks['cluster_label'], [1, 2, 3, 4])
    print(np.sum(selection), '/', cc.all_peaks.size)
    
    cc.extract_some_features(method='global_lda',  selection=selection)
    print(cc.some_features.shape)
    print(cc.some_features)
    
if __name__ == '__main__':
    test_catalogue_constructor()
    
    ## compare_nb_waveforms()
    
    #~ test_make_catalogue()
    #~ test_ratio_amplitude()
    
    #~ test_create_savepoint_catalogue_constructor()
    
    #~ debug_interp_centers0()
    
    #~ test_feature_with_lda_selection()


