from tridesclous import *
from tridesclous.online import *
from tridesclous.gui import QT


import  pyqtgraph as pg

import pyacq
from pyacq.viewers import QOscilloscope, QTimeFreq


import numpy as np
import time
import os
import shutil




def setup_catalogue():
    if os.path.exists('test_onlinepeeler'):
        shutil.rmtree('test_onlinepeeler')
    
    dataio = DataIO(dirname='test_onlinepeeler')
    
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    filenames = filenames[:1] #only first file
    #~ print(params)
    #~ exit()
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    channel_group = {0:{'channels':[5, 6, 7, 8]}}
    dataio.set_channel_groups(channel_group)
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)

    catalogueconstructor.set_preprocessor_params(chunksize=1024,
            memory_mode='memmap',
            
            #signal preprocessor
            highpass_freq=300,
            lostfront_chunksize=64,
            
            #peak detector
            peakdetector_engine='numpy',
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

    
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(n_left=-25, n_right=35,  nb_max=10000)
    t2 = time.perf_counter()
    print('extract_some_waveforms', t2-t1)
    print(catalogueconstructor)
    
    t1 = time.perf_counter()
    catalogueconstructor.clean_waveforms(alien_value_threshold=100.)
    t2 = time.perf_counter()
    print('clean_waveforms', t2-t1)


    # PCA
    t1 = time.perf_counter()
    catalogueconstructor.project(method='neighborhood_pca', n_components_by_neighborhood=3)
    t2 = time.perf_counter()
    print('project', t2-t1)
    
    # cluster
    t1 = time.perf_counter()
    catalogueconstructor.find_clusters(method='kmeans', n_clusters=13)
    t2 = time.perf_counter()
    print('find_clusters', t2-t1)
    
    # trash_small_cluster
    catalogueconstructor.trash_small_cluster()


    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    app.exec_()

    


def test_OnlinePeeler():
    dataio = DataIO(dirname='test_onlinepeeler')

    catalogue = dataio.load_catalogue(chan_grp=0)
    
    
    
    sigs = dataio.datasource.array_sources[0]
    
    sigs = sigs.astype('float32')
    sample_rate = dataio.sample_rate
    in_group_channels = dataio.channel_groups[0]['channels']
    #~ print(channel_group)
    
    chunksize = 1024
    
    
    # Device node
    #~ man = create_manager(auto_close_at_exit=True)
    #~ ng0 = man.create_nodegroup()
    ng0 = None
    dev = make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup=ng0, chunksize=chunksize)
    

    
    app = pg.mkQApp()
    
    dev.start()
    
    # Node QOscilloscope
    #~ oscope = QOscilloscope()
    #~ oscope.configure(with_user_dialog=True)
    #~ oscope.input.connect(dev.output)
    #~ oscope.initialize()
    #~ oscope.show()
    #~ oscope.start()
    #~ oscope.params['decimation_method'] = 'min_max'
    #~ oscope.params['mode'] = 'scan'    

    # Node Peeler
    peeler = OnlinePeeler()
    peeler.configure(catalogue=catalogue, in_group_channels=in_group_channels, chunksize=chunksize)
    peeler.input.connect(dev.output)
    stream_params = dict(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
    peeler.outputs['signals'].configure(**stream_params)
    peeler.outputs['spikes'].configure(**stream_params)
    peeler.initialize()
    peeler.start()
    
    # Node traceviewer
    #~ tviewer = OnlineTraceViewer()
    #~ tviewer.configure(peak_buffer_size = 1000, catalogue=lighter_catalogue(catalogue))
    #~ tviewer.inputs['signals'].connect(peeler.outputs['signals'])
    #~ tviewer.inputs['spikes'].connect(peeler.outputs['spikes'])
    #~ tviewer.initialize()
    #~ tviewer.show()
    #~ tviewer.start()
    #~ tviewer.params['xsize'] = 3.
    #~ tviewer.params['decimation_method'] = 'min_max'
    #~ tviewer.params['mode'] = 'scan'

    
    # waveform histogram viewer
    wviewer = OnlineWaveformHistViewer()
    wviewer.configure(peak_buffer_size = 1000, catalogue=catalogue)
    wviewer.inputs['signals'].connect(peeler.outputs['signals'])
    wviewer.inputs['spikes'].connect(peeler.outputs['spikes'])
    wviewer.initialize()
    wviewer.show()
    wviewer.start()    
    
    
    def terminate():
        dev.stop()
        oscope.stop()
        peeler.stop()
        tviewer.stop()
        app.quit()
    
    app.exec_()
    
    

def test_OnlinePeeler_no_catalogue():
    

    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    filename = filenames[0] #only first file
    #~ print(params)
    sigs = np.fromfile(filename, dtype=params['dtype']).reshape(-1, params['total_channel'])
    sigs = sigs.astype('float32')
    sample_rate = params['sample_rate']
    #~ dataio.set_data_source(type='RawData', filenames=filenames, **params)
    #~ channel_group = {0:{'channels':[5, 6, 7, 8]}}
    #~ dataio.set_channel_groups(channel_group)
    #~ print(filenames)
    #~ print(sigs.shape)
    #~ exit()
    
    
    channel_indexes = [5,6,7,8]
    
    chunksize = 1024
    
    
    #case 1 before medians estimation
    #~ empty_catalogue = make_empty_catalogue(
                #~ channel_indexes=channel_indexes,
                #~ n_left=-20, n_right=40, internal_dtype='float32',
                #~ peak_detector_params={'relative_threshold': np.inf},
                #~ signals_medians = None,
                #~ signals_mads = None,
        #~ )
    #case 2 after medians estimation
    preprocessor_params = {}
    signals_medians, signals_mads = estimate_medians_mads_after_preprocesing(sigs[:, channel_indexes], sample_rate,
                        preprocessor_params=preprocessor_params)
    empty_catalogue = make_empty_catalogue(
                channel_indexes=channel_indexes,
                n_left=-20, n_right=40, internal_dtype='float32',                
                preprocessor_params=preprocessor_params,
                peak_detector_params={'relative_threshold': 10},
                clean_waveforms_params={},
                
                signals_medians = signals_medians,
                signals_mads = signals_mads,
        
        )
    
    
    print(empty_catalogue)
    #~ print(empty_catalogue['params_signalpreprocessor'])
    #~ exit()
    
    
    
    
    
    #~ in_group_channels = dataio.channel_groups[0]['channels']
    #~ print(channel_group)
    
    chunksize = 1024
    
    
    # Device node
    #~ man = create_manager(auto_close_at_exit=True)
    #~ ng0 = man.create_nodegroup()
    ng0 = None
    dev = make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup=ng0, chunksize=chunksize)
    
    #~ exit()
    
    app = pg.mkQApp()
    
    dev.start()
    
    # Node QOscilloscope
    oscope = QOscilloscope()
    oscope.configure(with_user_dialog=True)
    oscope.input.connect(dev.output)
    oscope.initialize()
    oscope.show()
    oscope.start()
    oscope.params['decimation_method'] = 'min_max'
    oscope.params['mode'] = 'scan'
    oscope.params['scale_mode'] = 'by_channel'

    # Node Peeler
    peeler = OnlinePeeler()
    peeler.configure(catalogue=empty_catalogue, in_group_channels=channel_indexes, chunksize=chunksize)
    peeler.input.connect(dev.output)
    stream_params = dict(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
    peeler.outputs['signals'].configure(**stream_params)
    peeler.outputs['spikes'].configure(**stream_params)
    peeler.initialize()
    peeler.start()
    
    # Node traceviewer
    tviewer = OnlineTraceViewer()
    tviewer.configure(peak_buffer_size = 1000, catalogue=empty_catalogue)
    tviewer.inputs['signals'].connect(peeler.outputs['signals'])
    tviewer.inputs['spikes'].connect(peeler.outputs['spikes'])
    tviewer.initialize()
    tviewer.show()
    tviewer.params['xsize'] = 3.
    tviewer.params['decimation_method'] = 'min_max'
    tviewer.params['mode'] = 'scan'
    tviewer.params['scale_mode'] = 'same_for_all'
    tviewer.start()
    
    # waveform histogram viewer
    wviewer = OnlineWaveformHistViewer()
    wviewer.configure(peak_buffer_size = 1000, catalogue=empty_catalogue)
    wviewer.inputs['signals'].connect(peeler.outputs['signals'])
    wviewer.inputs['spikes'].connect(peeler.outputs['spikes'])
    wviewer.initialize()
    wviewer.show()
    wviewer.start()        

    
    def auto_scale():
        oscope.auto_scale()
        tviewer.auto_scale()
    
    def terminate():
        dev.stop()
        oscope.stop()
        peeler.stop()
        tviewer.stop()
        app.quit()
    
    timer = QT.QTimer(singleShot=True, interval=500)
    timer.timeout.connect(auto_scale)
    timer.start()
    
    app.exec_()
    

    
    
    
if __name__ =='__main__':
    #~ setup_catalogue()
    
    test_OnlinePeeler()
    
    #~ test_OnlinePeeler_no_catalogue()

