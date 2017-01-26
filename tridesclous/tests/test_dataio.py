import pytest
import os, tempfile, shutil
import numpy as np

from tridesclous import download_dataset
from tridesclous import DataIO
from tridesclous.dataio import RawDataSource,InMemoryDataSource




def test_InMemoryDataSource():
    nparrays = [np.random.randn(10000, 2) for i in range(5) ]
    datasource = InMemoryDataSource(nparrays=nparrays, sample_rate=5000.)
    assert datasource.total_channel == 2
    assert datasource.sample_rate == 5000.
    assert datasource.nb_segment==5
    assert datasource.get_segment_shape(0) == (10000, 2)
    data = datasource.get_signals_chunk(seg_num=0)
    assert data.shape==datasource.get_segment_shape(0)
    

def test_RawDataSource():
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    datasource = RawDataSource(filenames=filenames, **params)
    
    assert datasource.total_channel == 16
    assert datasource.sample_rate == 10000.
    assert datasource.nb_segment==3
    assert datasource.get_segment_shape(0) == (150000, 16)
    data = datasource.get_signals_chunk(seg_num=0)
    assert data.shape==datasource.get_segment_shape(0)

    

def test_DataIO():
    
    
    # initialze dataio
    if os.path.exists('test_DataIO'):
        shutil.rmtree('test_DataIO')
        
    dataio = DataIO(dirname='test_DataIO')
    print(dataio)


    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    
    #with geometry
    channels = list(range(14))
    channel_groups = {0:{'channels':range(14), 'geometry' : { c: [0, i] for i, c in enumerate(channels) }}}
    dataio.set_channel_groups(channel_groups)
    
    #with no geometry
    channel_groups = {0:{'channels':range(4)}}
    dataio.set_channel_groups(channel_groups)
    
    # add one group
    dataio.add_one_channel_group(channels=range(4,8), chan_grp=5)
    
    
    channel_groups = {0:{'channels':range(14)}}
    dataio.set_channel_groups(channel_groups)
    
    for seg_num in range(dataio.nb_segment):
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14
            #~ print(seg_num, i_stop, sigs_chunk.shape)
    
    
    #reopen existing
    dataio = DataIO(dirname='test_DataIO')
    print(dataio)
    
    #~ exit()
    
    for seg_num in range(dataio.nb_segment):
        #~ print('seg_num', seg_num)
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14



def test_DataIO_probes():
    # initialze dataio
    if os.path.exists('test_DataIO'):
        shutil.rmtree('test_DataIO')
        
    dataio = DataIO(dirname='test_DataIO')
    print(dataio)


    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames,  **params)
    
    probe_filename = 'A4x8-5mm-100-400-413-A32.prb'
    dataio.download_probe(probe_filename)
    dataio.download_probe('A4x8-5mm-100-400-413-A32')
    
    #~ print(dataio.channel_groups)
    #~ print(dataio.channels)
    #~ print(dataio.info['probe_filename'])
    
    assert dataio.nb_channel(0) == 8
    assert probe_filename == dataio.info['probe_filename']
    
    dataio = DataIO(dirname='test_DataIO')
    print(dataio)
    


def test_dataio_catalogue():
    if os.path.exists('test_DataIO'):
        shutil.rmtree('test_DataIO')
    
    dataio = DataIO(dirname='test_DataIO')
    
    catalogue = {}
    catalogue['chan_grp'] = 0
    catalogue['centers0'] = np.ones((300, 12, 50))
    
    catalogue['n_left'] = -15
    catalogue['params_signalpreprocessor'] = {'highpass_freq' : 300.}
    
    dataio.save_catalogue(catalogue, name='test')
    
    c2 = dataio.load_catalogue(name='test', chan_grp=0)
    print(c2)
    assert c2['n_left'] == -15
    assert np.all(c2['centers0']==1)
    assert catalogue['params_signalpreprocessor']['highpass_freq'] == 300.
    

    
    

    
    
if __name__=='__main__':
    #~ test_InMemoryDataSource()
    #~ test_RawDataSource()
    
    #~ test_DataIO()
    #~ test_DataIO_probes()
    test_dataio_catalogue()
    
    