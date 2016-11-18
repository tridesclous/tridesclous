import pytest
import os, tempfile, shutil
import numpy as np

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
    filenames = ['Tem06c06.IOT', 'Tem06c07.IOT', 'Tem06c08.IOT']
    datasource = RawDataSource(filenames=filenames, 
                    dtype='int16', total_channel=16, sample_rate=10000.)
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
    filenames = ['Tem06c06.IOT', 'Tem06c07.IOT', 'Tem06c08.IOT']
    params = dict(filenames=filenames, dtype='int16', total_channel=16, sample_rate=10000.)
    dataio.set_data_source(type='RawData', **params)
    dataio.set_channel_group(range(14))
    
    for seg_num in range(dataio.nb_segment):
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14
            #~ print(seg_num, i_stop, sigs_chunk.shape)
    
    #reopen existing
    dataio = DataIO(dirname='test_DataIO')
    #~ print(dataio.info)
    for seg_num in range(dataio.nb_segment):
        #~ print('seg_num', seg_num)
        for i_stop, sigs_chunk in dataio.iter_over_chunk(seg_num=seg_num, chunksize=1024):
            assert sigs_chunk.shape[0] == 1024
            assert sigs_chunk.shape[1] == 14
    
    
    
    
if __name__=='__main__':
    test_InMemoryDataSource()
    #~ test_RawDataSource()
    
    #~ test_DataIO()
    
    