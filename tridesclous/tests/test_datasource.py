import pytest
import os, tempfile, shutil
import numpy as np

from tridesclous import download_dataset
#~ from tridesclous import DataIO
from tridesclous.datasource import InMemoryDataSource, NEO_VERSION




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
    if NEO_VERSION>='0.6':
        return
    from tridesclous.datasource import RawDataSource
    
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    datasource = RawDataSource(filenames=filenames, **params)
    
    assert datasource.total_channel == 16
    assert datasource.sample_rate == 10000.
    assert datasource.nb_segment==3
    assert datasource.get_segment_shape(0) == (150000, 16)
    data = datasource.get_signals_chunk(seg_num=0)
    assert data.shape==datasource.get_segment_shape(0)


def test_NeoRawIOAggregator():
    
    #~ if NEO_VERSION is None or NEO_VERSION<'0.6':
        #~ return
    
    from tridesclous import BlackrockDataSource
    filename = '/home/samuel/Documents/files_for_testing_neo/blackrock/FileSpec2.3001.ns5'
    
    datasource = BlackrockDataSource(filenames=[filename, filename])
    print(datasource.total_channel)
    print(datasource.nb_segment)
    print(datasource.get_segment_shape(0), datasource.get_segment_shape(1))
    assert datasource.get_segment_shape(0)==datasource.get_segment_shape(1)
    print(datasource.get_channel_names())
    data = datasource.get_signals_chunk(seg_num=0)
    assert data.shape==datasource.get_segment_shape(0)

    



if __name__=='__main__':
    #~ test_InMemoryDataSource()
    #~ test_RawDataSource()
    test_NeoRawIOAggregator()
    