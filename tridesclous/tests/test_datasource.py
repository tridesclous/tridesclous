import pytest
import os, tempfile, shutil
import numpy as np

from tridesclous import download_dataset
#~ from tridesclous import DataIO
from tridesclous.datasource import RawDataSource,InMemoryDataSource



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





if __name__=='__main__':
    test_InMemoryDataSource()
    test_RawDataSource()
    