import numpy as np
import time
import os
import shutil

from tridesclous import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.cataloguetools import get_auto_params_for_catalogue, apply_all_catalogue_steps

from matplotlib import pyplot as plt

def test_get_auto_params_for_catalogue():
    if os.path.exists('test_cataloguetools'):
        shutil.rmtree('test_cataloguetools')
        
    dataio = DataIO(dirname='test_cataloguetools')
    #~ localdir, filenames, params = download_dataset(name='olfactory_bulb')
    localdir, filenames, params = download_dataset(name='locust')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    
    params = get_auto_params_for_catalogue(dataio)
    print(params)
    print(params['cluster_method'])
    
    
    
    
def test_apply_all_catalogue_steps():
    if os.path.exists('test_cataloguetools'):
        shutil.rmtree('test_cataloguetools')
        
    dataio = DataIO(dirname='test_cataloguetools')
    #~ localdir, filenames, params = download_dataset(name='olfactory_bulb')
    localdir, filenames, params = download_dataset(name='locust')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    
    params = get_auto_params_for_catalogue(dataio)
    
    cc = CatalogueConstructor(dataio, chan_grp=0)
    apply_all_catalogue_steps(cc, params, verbose=True)
    
    
    
    
    
if __name__ == '__main__':
    #~ test_get_auto_params_for_catalogue()
    test_apply_all_catalogue_steps()
    
    