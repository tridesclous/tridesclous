import numpy as np
import time
import os
import shutil
from pprint import pprint

from tridesclous import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.autoparams import get_auto_params_for_catalogue, get_auto_params_for_peelers

from matplotlib import pyplot as plt

def test_get_auto_params():
    if os.path.exists('test_autoparams'):
        shutil.rmtree('test_autoparams')
        
    dataio = DataIO(dirname='test_autoparams')
    #~ localdir, filenames, params = download_dataset(name='olfactory_bulb')
    localdir, filenames, params = download_dataset(name='locust')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    
    params = get_auto_params_for_catalogue(dataio)
    pprint(params)
    
    
    params = get_auto_params_for_peelers(dataio)
    pprint(params)
    
    
    #~ print(params['cluster_method'])
    #~ print(params['cluster_kargs'])


if __name__ == '__main__':
    test_get_auto_params()
