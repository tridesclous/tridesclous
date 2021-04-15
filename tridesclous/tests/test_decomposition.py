import numpy as np
import time
import os
import shutil

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor



from matplotlib import pyplot

from tridesclous.tests.testingtools import setup_catalogue

dataset_name='olfactory_bulb'


def setup_module():
    setup_catalogue('test_decomposition', dataset_name=dataset_name)

def teardown_module():
    if not(os.environ.get('APPVEYOR') in ('true', 'True')):
        # this fix appveyor teardown_module bug
        shutil.rmtree('test_decomposition')



def test_all_decomposition():
    dirname = 'test_decomposition'
    
    dataio = DataIO(dirname=dirname)
    cc = catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print(dataio)
    print(cc)
    
    methods = ['global_pca', 'pca_by_channel', 'peak_max', ] #'neighborhood_pca', 'tsne', 'pca_by_channel_then_tsne'
    for method in methods:
        t0 = time.perf_counter()
        cc.extract_some_features(method=method)
        t1 = time.perf_counter()
        print('extract_some_features', method, t1-t0)
    
    
    #~ from tridesclous.gui import mkQApp, CatalogueWindow
    #~ app = mkQApp()
    #~ win = CatalogueWindow(catalogueconstructor)
    #~ win.show()
    #~ app.exec_()

    

def debug_one_decomposition():
    dirname = 'test_catalogueconstructor'
    
    dataio = DataIO(dirname=dirname)
    cc = catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print(dataio)
    print(cc)
    
    t0 = time.perf_counter()
    #~ cc.extract_some_features(method='global_pca', n_components=7)
    #~ cc.extract_some_features(method='peak_max')
    cc.extract_some_features(method='pca_by_channel', n_components_by_channel=3)
    #~ cc.extract_some_features(method='neighborhood_pca', n_components_by_neighborhood=3, radius_um=500)
    
    print(cc.channel_to_features)
    print(cc.channel_to_features.shape)
    t1 = time.perf_counter()
    print('extract_some_features', t1-t0)
    
    #~ from tridesclous.gui import mkQApp, CatalogueWindow
    #~ app = mkQApp()
    #~ win = CatalogueWindow(catalogueconstructor)
    #~ win.show()
    #~ app.exec_()    


if __name__ == '__main__':
    #~ setup_module()
    #~ test_all_decomposition()
    
    debug_one_decomposition()

