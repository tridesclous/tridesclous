import numpy as np
import time
import os
import shutil

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor

from tridesclous import mkQApp, CatalogueWindow

from matplotlib import pyplot

from tridesclous.tests.testingtools import setup_catalogue


#~ dataset_name='olfactory_bulb'
dataset_name = 'purkinje'
#~ dataset_name='locust'
#~ dataset_name='striatum_rat'



def setup_module():
    cc, params = setup_catalogue('test_cluster', dataset_name=dataset_name)
    
    cc.find_clusters(method=params['cluster_method'], **params['cluster_kargs'])

def test_auto_split():
    dirname = 'test_cluster'
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    cc.find_clusters(method='pruningshears')
    print(cc)
    
    t1 = time.perf_counter()
    cc.auto_split_cluster()
    t2 = time.perf_counter()
    print('auto_split_cluster', t2-t1)
    
    print(cc)

def test_auto_merge():
    dirname = 'test_cluster'
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    t1 = time.perf_counter()
    cc.auto_merge_cluster()
    t2 = time.perf_counter()
    print('auto_merge_cluster', t2-t1)



    
    
    
if __name__ == '__main__':
    setup_module()
    test_auto_split()
    test_auto_merge()
    
    