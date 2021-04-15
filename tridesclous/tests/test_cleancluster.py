import numpy as np
import time
import os
import shutil

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor



from tridesclous.tests.testingtools import setup_catalogue


#~ dataset_name='olfactory_bulb'
dataset_name = 'purkinje'
#~ dataset_name='locust'
#~ dataset_name='striatum_rat'



def setup_module():
    cc, params = setup_catalogue('test_cleancluster', dataset_name=dataset_name)
    
    cc.find_clusters(method=params['cluster_method'], **params['cluster_kargs'])
    
    cc.create_savepoint(name='after_find_clusters')
    

def restore_savepoint(dirname, savepoint=None):
    folder = os.path.join(dirname, 'channel_group_0', 'catalogue_constructor')
    savepoint_folder = os.path.join(dirname, 'channel_group_0', 'catalogue_constructor_SAVEPOINT_' + savepoint)
    
    assert os.path.exists(savepoint_folder)
    
    shutil.rmtree(folder)
    shutil.copytree(savepoint_folder, folder)
    

def test_auto_split():
    dirname = 'test_cleancluster'
    
    restore_savepoint(dirname, savepoint='after_find_clusters')
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    cc.find_clusters(method='pruningshears')
    print(cc)
    print(cc.n_jobs)
    
    t1 = time.perf_counter()
    cc.cache_some_waveforms()
    t2 = time.perf_counter()
    print('cache_some_waveforms', t2-t1)
    
    t1 = time.perf_counter()
    cc.auto_split_cluster()
    t2 = time.perf_counter()
    print('auto_split_cluster', t2-t1)
    
    print(cc)
    
    cc.create_savepoint(name='after_auto_split')




def test_trash_not_aligned():
    dirname = 'test_cleancluster'
    
    restore_savepoint(dirname, savepoint='after_auto_split')
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    t1 = time.perf_counter()
    cc.trash_not_aligned()
    t2 = time.perf_counter()
    print('trash_not_aligned', t2-t1)
    
    cc.create_savepoint(name='after_trash_not_aligned')


def test_auto_merge():
    dirname = 'test_cleancluster'
    
    restore_savepoint(dirname, savepoint='after_trash_not_aligned')
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)

    t1 = time.perf_counter()
    cc.cache_some_waveforms()
    t2 = time.perf_counter()
    print('cache_some_waveforms', t2-t1)

    t1 = time.perf_counter()
    cc.compute_all_centroid()
    t2 = time.perf_counter()
    print('compute_all_centroid', t2-t1)

    
    t1 = time.perf_counter()
    cc.auto_merge_cluster()
    t2 = time.perf_counter()
    print('auto_merge_cluster', t2-t1)
    
    cc.create_savepoint(name='after_auto_merge_cluster')




def test_trash_low_extremum():
    dirname = 'test_cleancluster'
    
    restore_savepoint(dirname, savepoint='after_auto_merge_cluster')
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    print(cc)
    
    t1 = time.perf_counter()
    cc.trash_low_extremum()
    t2 = time.perf_counter()
    print('trash_low_extremum', t2-t1)
    
    cc.create_savepoint(name='after_trash_low_extremum')
    print(cc)


def test_trash_small_cluster():
    dirname = 'test_cleancluster'
    
    restore_savepoint(dirname, savepoint='after_trash_low_extremum')
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    t1 = time.perf_counter()
    cc.trash_small_cluster()
    t2 = time.perf_counter()
    print('trash_small_cluster', t2-t1)


    
    
    
if __name__ == '__main__':
    setup_module()
    test_auto_split()
    test_trash_not_aligned()
    test_auto_merge()
    test_trash_low_extremum()
    test_trash_small_cluster()
    
    