from tridesclous import *

import matplotlib.pyplot as plt

from tridesclous.tests.testingtools import setup_catalogue

from tridesclous.tests.testingtools import ON_CI_CLOUD
import pytest

import shutil
import os
import time


def setup_module():
    setup_catalogue('test_metrics', dataset_name='olfactory_bulb')

def teardown_module():
    if not(os.environ.get('APPVEYOR') in ('true', 'True')):
        # this fix appveyor teardown_module bug
        shutil.rmtree('test_metrics')
    
    

def test_all_metrics():
    dataio = DataIO(dirname='test_metrics')
    cc = CatalogueConstructor(dataio=dataio)
    
    #~ cc.compute_spike_waveforms_similarity()
    
    t0 = time.perf_counter()
    cc.compute_cluster_similarity()
    t1 = time.perf_counter()
    print('compute_cluster_similarity', t1-t0)
    print(cc.cluster_similarity)
    
    t0 = time.perf_counter()
    cc.compute_cluster_ratio_similarity()
    t1 = time.perf_counter()
    print('compute_cluster_ratio_similarity', t1-t0)
    print(cc.cluster_ratio_similarity)
    
    t0 = time.perf_counter()
    cc.compute_spike_silhouette()
    t1 = time.perf_counter()
    print('compute_spike_silhouette', t1-t0)
    print(cc.spike_silhouette)

    


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_cluster_ratio():
    dataio = DataIO(dirname='test_metrics')
    cc = CatalogueConstructor(dataio=dataio)
    
    cc.compute_cluster_similarity()
    cc.compute_cluster_ratio_similarity()

    for name in('cluster_similarity', 'cluster_ratio_similarity'):
        d = getattr(cc, name).copy()
        fig, ax = plt.subplots()
        im  = ax.matshow(d, cmap='viridis')
        im.set_clim(0,1)
        fig.colorbar(im)
        ax.set_title(name)
    
    


if __name__ == '__main__':
    setup_module()
    
    test_all_metrics()
    test_cluster_ratio()
    
    #~ plt.show()