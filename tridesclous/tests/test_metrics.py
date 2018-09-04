from tridesclous import *
import  pyqtgraph as pg
import matplotlib.pyplot as plt

from tridesclous.tests.testingtools import setup_catalogue

from tridesclous.tests.testingtools import ON_CI_CLOUD
import pytest

import shutil


def setup_module():
    setup_catalogue('test_metrics', dataset_name='olfactory_bulb')

def teardown_module():
    shutil.rmtree('test_metrics')
    
    

def test_all_metrics():
    dataio = DataIO(dirname='test_metrics')
    cc = CatalogueConstructor(dataio=dataio)
    
    cc.compute_spike_waveforms_similarity()
    cc.compute_cluster_similarity()
    cc.compute_cluster_ratio_similarity()
    cc.compute_spike_silhouette()


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_cluster_ratio():
    dataio = DataIO(dirname='test_metrics')
    cc = CatalogueConstructor(dataio=dataio)
    
    cc.compute_cluster_similarity()
    cc.compute_cluster_ratio_similarity()

    for name in('cluster_similarity', 'cluster_ratio_similarity'):
        d = getattr(cc, name)
        fig, ax = plt.subplots()
        im  = ax.matshow(d, cmap='viridis')
        im.set_clim(0,1)
        fig.colorbar(im)
        ax.set_title(name)
        
    del fig, ax
    
    


if __name__ == '__main__':
    setup_module()
    
    test_all_metrics()
    test_cluster_ratio()
    
    #~ plt.show()