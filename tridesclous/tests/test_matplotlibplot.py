import numpy as np
import matplotlib.pyplot as plt

from tridesclous import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.matplotlibplot import *


import matplotlib.pyplot as plt

from tridesclous.tests.testingtools import setup_catalogue
import shutil

from tridesclous.tests.testingtools import ON_CI_CLOUD
import pytest


def setup_module():
    setup_catalogue('test_matplotlibplot', dataset_name='olfactory_bulb')

def teardown_module():
    shutil.rmtree('test_matplotlibplot')

    
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_plot_probe_geometry():
    dataio = DataIO('test_matplotlibplot')
    plot_probe_geometry(dataio, chan_grp=0) 

    
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_plot_signals():
    dataio = DataIO('test_matplotlibplot')
    catalogueconstructor = CatalogueConstructor(dataio=dataio, chan_grp=0)
    
    plot_signals(dataio, signal_type='initial')
    plot_signals(dataio, signal_type='processed')
    plot_signals(catalogueconstructor, signal_type='processed', with_peaks=True, time_slice=(2., 3))
    plot_signals(catalogueconstructor, signal_type='processed', with_span=True, time_slice=(2., 3))
    

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_plot_waveforms_with_geometry():
    nb_channel = 32
    waveforms = np.random.randn(200, 45, nb_channel)
    channels = np.arange(nb_channel)
    #~ geometry = {c: [np.random.randint(100), np.random.randint(100)] for c in channels}
    geometry = np.random.randint(low=0, high=100, size=(200, 2))
    
    #~ , channels, geometry
    #~ print(geometry)
    
    plot_waveforms_with_geometry(waveforms, channels, geometry) 

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_plot_waveforms():
    dataio = DataIO('test_matplotlibplot')
    catalogueconstructor = CatalogueConstructor(dataio=dataio, chan_grp=0)
    
    plot_waveforms(catalogueconstructor)


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_plot_features_scatter_2d():
    dataio = DataIO('test_matplotlibplot')
    catalogueconstructor = CatalogueConstructor(dataio=dataio, chan_grp=0)
    
    #~ plot_features_scatter_2d(catalogueconstructor)
    plot_features_scatter_2d(catalogueconstructor, labels=[0])

    
if __name__ == '__main__':
    setup_module()
    
    test_plot_probe_geometry()
    test_plot_signals()
    test_plot_waveforms_with_geometry()
    test_plot_waveforms()
    test_plot_features_scatter_2d()
    



    plt.show()