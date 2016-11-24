import numpy as np
import matplotlib.pyplot as plt

from tridesclous import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.matplotlibplot import *


import matplotlib.pyplot as plt




def test_plot_waveforms():
    nb_channel = 32
    waveforms = np.random.randn(200, 45, nb_channel)
    channels = np.arange(nb_channel)
    geometry = {c: [np.random.randint(100), np.random.randint(100)] for c in channels}
    
    #~ , channels, geometry
    #~ print(geometry)
    
    plot_waveforms(waveforms, channels, geometry) 
    
    
    plt.show()
    
    
    
if __name__ == '__main__':
    test_plot_waveforms()
