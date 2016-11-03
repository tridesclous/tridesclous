from .version import version as __version__

from .datasets import get_dataset
from .dataio import RawDataIO
from .spikesorter import SpikeSorter


#~ from .mpl_plot import *

#~ try:
    #~ import PyQt5 # this force pyqtgraph to deal with Qt5
    #~ from .gui import *
#~ except ImportError:
    #~ import logging
    #~ logging.warning('Interactive GUI not availble')
    #~ pass

