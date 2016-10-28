from .version import version as __version__
#~ from .dataio import DataIO
#~ from .tools import *
#~ from .peakdetector import *
#~ from .waveformextractor import *
#~ from .clustering import Clustering
#~ from .peeler import Peeler
#~ from .filter import SignalFilter

from .spikesorter import SpikeSorter

#~ from .mpl_plot import *

try:
    import PyQt5 # this force pyqtgraph to deal with Qt5
    from .gui import *
except ImportError:
    import logging
    logging.warning('Interactive GUI not availble')
    pass

