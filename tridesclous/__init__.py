from .version import version as __version__

import PyQt5 # this force pyqtgraph to deal with Qt5

from .datasets import download_dataset, get_dataset
from .datasource import InMemoryDataSource, RawDataSource, BlackrockDataSource
from .dataio import DataIO
from .catalogueconstructor import CatalogueConstructor
from .peeler import Peeler
from .peeler_cl import Peeler_OpenCl

#~ from .mpl_plot import *


from .gui import *


#~ try:
    #~ import PyQt5 # this force pyqtgraph to deal with Qt5
    #~ from .gui import *
#~ except ImportError:
    #~ import logging
    #~ logging.warning('Interactive GUI not availble')
    #~ pass

