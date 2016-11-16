from .version import version as __version__

import PyQt5 # this force pyqtgraph to deal with Qt5

from .datasets import get_dataset
from .dataio import RawDataIO
from .catalogueconstructor import CatalogueConstructor
from .peeler import Peeler


#~ from .mpl_plot import *


from .gui import *


#~ try:
    #~ import PyQt5 # this force pyqtgraph to deal with Qt5
    #~ from .gui import *
#~ except ImportError:
    #~ import logging
    #~ logging.warning('Interactive GUI not availble')
    #~ pass

