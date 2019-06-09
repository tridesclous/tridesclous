"""

.. automodule:: tridesclous.dataio

.. automodule:: tridesclous.catalogueconstructor

.. automodule:: tridesclous.peeler




"""
from .version import version as __version__

import PyQt5 # this force pyqtgraph to deal with Qt5

# For matplotlib to Qt5 : 
#   * this avoid tinker problem when not installed
#   * work better with GUI
#   * trigger a warning on notebook
import matplotlib
import warnings
with warnings.catch_warnings():
    try:                                                                                                                                                                                                                                    
        warnings.simplefilter("ignore")
        matplotlib.use('Qt5Agg')                                                                                                                                                                                                            
    except:
        # on server without screen this is not possible.
        pass


from .datasets import download_dataset, get_dataset

#dynamic import 
from .datasource import data_source_classes
for c in data_source_classes.values():
    globals()[c.__name__] = c

from .tools import open_prb
from .dataio import DataIO
from .signalpreprocessor import offline_signal_preprocessor, estimate_medians_mads_after_preprocesing
from .catalogueconstructor import CatalogueConstructor
from .cataloguetools import apply_all_catalogue_steps, get_auto_params_for_catalogue
from .peeler import Peeler
from .cltools import get_cl_device_list, set_default_cl_device

# from .peeler_cl import Peeler_OpenCl

from .importers import import_from_spykingcircus, import_from_spike_interface

from .matplotlibplot import *
from .report import *

from .gui import *

