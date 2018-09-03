"""

.. automodule:: tridesclous.dataio

.. automodule:: tridesclous.catalogueconstructor

.. automodule:: tridesclous.peeler




"""
from .version import version as __version__

import PyQt5 # this force pyqtgraph to deal with Qt5

# this avoid tinker problem when not installed
import matplotlib
matplotlib.use('Qt5Agg')

from .datasets import download_dataset, get_dataset

#dynamic import 
from .datasource import data_source_classes
for c in data_source_classes.values():
    globals()[c.__name__] = c

from .tools import open_prb
from .dataio import DataIO
from .signalpreprocessor import offline_signal_preprocessor, estimate_medians_mads_after_preprocesing
from .catalogueconstructor import CatalogueConstructor
from .cataloguetools import apply_all_catalogue_steps
from .peeler import Peeler
# from .peeler_cl import Peeler_OpenCl

from .importers import import_from_spykingcircus

from .matplotlibplot import *


from .gui import *

