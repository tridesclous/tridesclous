"""

.. automodule:: tridesclous.dataio

.. automodule:: tridesclous.catalogueconstructor

.. automodule:: tridesclous.peeler




"""
from .version import version as __version__
import os
import warnings

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
from .cltools import get_cl_device_list, set_default_cl_device

from .autoparams import get_auto_params_for_catalogue, get_auto_params_for_peelers


from .importers import import_from_spykingcircus, import_from_spike_interface

# exclude because import matplotlib

# from .matplotlibplot import *
# from .report import *

# from .gui import *

