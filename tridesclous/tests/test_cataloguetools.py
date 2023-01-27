import numpy as np
import time
import os
import shutil
from pprint import pprint

from tridesclous import download_dataset
from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor
from tridesclous.autoparams import get_auto_params_for_catalogue
from tridesclous.cataloguetools import  apply_all_catalogue_steps

from matplotlib import pyplot as plt




def test_apply_all_catalogue_steps():
    if os.path.exists('test_cataloguetools'):
        shutil.rmtree('test_cataloguetools')

    dataio = DataIO(dirname='test_cataloguetools')
    #~ localdir, filenames, params = download_dataset(name='olfactory_bulb')
    localdir, filenames, params = download_dataset(name='locust')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)

    params = get_auto_params_for_catalogue(dataio)

    cc = CatalogueConstructor(dataio, chan_grp=0)
    apply_all_catalogue_steps(cc, params, verbose=True)





if __name__ == '__main__':
    test_apply_all_catalogue_steps()

