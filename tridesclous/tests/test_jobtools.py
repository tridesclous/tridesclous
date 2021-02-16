import numpy as np
import time
import os
import shutil
from pprint import pprint

from tridesclous import DataIO, CatalogueConstructor
from tridesclous.jobtools import run_parallel_read_process_write

from matplotlib import pyplot

from tridesclous.tests.testingtools import setup_catalogue

from tridesclous.catalogueconstructor import _dtype_peak


#~ dataset_name='olfactory_bulb'
dataset_name = 'purkinje'
#~ dataset_name='locust'
#~ dataset_name='striatum_rat'



def setup_module():
    cc, params = setup_catalogue('test_jobtools', dataset_name=dataset_name)
    pprint(params)


def test_run_parallel_signalprocessor():
    
    dirname = 'test_jobtools'
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    #~ print(cc)
    
    seg_num = 0
    length = 320000
    #~ n_jobs = 2
    n_jobs = 4
    
    cc.arrays.initialize_array('all_peaks', cc.memory_mode,  _dtype_peak, (-1, ))
    
    t0 = time.perf_counter()
    run_parallel_read_process_write(cc, seg_num, length, n_jobs)
    t1 = time.perf_counter()
    print(t1-t0)
    
    cc.arrays.finalize_array('all_peaks')
    



if __name__ == '__main__':
    #~ setup_module()
    test_run_parallel_signalprocessor()
    



