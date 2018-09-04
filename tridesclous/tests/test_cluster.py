import numpy as np
import time
import os
import shutil

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor

from tridesclous import mkQApp, CatalogueWindow

from matplotlib import pyplot

from tridesclous.tests.testingtools import setup_catalogue


#~ dataset_name='olfactory_bulb'
dataset_name = 'purkinje'
#~ dataset_name='locust'
#~ dataset_name='striatum_rat'



def setup_module():
    setup_catalogue('test_cluster', dataset_name=dataset_name)

def teardown_module():
    if not(os.environ.get('APPVEYOR') in ('true', 'True')):
        # this fix appveyor teardown_module bug
        shutil.rmtree('test_cluster')


def test_sawchaincut():
    #~ dirname = 'test_catalogueconstructor'
    #~ dirname = '/home/samuel/Documents/projet/tridesclous/example/tridesclous_locust/'

    #~ dirname = '/home/samuel/Documents/projet/DataSpikeSorting/GT 252/tdc_20170623_patch1/'
    #~ dirname = '/home/samuel/Documents/projet/tridesclous/example/tridesclous_locust/'
    #~ dirname = '/home/samuel/Documents/projet/tridesclous/example/tridesclous_olfactory_bulb/'
    #~ dirname = '/home/samuel/Documents/projet/tridesclous/example/tridesclous_olfactory_bulb/'
    #~ dirname = '/home/samuel/Documents/projet/DataSpikeSorting/kampff/tdc_2015_09_03_Cell9.0/'
    #~ dirname = '/home/samuel/Documents/projet/DataSpikeSorting/spikesortingtest/tdc_silico_0/'
    #~ dirname = '/home/samuel/Documents/projet/tridesclous/example/tridesclous_purkinje/'
    
    dirname = 'test_cluster'
    
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    #~ print(dataio)
    #~ print(cc)
    
    t0 = time.perf_counter()
    cc.find_clusters(method='sawchaincut', print_debug=True)
    t1 = time.perf_counter()
    print('cluster', t1-t0)
    #~ exit()
    
    #~ print(cc)


    if __name__ == '__main__':
        app = mkQApp()
        win = CatalogueWindow(cc)
        win.show()
        app.exec_()


if __name__ == '__main__':
    setup_module()
    test_sawchaincut()
    
