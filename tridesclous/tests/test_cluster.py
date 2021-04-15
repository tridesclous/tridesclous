import numpy as np
import time
import os
import shutil

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor



from matplotlib import pyplot

from tridesclous.tests.testingtools import setup_catalogue


#~ dataset_name='olfactory_bulb'
#~ dataset_name = 'purkinje'
#~ dataset_name='locust'
dataset_name='striatum_rat'



def setup_module():
    setup_catalogue('test_cluster', dataset_name=dataset_name)

def teardown_module():
    if not(os.environ.get('APPVEYOR') in ('true', 'True')):
        # this fix appveyor teardown_module bug
        shutil.rmtree('test_cluster')


def test_sawchaincut():
    dirname = 'test_cluster'
    
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    #~ print(dataio)
    #~ print(cc)
    
    t0 = time.perf_counter()
    cc.find_clusters(method='sawchaincut', print_debug=True)
    t1 = time.perf_counter()
    print('cluster', t1-t0)
    
    #~ print(cc)


    if __name__ == '__main__':
        from tridesclous.gui import mkQApp, CatalogueWindow
        app = mkQApp()
        win = CatalogueWindow(cc)
        win.show()
        app.exec_()


def test_pruningshears():

    dirname = 'test_cluster'
    
    
    dataio = DataIO(dirname=dirname)
    print(dataio)
    cc = CatalogueConstructor(dataio=dataio)
    #~ print(cc.mode)
    
    #~ cc.extract_some_features(method='pca_by_channel')
    #~ print(dataio)
    #~ print(cc)
    
    if dataset_name == 'olfactory_bulb':
        kargs = dict(adjacency_radius_um = 420)
    else:
        kargs = {}
    
    t0 = time.perf_counter()
    #~ cc.find_clusters(method='pruningshears', print_debug=True)
    #~ cc.find_clusters(method='pruningshears', print_debug=True, debug_plot=True, **kargs)
    cc.find_clusters(method='pruningshears', print_debug=False, debug_plot=False, **kargs)
    t1 = time.perf_counter()
    print('cluster', t1-t0)

    if __name__ == '__main__':
        from tridesclous.gui import mkQApp, CatalogueWindow
        app = mkQApp()
        win = CatalogueWindow(cc)
        win.show()
        app.exec_()
    


if __name__ == '__main__':
    setup_module()
    test_sawchaincut()
    test_pruningshears()
    
    
