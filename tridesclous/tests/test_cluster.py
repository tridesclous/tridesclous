import numpy as np
import time
import os
import shutil

from tridesclous.dataio import DataIO
from tridesclous.catalogueconstructor import CatalogueConstructor

from tridesclous import mkQApp, CatalogueWindow

from matplotlib import pyplot

# run test_catalogueconstructor.py before this





def test_dirtycut():
    dirname = 'test_catalogueconstructor'
    #~ dirname = '/home/samuel/Documents/projet/tridesclous/example/tridesclous_locust/'
    
    dataio = DataIO(dirname=dirname)
    cc = catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print(dataio)
    print(cc)
    
    
    cc.find_clusters(method='dirtycut', selection=None)
    
    print(cc)


    app = mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()


if __name__ == '__main__':
    test_dirtycut()
