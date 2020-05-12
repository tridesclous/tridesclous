"""
This script is equivalent of the jupyter notebook example_locust_dataset.ipynb
but in a standard python script.

"""

from tridesclous import *
import pyqtgraph as pg


from matplotlib import pyplot
import time
from pprint import pprint

dirname = 'tridesclous_olfactory_bulb'

def initialize_catalogueconstructor():
    #download dataset
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    print(filenames)
    print(params)

    print()
    #create a DataIO
    import os, shutil
    
    if os.path.exists(dirname):
        #remove is already exists
        shutil.rmtree(dirname)    
    dataio = DataIO(dirname=dirname)

    # feed DataIO
    dataio.set_data_source(type='RawData', filenames=filenames, **params)

    #The dataset contains 16 channels but 14 and 15 are respiration and trigs.
    dataio.add_one_channel_group(channels=range(14), chan_grp=0)

    print(dataio)



def apply_catalogue_steps_auto():
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)


    params = get_auto_params_for_catalogue(dataio, chan_grp=0)
    params['adjacency_radius_um'] = 400.
    
    cc.apply_all_steps(params, verbose=True)
    
    print(cc)
    



def open_cataloguewindow():
    dataio = DataIO(dirname=dirname)
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()    


def run_peeler():
    dataio = DataIO(dirname=dirname)
    catalogue = dataio.load_catalogue(chan_grp=0)

    peeler_params = get_auto_params_for_peelers(dataio, chan_grp=0)
    pprint(peeler_params)    
    
    peeler = Peeler(dataio)
    peeler.change_params(catalogue=catalogue, **peeler_params)
    
    t1 = time.perf_counter()
    peeler.run()
    t2 = time.perf_counter()
    print('peeler.run', t2-t1)
    
    
    
def open_PeelerWindow():
    dataio = DataIO(dirname=dirname)
    print(dataio)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()



if __name__ =='__main__':
    #~ initialize_catalogueconstructor()
    
    #~ apply_catalogue_steps_auto()
    #~ open_cataloguewindow()
    
    #~ run_peeler()
    open_PeelerWindow()
    
