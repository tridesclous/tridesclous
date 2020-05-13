"""
The dataset is here http://www.kampff-lab.org/validating-electrodes/

You have to download 2015_09_03_Cell9.0 and put files
somewhere on your machine.

Then change working_dir = ... to the correct path

"""
from tridesclous import *
import pyqtgraph as pg

import os
from urllib.request import urlretrieve
import time

from matplotlib import pyplot


# !!!!!!!! change the working dir here
#working_dir = '/home/samuel/Documents/projet/DataSpikeSorting/kampff/'
working_dir = '/media/samuel/SamCNRS/DataSpikeSorting/kampff/'


dirname = working_dir+'tdc_2015_09_09_Pair_6_0'


 
def initialize_catalogueconstructor():
    #setup file source
    filename = working_dir+'2015_09_09_Pair_6_0/'+'amplifier2015-09-09T17_46_43.bin'
    print(filename)
    dataio = DataIO(dirname=dirname)
    dataio.set_data_source(type='RawData', filenames=[filename], dtype='uint16',
                                     total_channel=128, sample_rate=30000.)
    print(dataio)#check
    
    #setup probe file
    # Pierre Yger have done the PRB file in spyking-circus lets download
    # it with the build-in dataio.download_probe
    dataio.download_probe('kampff_128', origin='spyking-circus')
    
    #initiailize catalogue
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    print(catalogueconstructor)

def apply_catalogue_steps_auto():
    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)

    params = get_auto_params_for_catalogue(dataio, chan_grp=1)
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
    catalogue = dataio.load_catalogue(chan_grp=1)
    
    print(dataio)
    peeler = Peeler(dataio)
    peeler.change_params(catalogue=catalogue,
                engine='geometrical',
                use_sparse_template=True,
                sparse_threshold_mad=1.5,
                argmin_method='opencl')
    
    t1 = time.perf_counter()
    peeler.run(duration=None)
    t2 = time.perf_counter()
    print('peeler.run_loop', t2-t1)


def open_PeelerWindow():
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=1)

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

    

