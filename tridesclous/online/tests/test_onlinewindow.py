from tridesclous import *
from tridesclous.online import *
from tridesclous.gui import QT


import  pyqtgraph as pg

import pyacq


import numpy as np
import time
import os
import shutil





def test_OnlineWindow():
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    filename = filenames[0] #only first file
    #~ print(params)
    sigs = np.fromfile(filename, dtype=params['dtype']).reshape(-1, params['total_channel'])
    sigs = sigs.astype('float32')
    sample_rate = params['sample_rate']
    #~ dataio.set_data_source(type='RawData', filenames=filenames, **params)
    #~ channel_group = {0:{'channels':[5, 6, 7, 8]}}
    #~ dataio.set_channel_groups(channel_group)
    #~ print(filenames)
    #~ print(sigs.shape)
    #~ exit()
    
    
    channel_indexes = [5,6,7,8]
        
    app = pg.mkQApp()
    
    w = OnlineWindow()
    w.resize(800, 600)
    w.show()
    
    
    app.exec_()





if __name__ =='__main__':
    test_OnlineWindow()
    
