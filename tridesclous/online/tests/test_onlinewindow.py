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
    # get sigs
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    filename = filenames[0] #only first file
    sigs = np.fromfile(filename, dtype=params['dtype']).reshape(-1, params['total_channel'])
    sigs = sigs.astype('float32')
    sample_rate = params['sample_rate']
    
    chunksize = 1024
    
    # Device node
    #~ man = create_manager(auto_close_at_exit=True)
    #~ ng0 = man.create_nodegroup()
    ng0 = None
    dev = make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup=ng0, chunksize=chunksize)
    
    
    
    chan_grp = 0
    channel_indexes = [5,6,7,8]
    workdir = 'test_onlinewindow'
    
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    
    app = pg.mkQApp()
    
    w = OnlineWindow()
    w.configure(chan_grp=chan_grp, channel_indexes=channel_indexes, chunksize=chunksize, workdir=workdir)
    w.input.connect(dev.output)
    w.initialize()
    
    w.resize(800, 600)
    w.show()
    
    dev.start()
    w.start()
    
    def terminate():
        dev.stop()
        w.stop()
        app.quit()
    
    
    app.exec_()





if __name__ =='__main__':
    test_OnlineWindow()
    
