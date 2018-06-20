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
    man = pyacq.create_manager(auto_close_at_exit=True)
    #~ ng0 = man.create_nodegroup()
    ng0 = None
    dev = make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup=ng0, chunksize=chunksize)
    
    
    
    channel_groups = {
        0 : [5,6,7,8],
        #~ 1 : [1,2,3,4],
        #~ 2 : [9,10,11],
    }
    
    workdir = 'test_onlinewindow'
    
    #~ if os.path.exists(workdir):
        #~ shutil.rmtree(workdir)
    
    app = pg.mkQApp()
    
    
    
    windows = []
    for chan_grp, channel_indexes in channel_groups.items():
        
        # nodegroup_firend
        #~ nodegroup_friend = man.create_nodegroup() 
        nodegroup_friend = None

        w = OnlineWindow()
        w.configure(chan_grp=chan_grp, channel_indexes=channel_indexes, chunksize=chunksize,
                        workdir=workdir, nodegroup_friend=nodegroup_friend)
        w.input.connect(dev.output)
        w.initialize()
        
        w.resize(800, 600)
        w.show()
        w.start()
        windows.append(w)
    
    dev.start()
    
    
    def terminate():
        dev.stop()
        for w in windows:
            w.stop()
        app.quit()
        man.close()
        
    
    
    app.exec_()





if __name__ =='__main__':
    test_OnlineWindow()
    
