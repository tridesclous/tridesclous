from tridesclous import *
from tridesclous.online import HAVE_PYACQ

if HAVE_PYACQ:
    from tridesclous.online import *
    import pyacq
    
from tridesclous.gui import QT


import  pyqtgraph as pg




import numpy as np
import time
import os
import shutil

import pytest

from tridesclous.tests.testingtools import ON_CI_CLOUD


@pytest.mark.skipif(not HAVE_PYACQ, reason='no pyacq')
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_TdcOnlineWindow():
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
        0 : {'channels': [5, 6, 7, 8],
                'geometry': {
                    5: [0.0, 50.0],
                    6: [ 50.0, 0.0],
                    7: [0.0, -50.0],
                    8: [-50,0.0]
                }
        },
        1 : {'channels': [1, 2, 3, 4],
                'geometry': {
                    1: [0.0, 50.0],
                    2: [ 50.0, 0.0],
                    3: [0.0, -50.0],
                    4: [-50,0.0]
                }
        },
        2 : {'channels': [9, 10, 11],
                'geometry': {
                    9: [0.0, 50.0],
                    10: [ 50.0, 0.0],
                    11: [0.0, -50.0],
                }
        },
    }

    workdir = 'test_tdconlinewindow'
    
    #~ if os.path.exists(workdir):
        #~ shutil.rmtree(workdir)
    
    app = pg.mkQApp()
    
    
    
        
    # nodegroup_firend
    #~ nodegroup_friends = [man.create_nodegroup() for chan_grp in channel_groups]
    nodegroup_friends = None

    w = TdcOnlineWindow()
    w.configure(channel_groups=channel_groups, chunksize=chunksize,
                    workdir=workdir, nodegroup_friends=nodegroup_friends)
    
    w.input.connect(dev.output)
    w.initialize()
    w.show()

    w.start()
    
    dev.start()
    
    
    #~ def terminate():
        #~ dev.stop()
        #~ w.stop()
        #~ app.quit()
        #~ man.close()
        
    
    if __name__ =='__main__':
        app.exec_()





if __name__ =='__main__':
    test_TdcOnlineWindow()
    
