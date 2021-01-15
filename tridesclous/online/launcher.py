"""
Main app mauncher for different device:
  * demo with pyacq buffer
  * open ephys
  * blackrock ...



"""
import os
import numpy as np

import pyqtgraph as pg

from tridesclous.tools import open_prb
from tridesclous.datasets import download_dataset
from .onlinewindow import TdcOnlineWindow

from ..gui import QT

import pyacq
from pyacq.devices import NumpyDeviceBuffer
from pyacq.devices import OpenEphysGUIRelay



    
    
def make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup = None, chunksize=1024):
    length, nb_channel = sigs.shape
    length -= length%chunksize
    sigs = sigs[:length, :]
    dtype = sigs.dtype
    channel_names = ['channel{}'.format(c) for c in range(sigs.shape[1])]
    
    if nodegroup is None:
        dev = NumpyDeviceBuffer()
    else:
        dev = nodegroup.create_node('NumpyDeviceBuffer')
    dev.configure(nb_channel=nb_channel, sample_interval=1./sample_rate, chunksize=chunksize, buffer=sigs, channel_names=channel_names)
    dev.output.configure(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
    dev.initialize()
    
    return dev




def start_online_window(pyacq_dev, prb_dict_or_file, workdir=None, n_process=1,
                pyacq_manager=None, chunksize=None, peeler_params={}, initial_catalogue_params={},
                outputstream_params=None):
    
    if pyacq_manager is None and n_process>=1:
        pyacq_manager = pyacq.create_manager(auto_close_at_exit=True)
    
    if workdir is None:
        workdir = os.path.join(os.path.expanduser("~/Desktop"), 'test_tdconlinewindow_'+''.join(str(e) for e in np.random.randint(0,10,8)))
    
    if n_process>=1:
        nodegroup_friends = [pyacq_manager.create_nodegroup() for i in range(n_process)]
    else:
        nodegroup_friends = None
    
    if isinstance(prb_dict_or_file, str):
        channel_groups = open_prb(prb_dict_or_file)
    elif isinstance(prb_dict_or_file, dict):
        channel_groups = prb_dict_or_file
    
    #~ if chunksize is None:
        #~ # todo set latency in ms
        #~ chunksize = 928
    
    
    sample_rate = pyacq_dev.output.params['sample_rate']
    #~ print('sample_rate', sample_rate)
    #~ exit()
    
    # force sharedmem
    if outputstream_params is None:
        
        # TODO cleaner buffer size settings see also OnlineWindow
        buffer_size = int(15. * sample_rate)        
        
        outputstream_params={'protocol': 'tcp', 'interface':'127.0.0.1', 'transfermode':'sharedmem'}
        outputstream_params['buffer_size'] = buffer_size

    
    win = TdcOnlineWindow()
    win.configure(channel_groups=channel_groups,
                    chunksize=chunksize,
                    workdir=workdir,
                    outputstream_params = outputstream_params,
                    nodegroup_friends=nodegroup_friends,
                    peeler_params=peeler_params,
                    initial_catalogue_params=initial_catalogue_params,
                    )
    
    win.input.connect(pyacq_dev.output)
    win.initialize()
    #~ win.show()
    #~ win.start()
    
    
    return pyacq_manager, win


def start_online_pyacq_buffer_demo(dataset_name='olfactory_bulb'):
    

    # get sigs
    localdir, filenames, params = download_dataset(name=dataset_name)
    filename = filenames[0] #only first file
    sigs = np.fromfile(filename, dtype=params['dtype']).reshape(-1, params['total_channel'])
    sigs = sigs.astype('float32')
    sample_rate = params['sample_rate']
    
    app = pg.mkQApp()
    
    chunksize = 1000
    
    man = pyacq.create_manager(auto_close_at_exit=True)
    ng0 = man.create_nodegroup() # process in background
    dev = make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup=ng0, chunksize=chunksize)

    channel_groups = {
        0 : {'channels': [5, 6, 7, 8],
                'geometry' : {
                    5 : [0., 50.],
                    6:  [50., 0.],
                    7: [0, -50.],
                    8:  [-50,0.],
                },
            },
        1 : {'channels': [1, 2, 3, 4],
                'geometry' : {
                    1 : [0., 50.],
                    2:  [50., 0.],
                    3: [0, -50.],
                    4:  [-50,0.],
                }
            }
        }
    
    #~ workdir = 'demo_onlinewindow'
    workdir = None
    
    man, win = start_online_window(dev, channel_groups,
                        workdir=None,
                        n_process=2,
                        #~ n_process=0,
                        pyacq_manager=man,
                        chunksize=chunksize,
                        #~ initial_catalogue_params={'preprocessor': {'pad_width':10}},
                        )


    win.show()
    
    win.start()
    dev.start()
    
    app.exec_()
    

def start_online_openephys(prb_filename=None, workdir=None):
    app = pg.mkQApp()
    
    if prb_filename is None:
        fd = QT.QFileDialog(fileMode=QT.QFileDialog.AnyFile, acceptMode=QT.QFileDialog.AcceptOpen)
        fd.setWindowTitle('Select a probe file')
        fd.setNameFilters(['Probe geometry (*.PRB, *.prb)', 'All (*)'])
        fd.setViewMode( QT.QFileDialog.Detail )
        if fd.exec_():
            prb_filename = fd.selectedFiles()[0]
        else:
            raise ValueError('Must select probe file')

    man = pyacq.create_manager(auto_close_at_exit=True)
    
    
    #~ dev = OpenEphysGUIRelay()
    ng0 = man.create_nodegroup() # process in background
    dev = ng0.create_node('OpenEphysGUIRelay')
    dev.configure(openephys_url='tcp://127.0.0.1:20000')
    dev.outputs['signals'].configure()
    dev.initialize()
    
    chunksize = dev.outputs['signals'].params['shape'][0]
    print('chunksize', chunksize)
    
    
    #~ prb_filename = 'probe_openephys_16ch.prb'
    
    man, win = start_online_window(dev, prb_filename,
                        workdir=workdir,
                        n_process=1,
                        pyacq_manager=man,
                        chunksize=chunksize)
    
    win.show()
    
    win.start()
    dev.start()
    
    app.exec_()    


