from tridesclous import *
from tridesclous.online import *


import  pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyacq
from pyacq.viewers import QOscilloscope, QTimeFreq


import numpy as np
import time
import os
import shutil




def setup_catalogue():
    if os.path.exists('tridesclous_onlinepeeler'):
        shutil.rmtree('tridesclous_onlinepeeler')
    
    dataio = DataIO(dirname='tridesclous_onlinepeeler')
    
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    filenames = filenames[:1] #only first file
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    channel_group = {0:{'channels':[5, 6, 7, 8]}}
    dataio.set_channel_groups(channel_group)
    
    
    
    
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    params = get_auto_params_for_catalogue(dataio, chan_grp=0)
    apply_all_catalogue_steps(catalogueconstructor, params, verbose=True)
    




    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    app = pg.mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    app.exec_()

    


def tridesclous_onlinepeeler():
    dataio = DataIO(dirname='tridesclous_onlinepeeler')
    catalogue = dataio.load_catalogue(chan_grp=0)
    
    #~ catalogue.pop('clusters')
    #~ def print_dict(d):
        #~ for k, v in d.items():
            #~ if type(v) is dict:
                #~ print('k', k, 'dict')
                #~ print_dict(v)
            #~ else:
                #~ print('k', k, type(v))
        
    #~ print_dict(catalogue)
    
    #~ from pyacq.core.rpc.serializer import MsgpackSerializer
    #~ serializer = MsgpackSerializer()
    #~ b = serializer.dumps(catalogue)
    #~ catalogue2 = serializer.loads(b)
    #~ print(catalogue2['clusters'])
    #~ exit()
    
    sigs = dataio.datasource.array_sources[0]
    
    sigs = sigs.astype('float32').copy()
    
    sample_rate = dataio.sample_rate
    in_group_channels = dataio.channel_groups[0]['channels']
    #~ print(channel_group)
    
    chunksize = 1024
    
    
    # Device node
    man = pyacq.create_manager(auto_close_at_exit=True)
    ng0 = man.create_nodegroup()
    #~ ng0 = None
    ng1 = man.create_nodegroup()
    #~ ng1 = None
    ng2 = man.create_nodegroup()
    #~ ng2 = None
    
    
    dev = make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup=ng0, chunksize=chunksize)
    #~ print(type(dev))
    #~ exit()
    #~ print(dev.output.params)
    #~ exit()

    
    app = pg.mkQApp()
    
    dev.start()
    
    # Node QOscilloscope
    oscope = QOscilloscope()
    oscope.configure(with_user_dialog=True)
    oscope.input.connect(dev.output)
    oscope.initialize()
    oscope.show()
    oscope.start()
    oscope.params['decimation_method'] = 'min_max'
    oscope.params['mode'] = 'scan'
    oscope.params['scale_mode'] = 'by_channel'
    

    # Node Peeler
    if ng1 is None:
        peeler = OnlinePeeler()
    else:
        ng1.register_node_type_from_module('tridesclous.online', 'OnlinePeeler')
        peeler = ng1.create_node('OnlinePeeler')
    
    peeler.configure(catalogue=catalogue, in_group_channels=in_group_channels, chunksize=chunksize)
    #~ print(dev.output.params)
    #~ print(peeler.input.connect)
    #~ exit()
    peeler.input.connect(dev.output)
    #~ exit()
    stream_params = dict(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
    peeler.outputs['signals'].configure(**stream_params)
    peeler.outputs['spikes'].configure(**stream_params)
    peeler.initialize()
    peeler.start()
    
    # Node traceviewer
    if ng2 is None:
        tviewer = OnlineTraceViewer()
    else:
        ng2.register_node_type_from_module('tridesclous.online', 'OnlineTraceViewer')
        tviewer = ng2.create_node('OnlineTraceViewer')
        
    tviewer.configure(catalogue=catalogue)
    tviewer.inputs['signals'].connect(peeler.outputs['signals'])
    tviewer.inputs['spikes'].connect(peeler.outputs['spikes'])
    tviewer.initialize()
    tviewer.show()
    tviewer.start()
    tviewer.params['xsize'] = 3.
    tviewer.params['decimation_method'] = 'min_max'
    tviewer.params['mode'] = 'scan'
    tviewer.params['scale_mode'] = 'same_for_all'
    #~ tviewer.params['mode'] = 'scroll'
    


    tfr_viewer = QTimeFreq()
    tfr_viewer.configure(with_user_dialog=True, nodegroup_friends=None)
    tfr_viewer.input.connect(dev.output)
    tfr_viewer.initialize()
    tfr_viewer.show()
    tfr_viewer.params['refresh_interval'] = 300
    tfr_viewer.params['timefreq', 'f_start'] = 1
    tfr_viewer.params['timefreq', 'f_stop'] = 100.
    tfr_viewer.params['timefreq', 'deltafreq'] = 5
    tfr_viewer.start()
    
    
    
    
    def ajust_yrange():
        oscope.auto_scale()
        tviewer.auto_scale()
        tviewer.params_controller.apply_ygain_zoom(.3)
    
    timer = QtCore.QTimer(interval=1000, singleShot=True)
    timer.timeout.connect(ajust_yrange)
    timer.start()
    
    def terminate():
        dev.stop()
        oscope.stop()
        peeler.stop()
        tviewer.stop()
        app.quit()
    
    app.exec_()
    
    
    
    
    
    
if __name__ =='__main__':
    setup_catalogue()
    
    tridesclous_onlinepeeler()


