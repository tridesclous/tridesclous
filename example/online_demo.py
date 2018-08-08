"""
This is a demo of tridesclous online.
It is based on pyacq framework.

In pyacq:
  * Manager is the big chief
  * Nodegroup is a process running somewhere on a PC (here on the same machine)
  * Node is an infinite loop for processing or a viewer

in tridesclous.online there are sevreal Nodes:
  * OnlinePeeler: this the peeler but for online data
  * OnlineTraceViewer: this is a viewer for dilplay processed signal and sorted spikes
  * OnlineWindow: some kind of integrated OnlinePeeler+OnlineTraceViewer with some action to
     construct the catalogue on the fly.


Here a demo.


"""

import numpy as np

import tridesclous as tdc
import tridesclous.online

import  pyqtgraph as pg
import pyacq



# get sigs
localdir, filenames, params = tdc.download_dataset(name='olfactory_bulb')
filename = filenames[0] #only first file
sigs = np.fromfile(filename, dtype=params['dtype']).reshape(-1, params['total_channel'])
sigs = sigs.astype('float32')
sample_rate = params['sample_rate']

# This will impact the latency
chunksize = 1024

# Here a convinient fonction to create a fake device in background
# by playing signal more or less at the good speed
man = pyacq.create_manager(auto_close_at_exit=True)
ng0 = man.create_nodegroup() # process in background
dev = tridesclous.online.make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup=ng0, chunksize=chunksize)


# The device have 16 channel, take some take 2 tetrodes only
# In fcat this dataset is absolutly not 2 tetrodes but for the demo, it is OK :)
channel_groups = {
    0 : {'channels': [5, 6, 7, 8]},
    1 : {'channels': [1, 2, 3, 4]},
}


# where catalogue will be saved
workdir = 'demo_onlinewindow'


# Qt Application
app = pg.mkQApp()



windows = []
    
# OnelinePeeler will occur in backgroun in a diffrent process
# and possibly on other machine to split up the workload
nodegroup_friends = [man.create_nodegroup() for _ in range(2)]

w = tridesclous.online.TdcOnlineWindow()
w.configure(channel_groups=channel_groups, chunksize=chunksize,
                workdir=workdir, nodegroup_friends=nodegroup_friends)
w.input.connect(dev.output)
w.initialize()

w.show()
w.start()
windows.append(w)

dev.start()


app.exec_()





