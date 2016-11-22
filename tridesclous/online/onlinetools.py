import numpy as np

from pyacq.devices import NumpyDeviceBuffer


def make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup = None, chunksize=1024):
    length, nb_channel = sigs.shape
    length -= length%chunksize
    sigs = sigs[:length, :]
    dtype = sigs.dtype
    
    if nodegroup is None:
        dev = NumpyDeviceBuffer()
    else:
        dev = nodegroup.create_node('NumpyDeviceBuffer')
    dev.configure(nb_channel=nb_channel, sample_interval=1./sample_rate, chunksize=chunksize, buffer=sigs)
    dev.output.configure(protocol='tcp', interface='127.0.0.1', transfermode='plaindata', dtype=dtype)
    dev.initialize()
    
    return dev
    