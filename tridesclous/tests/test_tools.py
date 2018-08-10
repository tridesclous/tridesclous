import pandas as pd
import numpy as np
from tridesclous.tools import *
from tridesclous.probe_list import probe_list

from urllib.request import urlretrieve
import time

def test_get_median_mad():
    pass
    

def test_FifoBuffer():
    n = 5
    fifo = FifoBuffer((1024+64, n), dtype='int16')
    for i in range(4):
        data = np.tile(np.arange(1024)[:, None], (1, n))+i*1024
        fifo.new_chunk(data, 1024*(i+1))
    data2 = fifo.get_data(3008, 4096)
    assert np.all(data2[0,:]==3008)
    assert np.all(data2[-1,:]==4095)
    #~ print(data2)

def test_get_neighborhood():
    geometry = [[0,0], [0, 100], [100, 100]]
    radius_um = 120
    n = get_neighborhood(geometry, radius_um)
    print(n)


def test_fix_prb_file_py2():
    distantfile = 'https://raw.githubusercontent.com/spyking-circus/spyking-circus/master/probes/kampff_128.prb'
    prb_filename = 'kampff_128.prb'
    urlretrieve(distantfile, prb_filename)

    fix_prb_file_py2(prb_filename)


def re_construct_probe_list():
    construct_probe_list()

def show_construct_probe_list():
    for k, v in probe_list.items():
        print('*'*20)
        print('**', k)
        print('*'*20)
        for e in v:
            print(e)


def test_compute_cross_correlograms():

    n = 1000000

    sample_rate=10000.
    bin_size=0.001
    window_size=0.2
    #~ symmetrize=True
    
    spike_times = np.random.rand(n)*10000
    spike_times.sort()
    spike_indexes = (spike_times*sample_rate).astype('int64')
    
    spike_labels = np.random.randint(0,6, size=n)
    
    spike_segments = np.zeros(n, dtype='int64')
    spike_segments[n//2] = 1
    
    cluster_labels = np.unique(spike_labels)
    

    t0 = time.perf_counter()
    cgc, bins = compute_cross_correlograms(spike_indexes, spike_labels, spike_segments,
                cluster_labels, sample_rate, window_size, bin_size)
    t1 = time.perf_counter()
    
    print(cgc.shape)
    print('compute_cross_correlograms', t1-t0)


def test_int32_to_rgba():
    r, g, b, a = int32_to_rgba(2654789321)
    print(r, g, b, a)
    
def test_rgba_to_int32():
    v = rgba_to_int32(158,60,222, 201)
    print(v)
    

if __name__ == '__main__':
    #~ test_get_median_mad()
    #~ test_FifoBuffer()
    #~ test_get_neighborhood()
    #~ test_fix_prb_file_py2()
    re_construct_probe_list()
    #~ show_construct_probe_list()
    #~ test_compute_cross_correlograms()
    #~ test_int32_to_rgba()
    #~ test_rgba_to_int32()
    