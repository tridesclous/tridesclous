import pandas as pd
import numpy as np
from tridesclous.tools import median_mad, FifoBuffer, get_neighborhood, fix_prb_file_py2

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
    

if __name__ == '__main__':
    #~ test_get_median_mad()
    #~ test_FifoBuffer()
    #~ test_get_neighborhood()
    test_fix_prb_file_py2()