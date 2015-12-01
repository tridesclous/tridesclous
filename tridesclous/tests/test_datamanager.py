import pytest
import h5py
import os, tempfile
import numpy as np


from tridesclous import DataManager
from urllib.request import urlretrieve

def download_locust():
    name = 'locust20010201.hdf5'
    distantfile = 'https://zenodo.org/record/21589/files/'+name
    localdir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.access(localdir, os.W_OK):
        localdir = tempfile.gettempdir()
    localfile = os.path.join(os.path.dirname(__file__), name)
    
    if not os.path.exists(localfile):
        urlretrieve(distantfile, localfile)
    hdf = h5py.File(localfile,'r')
    ch_names = ['ch09','ch11','ch13','ch16']
    sigs = np.array([hdf['Continuous_1']['trial_01'][name][...] for name in ch_names]).transpose()
    sigs = (sigs.astype('float32') - 2**15.) / 2**15
    
    return sigs



def test_spikesorter():
    if os.path.exists('test/data.h5'):
        os.remove('test/data.h5')
    data = DataManager(dirname = 'test')
    #~ print(data)
    #data from locust
    signals = download_locust()
    
    #~ for seg_num in range(3):
        #~ data.append_signals(signals, seg_num = seg_num,t_start = 0.+5*seg_num, sampling_rate =  15000., already_hp_filtered = True)
    
    #~ print(data)
    #~ print(data.segments)
    #~ print(data.store)
    print(data.summary(level=0))
    print(data.summary(level=1))
    
    #~ assert data.get_signals(seg_num=0).shape == (431548, 4)
    #~ assert data.get_signals(seg_num=0, t_start=3.).shape==(386548, 4)
    #~ assert data.get_signals(seg_num=0, t_stop=5.).shape == (75000, 4)
    #~ assert data.get_signals(seg_num=0, t_start=3., t_stop = 5.).shape == (30000, 4)
    
    
    
    
    
    
if __name__=='__main__':
    test_spikesorter()
    
    
    