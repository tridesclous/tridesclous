import pytest
import h5py
import os, tempfile
import numpy as np


from tridesclous import DataIO
from urllib.request import urlretrieve

def download_locust(trial_names = ['trial_01']):
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
    
    sigs_by_trials = []
    for trial_name in trial_names:
        sigs = np.array([hdf['Continuous_1'][trial_name][name][...] for name in ch_names]).transpose()
        sigs = (sigs.astype('float32') - 2**15.) / 2**15
        sigs_by_trials.append(sigs)
    
    sampling_rate = 15000.
    
    return sigs_by_trials, sampling_rate, ch_names



def test_dataio():
    if os.path.exists('datatest/data.h5'):
        os.remove('datatest/data.h5')
    dataio = DataIO(dirname = 'datatest')
    #~ print(data)
    #data from locust
    sigs_by_trials, sampling_rate, ch_names = download_locust(trial_names = ['trial_01', 'trial_02', 'trial_03'])
    
    
    for seg_num in range(3):
        sigs = sigs_by_trials[seg_num]
        dataio.append_signals_from_numpy(sigs, seg_num = seg_num,t_start = 0.+5*seg_num, sampling_rate =  sampling_rate,
                    already_hp_filtered = True, channels = ch_names)
    
    #~ print(data)
    #~ print(data.segments)
    #~ print(data.store)
    print(dataio.summary(level=0))
    print(dataio.summary(level=1))
    
    #~ assert data.get_signals(seg_num=0).shape == (431548, 4)
    #~ assert data.get_signals(seg_num=0, t_start=3.).shape==(386548, 4)
    #~ assert data.get_signals(seg_num=0, t_stop=5.).shape == (75000, 4)
    #~ assert data.get_signals(seg_num=0, t_start=3., t_stop = 5.).shape == (30000, 4)
    
    
def test_dataio_with_neo():
    if os.path.exists('datatest_neo/data.h5'):
        os.remove('datatest_neo/data.h5')
    dataio = DataIO(dirname = 'datatest_neo')
    
    import neo
    import quantities as pq
    blocks = neo.RawBinarySignalIO('Tem10c11.IOT').read(sampling_rate = 10.*pq.kHz,
                    t_start = 0. *pq.S, unit = pq.V, nbchannel = 16, bytesoffset = 0,
                    dtype = 'int16', rangemin = -10, rangemax = 10)
    
    channel_indexes = np.arange(14)
    dataio.append_signals_from_neo(blocks, channel_indexes = channel_indexes, 
                                already_hp_filtered = False)
    print(dataio.summary(level=1))

    
    
    
    
    
    
    
    
    
if __name__=='__main__':
    #~ test_dataio()
    test_dataio_with_neo()
    
    
    
    