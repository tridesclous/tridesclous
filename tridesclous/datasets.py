import numpy as np
import os
from urllib.request import urlretrieve


def read_bulbe_olfactive():
    """
    Dataset from Nathalie Buonvison BO.
    """
    localdir = os.path.dirname(__file__)
    data = np.memmap(os.path.join(localdir, 'Tem06c08.IOT'), dtype='int16').reshape(-1, 16)
    #~ data = (data.astype('float32') - 2**15.) / 2**15
    sample_rate = 10000.
    return data[:, :14], sample_rate



def get_dataset(name='BO'):
    if name=='BO':
        return read_bulbe_olfactive()
    


#~ def download_locust(trial_names = ['trial_01']):
    #~ name = 'locust20010201.hdf5'
    #~ distantfile = 'https://zenodo.org/record/21589/files/'+name
    #~ localdir = os.path.dirname(os.path.abspath(__file__))
    
    #~ if not os.access(localdir, os.W_OK):
        #~ localdir = tempfile.gettempdir()
    #~ localfile = os.path.join(os.path.dirname(__file__), name)
    
    #~ if not os.path.exists(localfile):
        #~ urlretrieve(distantfile, localfile)
    #~ hdf = h5py.File(localfile,'r')
    #~ ch_names = ['ch09','ch11','ch13','ch16']
    
    #~ sigs_by_trials = []
    #~ for trial_name in trial_names:
        #~ sigs = np.array([hdf['Continuous_1'][trial_name][name][...] for name in ch_names]).transpose()
        #~ sigs = (sigs.astype('float32') - 2**15.) / 2**15
        #~ sigs_by_trials.append(sigs)
    
    #~ sampling_rate = 15000.
    
    #~ return sigs_by_trials, sampling_rate, ch_names