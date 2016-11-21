import numpy as np
import os
import sys
import inspect
from urllib.request import urlretrieve


# Now all file are in raw binary format

datasets_info = {
    'locust':{
        'url': 'https://raw.githubusercontent.com/tridesclous/tridesclous_datasets/master/locust/',
        'filenames': ['locust_trial_01.raw', 'locust_trial_02.raw'],
        'shape' : (-1,4),
        'dtype': 'int16',
        'sample_rate': 15000.,
    },
    
    'olfactory_bulb':{
        'url': 'https://raw.githubusercontent.com/tridesclous/tridesclous_datasets/master/olfactory_bulb/',
        'filenames': ['OB_file1.raw', 'OB_file2.raw', 'OB_file3.raw'],
        'shape' : (-1,16),
        'dtype': 'int16',
        'sample_rate': 10000.,
    }
}


def download_dataset(name='locust', localdir=None):
    assert name in datasets_info
    
    if localdir is None:
        # get path of the calling function see
        #http://stackoverflow.com/questions/11757801/get-the-file-of-the-function-one-level-up-in-the-stack
        localdir = os.path.abspath(os.path.dirname(inspect.getfile(sys._getframe(1))))
        if not os.access(localdir, os.W_OK):
            localdir = tempfile.gettempdir()
        localdir = os.path.join(localdir, name)
    
    if not os.path.exists(localdir):
        os.mkdir(localdir)
    
    info = datasets_info[name]
    
    filenames = info['filenames']
    
    for filename in filenames:
        localfile = os.path.join(localdir, filename)
        if not os.path.exists(localfile):
            distantfile = info['url'] + filename
            urlretrieve(distantfile, localfile)
    
    return localdir, filenames

def get_dataset(name='locust', localdir=None, seg_num=0):
    assert name in datasets_info
    
    localdir, filenames = download_dataset(name=name, localdir=localdir)
    filename = filenames[seg_num]
    
    sample_rate = datasets_info[name]['sample_rate']
    dtype = datasets_info[name]['dtype']
    shape = datasets_info[name]['shape']
    data = np.memmap(os.path.join(localdir, filename), dtype=dtype).reshape(shape)
    
    return data, sample_rate
    




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