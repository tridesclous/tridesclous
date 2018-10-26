import numpy as np
import os
import sys
import inspect
from urllib.request import urlretrieve


# For now, all testing file are in raw binary format
datasets_info = {
    'locust':{
        'url': 'https://raw.githubusercontent.com/tridesclous/tridesclous_datasets/master/locust/',
        'filenames': ['locust_trial_01.raw', 'locust_trial_02.raw'],
        'total_channel': 4,
        'dtype': 'int16',
        'sample_rate': 15000.,
        'bit_to_microVolt' : None, # TODO find it
    },
    
    'olfactory_bulb':{
        'url': 'https://raw.githubusercontent.com/tridesclous/tridesclous_datasets/master/olfactory_bulb/',
        'filenames': ['OB_file1.raw', 'OB_file2.raw', 'OB_file3.raw'],
        'total_channel': 16,
        'dtype': 'int16',
        'sample_rate': 10000.,
        'channel_group': list(range(14)),
        'bit_to_microVolt' : 20. * 1000./2**16, # This is not sure, but the order of magnitude is OK.
    },

    'purkinje':{
        'url': 'https://raw.githubusercontent.com/tridesclous/tridesclous_datasets/master/purkinje/',
        'filenames': ['purkinje_extra_cellular.raw'],
        'total_channel': 4,
        'dtype': 'float32',
        'sample_rate': 15000.,
        'bit_to_microVolt' : None, # TODO find it
    },
    'striatum_rat': {
        'url': 'https://raw.githubusercontent.com/tridesclous/tridesclous_datasets/master/striatum_rat/',
        'filenames': ['tetrode_striatum.dat'],
        'total_channel': 4,
        'dtype': 'int16',
        'sample_rate': 20000.,
        'prb_filename' : 'tetrode_striatum.prb',
        'bit_to_microVolt' : 0.195, # This is not sure, must be checked
    },
    
    
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
    if 'prb_filename' in info:
        filenames = filenames + [ info['prb_filename'] ]
    
    for filename in filenames:
        #~ print(filename)
        localfile = os.path.join(localdir, filename)
        
        if not os.path.exists(localfile):
            distantfile = info['url'] + filename
            print('download_dataset', localfile, 'from', distantfile)
            urlretrieve(distantfile, localfile)
    
    filenames = [os.path.join(localdir, f) for f in info['filenames']]
    params = {k: datasets_info[name][k] for k in ('dtype', 'sample_rate', 'total_channel', 'bit_to_microVolt')}
    return localdir, filenames, params


def get_dataset(name='locust', localdir=None, seg_num=0):
    assert name in datasets_info
    
    localdir, filenames, params = download_dataset(name=name, localdir=localdir)
    filename = filenames[seg_num]
    data = np.memmap(os.path.join(localdir, filename), dtype=params['dtype'])
    
    data = data.reshape(-1, params['total_channel'])
    
    info = datasets_info[name]
    
    if 'channel_group' in info:
        data = data[:, datasets_info[name]['channel_group']]
    
    return data, params['sample_rate']
    

