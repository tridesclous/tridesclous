import numpy as np
import os

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
    
    