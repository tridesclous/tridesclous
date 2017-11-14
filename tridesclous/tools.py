import numpy as np
import sklearn.metrics.pairwise
import re
from urllib.request import urlretrieve
import zipfile
import os

def median_mad(data, axis=0):
    """
    Compute along axis the median and the mad.
    
    Arguments
    ----------------
    data : np.ndarray
    
    
    Returns
    -----------
    med: np.ndarray
    mad: np.ndarray
    
    
    """
    med = np.median(data, axis=axis)
    mad = np.median(np.abs(data-med),axis=axis)*1.4826
    return med, mad


def get_pairs_over_threshold(m, labels, threshold):
    """
    detect pairs over threhold in a similarity matrice
    """
    m = np.triu(m)
    ind0, ind1 = np.nonzero(m>threshold)
    
    #remove diag
    keep = ind0!=ind1
    ind0 = ind0[keep]
    ind1 = ind1[keep]
    
    pairs = list(zip(labels[ind0], labels[ind1]))
    
    return pairs
    


class FifoBuffer:
    """
    Kind of fifo on axis 0 than ensure to have the buffer and partial of previous buffer
    continuous in memory.
    
    This is not efficient if shape[0]is lot greater than chunksize.
    But if shape[0]=chunksize+smallsize, it should be OK.
    
    
    """
    def __init__(self, shape, dtype):
        self.buffer = np.zeros(shape, dtype=dtype)
        self.last_index = None
        
    def new_chunk(self, data, index):
        if self.last_index is not None:
            assert self.last_index+data.shape[0]==index
        
        n = self.buffer.shape[0]-data.shape[0]
        #roll the end
        
        self.buffer[:n] = self.buffer[-n:]
        self.buffer[n:] = data
        self.last_index = index
    
    def get_data(self, start, stop):
        start = start - self.last_index + self.buffer .shape[0]
        stop = stop - self.last_index + self.buffer .shape[0]
        assert start>=0
        assert stop<=self.buffer.shape[0]
        return self.buffer[start:stop]


def get_neighborhood(geometry, radius_um):
    """
    get neighborhood given geometry array and radius
    
    params
    -----
    geometry: numpy array (nb_channel, 2) intresect units ar micro meter (um)
    
    radius_um: radius in micro meter
    
    returns
    ----
    
    neighborhood: boolean numpy array (nb_channel, nb_channel)
    
    """
    d = sklearn.metrics.pairwise.euclidean_distances(geometry)
    return d<=radius_um
    

def fix_prb_file_py2(probe_filename):
    """
    prb file can be define in python2
    unfortunatly some of them are done with python
    and this is not working in python3
    range(0, 17) + range(18, 128)
    
    This script tryp to change range(...) by list(range(...)))
    """
    with open(probe_filename, 'rb') as f:
        prb = f.read()
    

    pattern = b"list\(range\([^()]*\)\)"
    already_ok = re.findall( pattern, prb)
    if len(already_ok)>0:
        return
    #~ print(already_ok)


    pattern = b"range\([^()]*\)"
    changes = re.findall( pattern, prb)
    for change in changes:
        prb = prb.replace(change, b'list('+change+b')')

    with open(probe_filename, 'wb') as f:
        f.write(prb)


def construct_probe_list():
    #this download probes list from klusta and circus
    # and put them in a file
    urls = [ 
                ('kwikteam', 'https://codeload.github.com/kwikteam/probes/zip/master', 'probes-master/'),
                ('spyking-circus', 'https://codeload.github.com/spyking-circus/spyking-circus/zip/master', 'spyking-circus-master/probes/'),
    ]
    
    #download zip from github
    probes = {}
    for name, url, path in urls:
        zip_name = url.split('/')[-1]
        urlretrieve(url, name+'.zip')
        
        with zipfile.ZipFile( name+'.zip') as zfile:
            probes[name] = []
            for f in zfile.namelist():
                if f.startswith(path) and f != path and not f.endswith('/'):
                    probes[name].append(f.replace(path, '').replace('.prb', ''))
    
    #generate .py with probe list
    with open('probe_list.py', 'w+') as f:
        f.write('# This file is generated do not modify!!!\n')
        f.write('probe_list = {\n')
        for name, url, path in urls:
            f.write('    #########\n')
            f.write('    "{}" : [\n'.format(name))
            for probe in probes[name]:
                f.write('        "{}",\n'.format(probe))
            f.write('    ],\n')
        f.write('}\n')

def download_probe(local_dirname, probe_name, origin='kwikteam'):
    if origin == 'kwikteam':
        #Max Hunter made a list of neuronexus probes, many thanks
        baseurl = 'https://raw.githubusercontent.com/kwikteam/probes/master/'
    elif origin == 'spyking-circus':
        # Pierre Yger made a list of various probe file, many thanks
        baseurl = 'https://raw.githubusercontent.com/spyking-circus/spyking-circus/master/probes/'
    else:
        raise(NotImplementedError)
    
    
    if not probe_name.endswith('.prb'):
        probe_name += '.prb'
    
    probe_filename = probe_name
    if '/' in probe_filename:
        probe_filename = probe_filename.split('/')[-1]
    probe_filename = os.path.join(local_dirname, probe_filename)
    
    urlretrieve(baseurl+probe_name, probe_filename)
    fix_prb_file_py2(probe_filename)#fix range to list(range
    
    return probe_filename
    


#~ if __name__=='__main__':
    #~ construct_probe_list()

