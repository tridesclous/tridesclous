import numpy as np
import sklearn.metrics.pairwise
import re
from urllib.request import urlretrieve
import zipfile
import os
import json

from . import labelcodes

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
            assert self.last_index+data.shape[0]==index, 'FifoBuffer self.last_index+data.shape[0]==index {} {}'.format(self.last_index+data.shape[0], index)
        
        assert data.shape[0]<=self.buffer.shape[0]
        n = self.buffer.shape[0]-data.shape[0]
        assert n>0
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
    
    def reset(self):
        self.last_index = None
        self.buffer[:] = 0


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

def open_prb(probe_filename):
    d = {}
    with open(probe_filename) as f:
        exec(f.read(), None, d)
    channel_groups = d['channel_groups']
    return channel_groups

read_prb = open_prb

def create_prb_file_from_dict(channel_groups, filename):
    # transform array to list
    #~ channel_groups_ = {}
    #~ for chan_grp, channel_group in channel_groups.items():
        #~ channel_groups_[chan_grp] = dict(channel_group)
        #~ if isinstance(channel_group['channels'], np.ndarray):
            #~ channel_groups_[chan_grp]['channels'] = channel_group['channels'].tolist()
    
    # write with hack on json to put key as inteteger (normally not possible in json)
    with open(filename, 'w', encoding='utf8') as f:
        txt = json.dumps(channel_groups,indent=4)
        for chan_grp in channel_groups.keys():
            txt = txt.replace('"{}":'.format(chan_grp), '{}:'.format(chan_grp))
            for chan in channel_groups[chan_grp]['channels']:
                txt = txt.replace('"{}":'.format(chan), '{}:'.format(chan))
        txt = 'channel_groups = ' +txt
        f.write(txt)


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
    """
    Download prb file from either kwikteam or spyking-circus github.
    """
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
    

def compute_cross_correlograms(spike_indexes, spike_labels, 
                spike_segments, cluster_labels, sample_rate,
                window_size=0.1, bin_size=0.001, symmetrize=False,
                check_sorted=False):
    """
    Compute several cross-correlogram in one course
    from sevral cluster.
    
    This very elegant implementation is copy from phy package
    written by Cyril Rossant.
    
    
    Some sligh modification have been made to fit tdc datamodel because
    there are several segment handling in tdc.
    
    """
    assert sample_rate > 0.

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds
    binsize = int(sample_rate * bin_size)  # in samples
    assert binsize >= 1
    real_bin_size = binsize/sample_rate

    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds
    winsize_bins = 2 * int(.5 * window_size / bin_size) + 1

    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1

    # Take the cluster oder into account.
    cluster_labels = cluster_labels.copy()
    cluster_labels.sort()
    
    # Like spike_labels, but with 0..n_clusters-1 indices.
    spike_labels_i = np.searchsorted(cluster_labels, spike_labels)
    
    cluster_labels = np.array(cluster_labels)
    n_clusters = cluster_labels.size
    correlograms = np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype='int32')
    
    nb_seg = np.max(spike_segments) + 1
    for seg_num in range(nb_seg):
    
        # Shift between the two copies of the spike trains.
        shift = 1

        
        keep = (spike_segments==seg_num) & np.isin(spike_labels, cluster_labels)
        sp_indexes = spike_indexes[keep]
        sp_labels = spike_labels[keep]
        sp_segments = spike_segments[keep]
        sp_labels_i = spike_labels_i[keep]
        if check_sorted:
            # theprevent bugs if spike vector is not sorted
            # could append in peeler if some case (too bad!!!)
            order = np.argsort(sp_indexes)
            sp_indexes = sp_indexes[order]
            sp_labels = sp_labels[order]
            sp_segments = sp_segments[order] 
            sp_labels_i = sp_labels_i[order]

        # At a given shift, the mask precises which spikes have matching spikes
        # within the correlogram time window.
        mask = np.ones_like(sp_indexes, dtype="bool")

        # The loop continues as long as there is at least one spike with
        # a matching spike.
        while mask[:-shift].any():
            # Number of time samples between spike i and spike i+shift.
            #~ spike_diff = _diff_shifted(spike_indexes, shift)
            spike_diff = sp_indexes[shift:] - sp_indexes[:len(sp_indexes) - shift]
            
            # Binarize the delays between spike i and spike i+shift.
            spike_diff_b = spike_diff // binsize

            # Spikes with no matching spikes are masked.
            mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

            # Cache the masked spike delays.
            m = mask[:-shift].copy()
            d = spike_diff_b[m]

            # Find the indices in the raveled correlograms array that need
            # to be incremented, taking into account the spike clusters.
            indices = np.ravel_multi_index((sp_labels_i[:-shift][m],
                                            sp_labels_i[+shift:][m],
                                            d),
                                           correlograms.shape)

            # Increment the matching spikes in the correlograms array.
            bbins = np.bincount(indices)
            correlograms.ravel()[:len(bbins)] += bbins
            
            
            shift += 1

        # Remove ACG peaks.
        correlograms[np.arange(n_clusters),
                 np.arange(n_clusters),
                 0] = 0
        
    if symmetrize:
        # We symmetrize c[i, j, 0].
        # This is necessary because the algorithm in correlograms()
        # is sensitive to the order of identical spikes.
        correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                          correlograms[..., 0].T)
        sym = correlograms[..., 1:][..., ::-1]
        sym = np.transpose(sym, (1, 0, 2))
        correlograms = np.dstack((sym, correlograms))
        
        bins = np.arange(correlograms.shape[2]+1)*real_bin_size - real_bin_size*winsize_bins/2.

    else:
        bins = np.arange(correlograms.shape[2]+1)*real_bin_size - real_bin_size/2.
    
    return correlograms, bins
    


def get_color_palette(n, palette='husl', output='int32'):
    # this depend now on seaborn but seaborn will be renmoved as dependency soon
    # because it break joblib
    import seaborn as sns
    
    if output == 'rgb':
        colors = sns.color_palette(palette, n)
        return colors
    elif output == 'int32':
        colors_int32 = np.array([rgba_to_int32(r,g,b) for r,g,b in sns.color_palette(palette, n)])
        return colors_int32
    else:
        raise NotImplementedError


def int32_to_rgba(v, mode='int'):
    r = (v>>24) & 0xFF
    g = (v>>16) & 0xFF
    b = (v>>8) & 0xFF
    a = (v>>0) & 0xFF
    if mode == 'int':
        return r, g, b, a
    elif mode=='float':
        return r/255., g/255., b/255., a/255.


def rgba_to_int32(r, g, b, a=None):
    if type(r) == int:
        if a is None:
            a = 255
        #ensure max255
        r, g, b = (r & 0xFF), (g & 0xFF), (b & 0xFF)
        v = (r<<24) + (g<<16) + (b<<8) + a
    else:
        if a is None:
            a = 1.
        v = (int(r*255.)<<24) + (int(g*255.)<<16) + (int(b*255.)<<8) + int(a*255.)
    return v
        
def make_color_dict(clusters):
    colors = {}
    for cluster in clusters:
        r, g, b, a = int32_to_rgba(cluster['color'], mode='float')
        colors[cluster['cluster_label']] =  (r, g, b)
    colors[labelcodes.LABEL_TRASH] = (.4, .4, .4)
    colors[labelcodes.LABEL_UNCLASSIFIED] = (.6, .6, .6)
    colors[labelcodes.LABEL_NOISE] = (.8, .8, .8)
    colors[labelcodes.LABEL_ALIEN] = (.4, .8, .1)
    colors[labelcodes.LABEL_NO_WAVEFORM] = (.6, .6, .6)
    
    
    return colors



#~ if __name__=='__main__':
    #~ construct_probe_list()

