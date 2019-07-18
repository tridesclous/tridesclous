from collections import OrderedDict, namedtuple
import numpy as np

from .cltools import HAVE_PYOPENCL

from .labelcodes import (LABEL_TRASH, LABEL_UNCLASSIFIED, LABEL_ALIEN)

LABEL_LEFT_LIMIT = -11
LABEL_RIGHT_LIMIT = -12
LABEL_MAXIMUM_SHIFT = -13
# good label are >=0

_dtype_spike = [('index', 'int64'), ('cluster_label', 'int64'), ('jitter', 'float64'),]

Spike = namedtuple('Spike', ('index', 'cluster_label', 'jitter'))


def make_prediction_one_spike(spike_index, spike_label, spike_jitter, dtype, catalogue):
    assert spike_label >= 0
    cluster_idx = catalogue['label_to_index'][spike_label]
    r = catalogue['subsample_ratio']
    pos = spike_index + catalogue['n_left']
    #TODO debug that sign
    shift = -int(np.round(spike_jitter))
    pos = pos + shift
    int_jitter = int((spike_jitter+shift)*r) + r//2
    pred = catalogue['interp_centers0'][cluster_idx, int_jitter::r, :]
    
    return pos, pred


def make_prediction_signals(spikes, dtype, shape, catalogue, safe=True):
    #~ n_left, peak_width, 
    
    prediction = np.zeros(shape, dtype=dtype)
    for i in range(spikes.size):
        k = spikes[i]['cluster_label']
        if k<0: continue
        
        pos, pred = make_prediction_one_spike(spikes[i]['index'], spikes[i]['cluster_label'], spikes[i]['jitter'], dtype, catalogue)
        
        if pos>=0 and  pos+catalogue['peak_width']<shape[0]:
            prediction[pos:pos+catalogue['peak_width'], :] += pred
        else:
            if not safe:
                print(spikes)
                n_left = catalogue['n_left']
                width = catalogue['peak_width']
                local_pos = spikes['index'] - np.round(spikes['jitter']).astype('int64') + n_left
                print(local_pos)
                #~ spikes['LABEL_LEFT_LIMIT'][(local_pos<0)] = LABEL_LEFT_LIMIT
                print('LEFT', (local_pos<0))
                #~ spikes['cluster_label'][(local_pos+width)>=shape[0]] = LABEL_RIGHT_LIMIT
                print('LABEL_RIGHT_LIMIT', (local_pos+width)>=shape[0])
                
                print('i', i)
                print(dtype, shape, catalogue['n_left'], catalogue['peak_width'], pred.shape)
                raise(ValueError('Border error {} {} {} {} {}'.format(pos, catalogue['peak_width'], shape, jitter, spikes[i])))
                
        
    return prediction


def get_auto_params_for_peelers(dataio, chan_grp=0):
    nb_chan = dataio.nb_channel(chan_grp=chan_grp)
    params = {}
    
    if nb_chan <=8:
        params['use_sparse_template'] = False
        params['sparse_threshold_mad'] = 1.5
        params['use_opencl_with_sparse'] = False
    else:
        params['use_sparse_template'] = True
        params['sparse_threshold_mad'] = 1.5
        params['use_opencl_with_sparse'] = HAVE_PYOPENCL

    return params
    
    
