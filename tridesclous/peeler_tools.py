from collections import OrderedDict, namedtuple
import numpy as np

from .cltools import HAVE_PYOPENCL

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


from .labelcodes import (LABEL_TRASH, LABEL_UNCLASSIFIED, LABEL_ALIEN)

LABEL_LEFT_LIMIT = -11
LABEL_RIGHT_LIMIT = -12
LABEL_MAXIMUM_SHIFT = -13

LABEL_NO_MORE_PEAK = -20

# good label are >=0

_dtype_spike = [('index', 'int64'), ('cluster_label', 'int64'), ('jitter', 'float64'),]

Spike = namedtuple('Spike', ('index', 'cluster_label', 'jitter'))


def make_prediction_on_spike_with_label(spike_index, spike_label, spike_jitter, dtype, catalogue):
    assert spike_label >= 0
    cluster_idx = catalogue['label_to_index'][spike_label]
    return make_prediction_one_spike(spike_index, cluster_idx, spike_jitter, dtype, catalogue)

def make_prediction_one_spike(spike_index, cluster_idx, spike_jitter, dtype, catalogue, long=True):
    if not catalogue['inter_sample_oversampling'] or spike_jitter is None or np.isnan(spike_jitter):
        if long:
            pos = spike_index + catalogue['n_left_long']
            pred = catalogue['centers0_long'][cluster_idx, :, :]
        else:
            pos = spike_index + catalogue['n_left']
            pred = catalogue['centers0'][cluster_idx, :, :]
    else:
        raise NotImplementedError # TODO propagate center_long to this section and peeler
        r = catalogue['subsample_ratio']
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
        
        pos, pred = make_prediction_on_spike_with_label(spikes[i]['index'], spikes[i]['cluster_label'], spikes[i]['jitter'], dtype, catalogue)
        
        peak_width_long = catalogue['centers0_long'].shape[1]
        
        if pos>=0 and  pos+peak_width_long<shape[0]:
            prediction[pos:pos+peak_width_long, :] += pred
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




