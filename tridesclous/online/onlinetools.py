import numpy as np
import pyqtgraph as pg
from pyacq.devices import NumpyDeviceBuffer
from tridesclous.gui.tools import get_dict_from_group_param
from tridesclous.gui.gui_params import preprocessor_params, peak_detector_params, clean_waveforms_params




preprocessor_params_default = get_dict_from_group_param(
                pg.parametertree.Parameter.create(name='',
                    type='group', children=preprocessor_params))


peak_detector_params_default = get_dict_from_group_param(
                pg.parametertree.Parameter.create(name='',
                    type='group', children=peak_detector_params))

clean_waveforms_params_default = get_dict_from_group_param(
                pg.parametertree.Parameter.create(name='',
                    type='group', children=clean_waveforms_params))



def make_pyacq_device_from_buffer(sigs, sample_rate, nodegroup = None, chunksize=1024):
    length, nb_channel = sigs.shape
    length -= length%chunksize
    sigs = sigs[:length, :]
    dtype = sigs.dtype
    channel_names = ['channel{}'.format(c) for c in range(sigs.shape[1])]
    
    if nodegroup is None:
        dev = NumpyDeviceBuffer()
    else:
        dev = nodegroup.create_node('NumpyDeviceBuffer')
    dev.configure(nb_channel=nb_channel, sample_interval=1./sample_rate, chunksize=chunksize, buffer=sigs, channel_names=channel_names)
    dev.output.configure(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
    dev.initialize()
    
    return dev


def make_empty_catalogue(chan_grp=0,
                channel_indexes=[],
                n_left=-20,
                n_right=40,
                internal_dtype='float32',
                
                preprocessor_params={},
                peak_detector_params={},
                clean_waveforms_params={},
                
                signals_medians = None,
                signals_mads = None,
                
                ):
    catalogue = {}
    
    catalogue['chan_grp'] = chan_grp
    catalogue['n_left'] = n_left
    catalogue['n_right'] = n_right
    catalogue['peak_width'] = catalogue['n_right'] - catalogue['n_left']
    
    catalogue['cluster_labels'] = np.array([], dtype='int64')
    catalogue['clusters'] = np.array([], dtype='int64')
    
    full_width = n_right - n_left
    nchan = len(channel_indexes)
    
    centers0 = np.zeros((0, full_width - 4, nchan), dtype=internal_dtype)
    centers1 = np.zeros_like(centers0)
    centers2 = np.zeros_like(centers0)
    catalogue['centers0'] = centers0
    catalogue['centers1'] = centers1
    catalogue['centers2'] = centers2
    
    subsample = np.arange(1.5, full_width-2.5, 1/20.)
    catalogue['subsample_ratio'] = 20
    interp_centers0 = np.zeros((0, subsample.size, nchan), dtype=internal_dtype)
    catalogue['interp_centers0'] = interp_centers0
    
    catalogue['label_to_index'] = {}

    #find max  channel for each cluster for peak alignement
    catalogue['max_on_channel'] = np.zeros_like(catalogue['cluster_labels'])
    
    
    preprocessor_params_ = dict(preprocessor_params_default)
    preprocessor_params_.update(preprocessor_params)
    preprocessor_params_.pop('chunksize')


    peak_detector_params_ = dict(peak_detector_params_default)
    peak_detector_params_.update(peak_detector_params)

    clean_waveforms_params_ = dict(clean_waveforms_params_default)
    clean_waveforms_params_.update(clean_waveforms_params)
    
    if signals_medians is None:
        signals_medians = signals_medians = np.zeros(nchan, dtype=internal_dtype)
    if signals_mads is None:
        signals_mads = signals_medians = np.ones(nchan, dtype=internal_dtype)
    
    #params
    catalogue['signal_preprocessor_params'] = preprocessor_params_
    catalogue['peak_detector_params'] = peak_detector_params_
    catalogue['clean_waveforms_params'] = clean_waveforms_params_
    catalogue['signals_medians'] = signals_medians
    catalogue['signals_mads'] = signals_mads
    
    catalogue['empty_catalogue'] = True # a key to detect real/fake catalogue
    
    
    return catalogue


def lighter_catalogue(catalogue):
    """
    The trace viewer need the catalogue for some fiew kargs.
    Since with pyacq rpc system all kargs are serializer since function
    make the catalogue lightened with only needed keys
    """
    lightened_catalogue = {}
    keys = ['clusters', 'cluster_labels', 'max_on_channel']
    for k in keys:
        lightened_catalogue[k] = catalogue[k]
    
    return lightened_catalogue

