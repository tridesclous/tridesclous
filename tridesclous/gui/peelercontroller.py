from .myqt import QT
import pyqtgraph as pg

import numpy as np
#import seaborn as sns


from .base import ControllerBase

from .. import labelcodes
from ..tools import make_color_dict

from ..peeler import _dtype_spike

spike_visible_modes = ['selected', 'all',  'collision']

#~ _dtype_spike = [('index', 'int64'), ('cluster_label', 'int64'), ('jitter', 'float64'),]

_dtype_complement = [('cell_label', 'int64'), ('channel', 'int64'), ('segment', 'int64'), ('visible', 'bool'),
                ('selected', 'bool')]


class PeelerController(ControllerBase):
    def __init__(self, parent=None, dataio=None, catalogue=None):
        ControllerBase.__init__(self, parent=parent)
        self.dataio = dataio
        self.catalogue = catalogue
        
        self.chan_grp = catalogue['chan_grp']
        self.nb_channel = self.dataio.nb_channel(self.chan_grp)
        self.channels = np.arange(self.nb_channel, dtype='int64')
        
        self.init_plot_attributes()
        self.update_visible_spikes()
    
    def init_plot_attributes(self):
        #concatenate all spikes for all segments
        all_spikes = []
        for i in range(self.dataio.nb_segment):
            local_spikes = self.dataio.get_spikes(seg_num=i, chan_grp=self.chan_grp)
            spikes = np.zeros(local_spikes.shape, dtype=_dtype_spike+_dtype_complement)
            for k, _ in _dtype_spike:
                spikes[k] = local_spikes[k]
            spikes['segment'] = i
            spikes['visible'] = spikes['cluster_label']>=0
            spikes['selected'] = False
            
            
            clusters = self.catalogue['clusters']
            cluster_labels = clusters['cluster_label']
            cell_labels = clusters['cell_label']
            
            #set cell_label <0 are the same >0 are converted for 'clusters' table in catalogue
            mask = spikes['cluster_label']<0
            spikes['cell_label'][mask] = spikes['cluster_label'][mask]
            
            mask = spikes['cluster_label']>=0
            # searchsorter wotk with ordered
            clus = clusters[clusters['cluster_label']>=0]
            order = np.argsort(clus['cluster_label'])
            clus = clus[order].copy()
            spike_cluster_index = np.searchsorted(clus['cluster_label'], spikes['cluster_label'][mask].copy())
            spikes['cell_label'][mask] = clus['cell_label'][spike_cluster_index]
            
            all_spikes.append(spikes)
        self._spikes = np.concatenate(all_spikes)
        
        self.nb_spike = int(self.spikes.size)
        
        self.cluster_labels = self.catalogue['clusters']['cluster_label']
        
        # set channel for each cluster
        for i, k in enumerate(self.cluster_labels):
            #~ chan = self.catalogue['extremum_channel'][i]
            chan = self.catalogue['clusters']['extremum_channel'][i]
            sel = self.spikes['cluster_label'] == k
            self.spikes['channel'][sel] = chan
        
        # compute sparse mask
        # TODO : put this mask in catalogue directly
        self.sparse_threshold = 1.5
        self.centroids_sparse_mask = np.zeros((self.cluster_labels.size, self.nb_channel), dtype='bool')
        for k in self.cluster_labels:
            ind = self.catalogue['label_to_index'][k] 
            median = self.catalogue['centers0'][ind, :, :]
            self.centroids_sparse_mask[ind, :] = np.any(np.abs(median) > self.sparse_threshold, axis=0)
        
        #~ self.cluster_labels = np.unique(self.spikes['cluster_label'])#TODO take from catalogue
        
        
        self.cluster_count = { k:np.sum(self.spikes['cluster_label']==k) for k in self.cluster_labels}
        
        
        self.cluster_visible = {k:k>=0 for k  in self.cluster_labels}

        
        #qt colors
        self.colors = make_color_dict(self.catalogue['clusters'])
        self.qcolors = {}
        for k, color in self.colors.items():
            r, g, b = color
            self.qcolors[k] = QT.QColor(int(r*255), int(g*255), int(b*255))
        
        self.spike_visible_mode = spike_visible_modes[0]
    
    def check_plot_attributes(self):
        for k in self.cluster_labels:
            if k not in self.cluster_visible:
                self.cluster_visible[k] = k>=0
        
        for k in list(self.cluster_visible.keys()):
            if k not in self.cluster_labels:
                self.cluster_visible.pop(k)
        
        #~ self.refresh_colors(reset=False)
    
    @property
    def have_sparse_template(self):
        return True

    @property
    def spikes(self):
        return self._spikes

    @property
    def spike_selection(self):
        return self.spikes['selected']

    @property
    def spike_segment(self):
        return self.spikes['segment']

    @property
    def spike_index(self):
        return self.spikes['index']

    @property
    def spike_channel(self):
        return self.spikes['index']

    @property
    def positive_cluster_labels(self):
        cluster_labels = self.clusters['cluster_label']
        return cluster_labels[cluster_labels>=0]

    @property
    def clusters(self):
        return self.catalogue['clusters']
    
    def get_waveforms_shape(self):
        shape = self.catalogue['centers0'].shape[1:]
        return shape
    
    def get_sparse_channels(self, label):
        ind = self.catalogue['label_to_index'][label] 
        chans,  = np.nonzero(self.centroids_sparse_mask[ind, :])
        return chans
    
    def get_common_sparse_channels(self, labels):
        inds = [self.catalogue['label_to_index'][label]  for label in labels]
        chans,  = np.nonzero(np.any(self.centroids_sparse_mask[inds, :], axis=0))
        return chans
    
    def get_waveform_centroid(self, label, metric, sparse=False, channels=None):
        if metric in ('mean', 'std', 'mad'):
            return None, None
        

        if label in self.catalogue['label_to_index']:
            i = self.catalogue['label_to_index'][label]
            wf = self.catalogue['centers0'][i, :, :].copy()

            if sparse:
                assert channels is None
                chans = self.get_sparse_channels(label)
                wf = wf[:, chans].copy()
            elif channels is not None:
                chans = channels
                wf = wf[:, chans].copy()
            else:
                chans = self.channels
                wf = wf.copy()
            
            return wf, chans
        else:
            return None, None

    def get_min_max_centroids(self):
        if self.catalogue['centers0'].shape[0]>0:
            wf_min = np.min(self.catalogue['centers0'])
            wf_max = np.max(self.catalogue['centers0'])
        else:
            wf_min = 0.
            wf_max = 0.
        return wf_min, wf_max

    def get_waveform_left_right(self):
        return self.catalogue['n_left'], self.catalogue['n_right']
    
    def get_threshold(self):
        threshold = self.catalogue['peak_detector_params']['relative_threshold']
        if self.catalogue['peak_detector_params']['peak_sign']=='-':
            threshold = -threshold
        return threshold

    def get_extremum_channel(self, label):
        if label in self.catalogue['label_to_index']:
            cluster_idx = self.catalogue['label_to_index'][label]
            #~ c = self.catalogue['extremum_channel'][cluster_idx]
            c = self.catalogue['clusters']['extremum_channel'][cluster_idx]
            return c
    
    def get_some_waveforms(self, seg_nums, peak_sample_indexes, channel_indexes):
        n_left, n_right = self.get_waveform_left_right()
        waveforms = self.dataio.get_some_waveforms(seg_nums=seg_nums, chan_grp=self.chan_grp, 
                            peak_sample_indexes=peak_sample_indexes,
                            n_left=n_left, n_right=n_right, channel_indexes=channel_indexes)
        return waveforms

    def change_spike_visible_mode(self, mode):
        assert mode in spike_visible_modes
        #~ print(mode)
        self.spike_visible_mode = mode
        
    def update_visible_spikes(self):
        #~ print('update_visible_spikes', self.spike_visible_mode)
        #~ ['selected', 'all',  'collision']
        if self.spike_visible_mode=='selected':
            visibles = np.array([k for k, v in self.cluster_visible.items() if v ])
            self.spikes['visible'][:] = np.isin(self.spikes['cluster_label'], visibles)
        elif self.spike_visible_mode=='all':
            self.spikes['visible'][:] = True
        elif self.spike_visible_mode=='collision':
            self.spikes['visible'][:] = False
            d = np.diff(self.spikes['index'])
            labels0 = self.spikes['cluster_label'][:-1]
            labels1 = self.spikes['cluster_label'][1:]
            mask = (d>0) & (d< int(self.dataio.sample_rate * 0.001) ) & (labels0>0) & (labels1>0)
            ind, = np.nonzero(mask)
            self.spikes['visible'][ind] = True
            self.spikes['visible'][ind+1] = True
            
            
            
    
    def on_cluster_visibility_changed(self):
        #~ print('on_cluster_visibility_changed')
        self.update_visible_spikes()
        ControllerBase.on_cluster_visibility_changed(self)
