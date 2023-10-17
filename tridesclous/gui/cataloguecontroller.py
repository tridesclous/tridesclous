from .myqt import QT
import pyqtgraph as pg


import numpy as np
#import seaborn as sns

from ..tools import median_mad, rgba_to_int32
from .. import labelcodes
from ..catalogueconstructor import _persistent_metrics
from .base import ControllerBase


import time




class CatalogueController(ControllerBase):
    
    
    def __init__(self, catalogueconstructor=None,parent=None):
        ControllerBase.__init__(self, parent=parent)
        
        self.cc = self.catalogueconstructor = catalogueconstructor
        self.dataio = catalogueconstructor.dataio
        self.nb_channel = self.dataio.nb_channel(self.chan_grp)
        self.channels = np.arange(self.nb_channel, dtype='int64')

        self.init_plot_attributes()
    
    def get_catalogueconstructor_attr(self, name):
        return getattr(self.cc, name)
    
    def reload_data(self):
        self.cc.reload_data()
        self.init_plot_attributes()
    
    def init_plot_attributes(self):
        self.cluster_visible = {k:i<20 for i, k  in enumerate(self.cluster_labels)}
        self.do_cluster_count()
        self.spike_selection = np.zeros(self.cc.nb_peak, dtype='bool')
        self.spike_visible = np.ones(self.cc.nb_peak, dtype='bool')
        self.refresh_colors(reset=False)
        self.check_plot_attributes()
        
        if self.cc.centroids_median is None:
            self.cc.compute_all_centroid()
        
        
        

    def check_plot_attributes(self):
        #cluster visibility
        for k in self.cluster_labels:
            if k not in self.cluster_visible:
                self.cluster_visible[k] = True
        for k in list(self.cluster_visible.keys()):
            if k not in self.cluster_labels and k>=0:
                self.cluster_visible.pop(k)
        
        if labelcodes.LABEL_NOISE not in self.cluster_visible:
            #~ print('self.cluster_visible[labelcodes.LABEL_NOISE] = True')
            self.cluster_visible[labelcodes.LABEL_NOISE] = True
        
        #~ if labelcodes.LABEL_NOISE not in self.cluster_count:
        #~ if self.cc.some_noise_snippet is not None:
            #~ self.cluster_count[labelcodes.LABEL_NOISE] = self.cc.some_noise_snippet.shape[0]
        #~ else:
            #~ self.cluster_count[labelcodes.LABEL_NOISE] = 0
        
        self.refresh_colors(reset=False)
        self.do_cluster_count()
    
    def do_cluster_count(self):
        self.cluster_count = { c['cluster_label']:c['nb_peak'] for c in self.clusters}
        
        if self.cc.some_noise_snippet is not None:
            self.cluster_count[labelcodes.LABEL_NOISE] = self.cc.some_noise_snippet.shape[0]
        else:
            self.cluster_count[labelcodes.LABEL_NOISE] = 0
    
    @property
    def have_sparse_template(self):
        return self.cc.mode == 'sparse'
    
    #map some attribute
    @property
    def chan_grp(self):
        return self.cc.chan_grp

    @property
    def spikes(self):
        return self.cc.all_peaks
    
    @property
    def clusters(self):
        return self.cc.clusters
    
    @property
    def cluster_labels(self):
        return self.cc.clusters['cluster_label']
    
    @property
    def positive_cluster_labels(self):
        return self.cc.positive_cluster_labels
    
    @property
    def cell_labels(self):
        return self.cc.clusters['cell_label']
        
    @property
    def spike_index(self):
        return self.cc.all_peaks['index']
    
    @property
    def spike_label(self):
        return self.cc.all_peaks['cluster_label']
    
    @property
    def spike_channel(self):
        return self.cc.all_peaks['channel']

    @property
    def spike_segment(self):
        return self.cc.all_peaks['segment']
    
    @property
    def some_features(self):
        return self.cc.some_features
    
    @property
    def some_peaks_index(self):
        return self.cc.some_peaks_index

    #~ @property
    #~ def some_waveforms(self):
        #~ return self.cc.some_waveforms
    
    #~ def get_waveforms_shape(self):
        #~ if self.cc.some_waveforms is not None:
            #~ shape = self.cc.some_waveforms.shape[1:]
            #~ return shape

    def get_waveform_left_right(self):
        if 'extract_waveforms' in self.cc.info:
            d = self.cc.info['extract_waveforms']
            return d['n_left'], d['n_right']
        else:
            return None, None
    
    def get_peak_sign(self):
        d = self.cc.info['peak_detector']
        return d['peak_sign']
    
    def get_some_waveforms(self, seg_nums, peak_sample_indexes, channel_indexes):
        n_left, n_right = self.get_waveform_left_right()
        waveforms = self.dataio.get_some_waveforms(seg_nums=seg_nums, chan_grp=self.chan_grp, 
                            peak_sample_indexes=peak_sample_indexes,
                            n_left=n_left, n_right=n_right, channel_indexes=channel_indexes)
        return waveforms
    
    def get_sparse_channels(self, label):
        ind = self.cc.index_of_label(label)
        chans,  = np.nonzero(self.cc.centroids_sparse_mask[ind, :])
        return chans

    def get_common_sparse_channels(self, labels):
        if -1 in labels:
            chans = self.channels
        elif len(labels) == 0:
            chans = self.channels
        else:
            inds = [self.cc.index_of_label(label) for label in labels]
            chans,  = np.nonzero(np.any(self.cc.centroids_sparse_mask[inds, :], axis=0))
        return chans
    
    def get_waveform_centroid(self, label, metric, sparse=False, channels=None):
        if label in self.cc.clusters['cluster_label'] and self.cc.centroids_median is not None:
            ind = self.cc.index_of_label(label)
            attr = getattr(self.cc, 'centroids_'+metric)
            wf = attr[ind, :, :]
            if sparse:
                assert channels is None
                chans = self.get_sparse_channels(label)
                wf = wf[:, chans]
            elif channels is not None:
                chans = channels
                wf = wf[:, chans]
            else:
                chans = self.channels
            
            return wf, chans
        else:
            return None, None

    def get_min_max_centroids(self):
        if self.cc.centroids_median is not None and self.cc.centroids_median.size>0:
            wf_min = self.cc.centroids_median.min()
            wf_max = self.cc.centroids_median.max()
        else:
            wf_min = 0.
            wf_max = 0.
        return wf_min, wf_max


    @property
    def some_noise_index(self):
        return self.cc.some_noise_index

    @property
    def some_noise_snippet(self):
        return self.cc.some_noise_snippet
    
    @property
    def some_noise_features(self):
        return self.cc.some_noise_features
    
    #~ @property
    #~ def centroids(self):
        #~ return self.cc.centroids

    @property
    def info(self):
        return self.cc.info
    
    @property        
    def geometry(self):
        return self.cc.geometry

    @property        
    def channel_to_features(self):
        return self.cc.channel_to_features
    
    def change_spike_label(self, mask, label):
        self.cc.change_spike_label(mask, label)
        self.check_plot_attributes()
        
    def get_threshold(self):
        threshold = self.cc.info['peak_detector']['relative_threshold']
        if self.cc.info['peak_detector']['peak_sign']=='-':
            threshold = -threshold
        return threshold
    
    def get_extremum_channel(self, label):
        if label<0:
            return None
        
        ind,  = np.nonzero(self.cc.clusters['cluster_label']==label)
        if ind.size!=1:
            return None
        ind = ind[0]
        
        extremum_channel = self.cc.clusters['extremum_channel'][ind]
        if extremum_channel>=0:
            return extremum_channel
        else:
            return None
        
        #~ if label in self.centroids:
            #~ c = self.centroids[label]['extremum_channel']
            #~ return c
    
    def on_new_cluster(self, label_changed=None):
        """
        label_changed can be remove/add/modify
        """
        print('!!!!!!! controller.on_new_cluster')
        # TODO simplify this should not be called anymore...)
        
        if len(self.cc.all_peaks['index']) == 0:
            return
        
        self.cc.on_new_cluster()
        
        self.check_plot_attributes()
        
        self.cc.compute_centroid(label_changed=label_changed)
        
        #reset some metrics
        for name in _persistent_metrics:
            setattr(self.cc, name, None)
        
        self.check_plot_attributes()

    
    def refresh_colors(self, reset=True, palette = 'husl'):
        self.cc.refresh_colors(reset=reset, palette=palette)
        
        self.qcolors = {}
        for k, color in self.cc.colors.items():
            r, g, b = color
            self.qcolors[k] = QT.QColor(int(r*255), int(g*255), int(b*255))

    def set_cluster_attributes(self, label, color=None, annotations=None, tag=None):
        if label not in self.cc.clusters['cluster_label']:
            return
        
        clusters = self.cc.clusters
        ## ind = np.searchsorted(clusters['cluster_label'], label)   ## wrong because searchsortedmust be ordered
        ind = np.nonzero(clusters['cluster_label'] == label)[0][0]
        
        if color is not None:
            if type(color) == QT.QColor:
                r, g, b, a = color.getRgb()
                clusters['color'][ind] = np.uint32(rgba_to_int32(r, g, b))
                self.refresh_colors(reset=False)
        
        if annotations is not None:
            clusters['annotations'][ind] = annotations

        if tag is not None:
            clusters['tag'][ind] = tag
        

        #~ self.colors_changed.emit()
        

    def merge_cluster(self, labels_to_merge):
        #TODO: maybe take the first cluster label instead of new one (except -1)
        new_label = min(labels_to_merge)
        if new_label<0:
            new_label = max(max(self.cluster_labels)+1, 0)
        
        mask = np.zeros(self.spike_label.size, dtype='bool')
        for k in labels_to_merge:
            mask |= (self.spike_label == k)
        self.change_spike_label(mask, new_label)
    
    def tag_same_cell(self, labels_to_group):
        self.cc.tag_same_cell(labels_to_group)
    
    def update_visible_spikes(self):
        visibles = np.array([k for k, v in self.cluster_visible.items() if v ])
        self.spike_visible[:] = np.isin(self.spike_label, visibles)

    def on_cluster_visibility_changed(self):
        self.update_visible_spikes()
        ControllerBase.on_cluster_visibility_changed(self)

    def order_clusters(self):
        self.cc.order_clusters()
        self.check_plot_attributes()
    
    def extract_some_features(self, method='global_pca', selection=None, **kargs):
        self.cc.extract_some_features(method=method, selection=selection, **kargs)
    
    def split_cluster(self, *args,  **kargs):
        #~ print('controller.split_cluster', args, kargs)
        
        self.cc.split_cluster(*args,  **kargs)
        self.check_plot_attributes()
        #~ print(self.cluster_count)
    
    @property
    def spike_waveforms_similarity(self):
        return self.cc.spike_waveforms_similarity

    @property
    def cluster_similarity(self):
        return self.cc.cluster_similarity

    @property
    def cluster_ratio_similarity(self):
        return self.cc.cluster_ratio_similarity

    @property
    def spike_silhouette(self):
        return self.cc.spike_silhouette
        
    def compute_spike_waveforms_similarity(self, **kargs):
        return self.cc.compute_spike_waveforms_similarity(**kargs)

    def compute_cluster_similarity(self, **kargs):
        return self.cc.compute_cluster_similarity(**kargs)

    def compute_cluster_ratio_similarity(self, **kargs):
        return self.cc.compute_cluster_ratio_similarity(**kargs)
        
    def compute_spike_silhouette(self, **kargs):
        return self.cc.compute_spike_silhouette(**kargs)
    
    def detect_similar_waveform_ratio(self, threshold=0.9):
        return self.cc.detect_similar_waveform_ratio(threshold=threshold)
        
    def detect_high_similarity(self, threshold=0.95):
        return self.cc.detect_high_similarity(threshold=threshold)
    
    

#~ _mapped_properties = ['clusters', 'positive_cluster_labels', 'some_features'
                #~ 'some_peaks_index', 'some_waveforms', 'some_noise_snippet', 'some_noise_features',
                #~ 'centroids',
                
                #~ ]

#~ _mapped_methods = []

#~ for prop in _mapped_properties:
    #~ getter = lambda self: self.get_catalogueconstructor_attr(prop)
    #~ print(prop, getter)
    #~ setattr(CatalogueController, prop, property(getter))

#~ print(CatalogueController.clusters)
#~ print(CatalogueController.positive_cluster_labels)


