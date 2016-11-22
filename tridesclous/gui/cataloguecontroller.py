import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import seaborn as sns

from ..tools import median_mad
from .base import ControllerBase


import time

class CatalogueController(ControllerBase):
    def __init__(self, parent=None, catalogueconstructor=None):
        ControllerBase.__init__(self, parent=parent)
        self.dataio = catalogueconstructor.dataio
        self.cc = catalogueconstructor = catalogueconstructor
        self.cc.on_new_cluster()
        
        self.init_plot_attributes()
        #~ self.on_new_cluster()
        #~ self.init_plot_attributes()
        #~ self.refresh_colors()
    
    def init_plot_attributes(self):
        self.cluster_visible = {k:True for k  in self.cluster_labels}
        self.cluster_count = { k:np.sum(self.cc.all_peaks['label']==k) for k in self.cluster_labels}
        self.spike_selection = np.zeros(self.cc.nb_peak, dtype='bool')
        self.spike_visible = np.ones(self.cc.nb_peak, dtype='bool')
        self.refresh_colors(reset=True)
        self.cc.compute_centroid()
        

    def check_plot_attributes(self):
        #cluster visibility
        for k in self.cluster_labels:
            if k not in self.cluster_visible:
                self.cluster_visible[k] = True
        for k in list(self.cluster_visible.keys()):
            if k not in self.cluster_labels:
                self.cluster_visible.pop(k)
        
        self.refresh_colors(reset=False)
    
    #map some attribute
    @property
    def cluster_labels(self):
        return self.cc.cluster_labels
        
    @property
    def spike_index(self):
        #~ return self.cc.peak_pos
        return self.cc.all_peaks['index']
    
    @property
    def spike_label(self):
        #~ return self.cc.peak_label
        return self.cc.all_peaks['label']

    @property
    def spike_segment(self):
        #~ return self.cc.peak_segment
        return self.cc.all_peaks['segment']
    
    @property
    def some_features(self):
        return self.cc.some_features
    
    @property
    def some_peaks_index(self):
        return self.cc.some_peaks_index
    
    @property
    def centroids(self):
        return self.cc.centroids

    @property
    def info(self):
        return self.cc.info
    
    def change_spike_label(self, mask, label, on_new_cluster=True):
        label_changed = np.unique(self.cc.all_peaks['label'][mask]).tolist() + [label]
        label_changed = np.unique(label_changed)
        self.cc.all_peaks['label'][mask] = label
        
        if on_new_cluster:
            self.on_new_cluster(label_changed=label_changed)
            self.refresh_colors(reset=False)
        
    def get_threshold(self):
        threshold = self.cc.info['params_peakdetector']['relative_threshold']
        if self.cc.info['params_peakdetector']['peak_sign']=='-':
            threshold = -threshold
        return threshold
    
    def on_new_cluster(self, label_changed=None):
        """
        label_changed can be remove/add/modify
        """
        if self.cc.all_peaks['index']==[]: return
        self.cc.on_new_cluster()
        
        if label_changed is None:
            #re count evry clusters
            self.cluster_count = { k:np.sum(self.cc.all_peaks['label']==k) for k in self.cluster_labels}
        else:
            for k in label_changed:
                if k in self.cluster_labels:
                    self.cluster_count[k] = np.sum(self.cc.all_peaks['label']==k)
                else:
                    if k in self.cluster_count:
                        self.cluster_count.pop(k)
        
        self.cc.compute_centroid(label_changed=label_changed)
        
        self.check_plot_attributes()

    
    def refresh_colors(self, reset=True, palette = 'husl'):
        self.cc.refresh_colors(reset=reset, palette=palette)
        
        self.qcolors = {}
        for k, color in self.cc.colors.items():
            r, g, b = color
            self.qcolors[k] = QtGui.QColor(r*255, g*255, b*255)

    def merge_cluster(self, labels_to_merge):
        #TODO: maybe take the first cluster label instead of new one (except -1)
        new_label = max(self.cluster_labels)+1
        for k in labels_to_merge:
            mask = self.spike_label == k
            self.change_spike_label(mask, new_label, on_new_cluster=False)
        self.on_new_cluster()
    
    def update_visible_spikes(self):
        visibles = np.array([k for k, v in self.cluster_visible.items() if v ])
        self.spike_visible[:] = np.in1d(self.spike_label, visibles)

    def on_cluster_visibility_changed(self):
        self.update_visible_spikes()
        ControllerBase.on_cluster_visibility_changed(self)

    def order_clusters(self):
        self.cc.order_clusters()
        self.on_new_cluster()
        self.refresh_colors(reset = True)
    
    def project(self, method='pca', selection=None, **kargs):
        self.cc.project(method=method, selection=selection, **kargs)
    
    def split_cluster(self, label_to_split, n, method=None, order_clusters=True, **kargs):
        self.cc.split_cluster(label_to_split, n, method=method, order_clusters=order_clusters, **kargs)
        self.on_new_cluster()
        self.refresh_colors(reset = False)

