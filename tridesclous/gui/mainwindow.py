import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..spikesorter import SpikeSorter
from .traceviewer import TraceViewer
from .lists import PeakList, ClusterList
from .ndscatter import NDScatter
from .catalogueviewer import CatalogueViewer

import itertools

class SpikeSortingWindow(QtGui.QMainWindow):
    def __init__(self, spikesorter):
        QtGui.QMainWindow.__init__(self)
        
        self.spikesorter = spikesorter
        
        self.traceviewer = TraceViewer(spikesorter = spikesorter)
        self.peaklist = PeakList(spikesorter = spikesorter)
        self.clusterlist = ClusterList(spikesorter = spikesorter)
        self.ndscatter = NDScatter(spikesorter = spikesorter)
        self.catalogueviewer = CatalogueViewer(spikesorter = spikesorter)
        
        all = [self.traceviewer, self.peaklist, self.clusterlist, self.ndscatter, self.catalogueviewer]
        
        for w1, w2 in itertools.combinations(all,2):
            w1.peak_selection_changed.connect(w2.on_peak_selection_changed)
            w2.peak_selection_changed.connect(w1.on_peak_selection_changed)
            
            w1.peak_cluster_changed.connect(w2.on_peak_cluster_changed)
            w2.peak_cluster_changed.connect(w1.on_peak_cluster_changed)

            w1.colors_changed.connect(w2.on_colors_changed)
            w2.colors_changed.connect(w1.on_colors_changed)

            w1.cluster_visibility_changed.connect(w2.on_cluster_visibility_changed)
            w2.cluster_visibility_changed.connect(w1.on_cluster_visibility_changed)
        

        docks = {}

        docks['catalogueviewer'] = QtGui.QDockWidget('catalogueviewer',self)
        docks['catalogueviewer'].setWidget(self.catalogueviewer)
        #~ self.tabifyDockWidget(docks['ndscatter'], docks['catalogueviewer'])
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['catalogueviewer'])
        
        docks['traceviewer'] = QtGui.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        #~ self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['traceviewer'])
        self.tabifyDockWidget(docks['catalogueviewer'], docks['traceviewer'])
        
        docks['peaklist'] = QtGui.QDockWidget('peaklist',self)
        docks['peaklist'].setWidget(self.peaklist)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['peaklist'])
        
        docks['clusterlist'] = QtGui.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.splitDockWidget(docks['peaklist'], docks['clusterlist'], QtCore.Qt.Horizontal)
        
        docks['ndscatter'] = QtGui.QDockWidget('ndscatter',self)
        docks['ndscatter'].setWidget(self.ndscatter)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['ndscatter'])
        

        
        
        self.spikesorter.refresh_colors()

    
    @classmethod
    def from_classes(cls, dataio, peakdetector, waveformextractor, clustering):
        spikesorter = SpikeSorter(dataio = dataio)
        
        spikesorter.threshold = peakdetector.threshold
        spikesorter.peak_selection = pd.Series(name = 'selected', index = clustering.labels.index, dtype = bool)
        spikesorter.peak_selection[:] = False
        spikesorter.all_waveforms  = waveformextractor.get_ajusted_waveforms()
        spikesorter.clustering = clustering
        
        spikesorter.on_new_cluster()
        spikesorter.refresh_colors()
        
        return SpikeSortingWindow(spikesorter)
        
        