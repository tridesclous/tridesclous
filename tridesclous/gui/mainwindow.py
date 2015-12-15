import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..spikesorter import SpikeSorter
from .traceviewer import TraceViewer
from .lists import PeakList, ClusterList
from .ndscatter import NDScatter

import itertools

class SpikeSortingWindow(QtGui.QMainWindow):
    def __init__(self, spikesorter):
        QtGui.QMainWindow.__init__(self)
        
        self.spikesorter = spikesorter
        
        self.traceviewer = TraceViewer(spikesorter = spikesorter)
        self.peaklist = PeakList(spikesorter = spikesorter)
        self.clusterlist = ClusterList(spikesorter = spikesorter)
        self.ndscatter = NDScatter(spikesorter = spikesorter)
        
        all = [self.traceviewer, self.peaklist, self.ndscatter]
        
        for w1, w2 in itertools.combinations(all,2):
            w1.peak_selection_changed.connect(w2.on_peak_selection_changed)
            w2.peak_selection_changed.connect(w1.on_peak_selection_changed)
        
        docks = {}
        docks['traceviewer'] = QtGui.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['traceviewer'])
        docks['peaklist'] = QtGui.QDockWidget('peaklist',self)
        docks['peaklist'].setWidget(self.peaklist)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['peaklist'])
        docks['ndscatter'] = QtGui.QDockWidget('ndscatter',self)
        docks['ndscatter'].setWidget(self.ndscatter)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['ndscatter'])

        docks['clusterlist'] = QtGui.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.splitDockWidget(docks['peaklist'], docks['clusterlist'], QtCore.Qt.Horizontal)
        
        self.spikesorter.refresh_colors()

    
    @classmethod
    def from_classes(cls, dataio, peakdetector, waveformextractor, clustering):
        spikesorter = SpikeSorter(dataio = dataio)
        
        spikesorter.all_peaks = pd.DataFrame(np.zeros(peakdetector.peak_index.size, dtype = 'int32'), columns = ['label'], index = peakdetector.peak_index)
        spikesorter.all_peaks['label'] = clustering.labels
        spikesorter.all_peaks['selected'] = False
        spikesorter.all_waveforms  = waveformextractor.get_ajusted_waveforms()
        spikesorter.clustering = clustering
        
        spikesorter.refresh_colors()
        
        return SpikeSortingWindow(spikesorter)
        
        