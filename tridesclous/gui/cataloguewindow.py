import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..dataio import DataIO
from ..spikesorter import SpikeSorter
from .traceviewer import TraceViewer
from .lists import PeakList, ClusterList
from .ndscatter import NDScatter
from .waveformviewer import WaveformViewer

import itertools
import datetime

class CatalogueWindow(QtGui.QMainWindow):
    def __init__(self, spikesorter):
        QtGui.QMainWindow.__init__(self)
        
        self.spikesorter = spikesorter
        
        self.traceviewer = TraceViewer(spikesorter = spikesorter)
        self.peaklist = PeakList(spikesorter = spikesorter)
        self.clusterlist = ClusterList(spikesorter = spikesorter)
        self.ndscatter = NDScatter(spikesorter = spikesorter)
        self.WaveformViewer = WaveformViewer(spikesorter = spikesorter)
        
        self.all_view = [self.traceviewer, self.peaklist, self.clusterlist, self.ndscatter, self.WaveformViewer]
        
        for w1, w2 in itertools.combinations(self.all_view,2):
            w1.peak_selection_changed.connect(w2.on_peak_selection_changed)
            w2.peak_selection_changed.connect(w1.on_peak_selection_changed)
            
            w1.peak_cluster_changed.connect(w2.on_peak_cluster_changed)
            w2.peak_cluster_changed.connect(w1.on_peak_cluster_changed)

            w1.colors_changed.connect(w2.on_colors_changed)
            w2.colors_changed.connect(w1.on_colors_changed)

            w1.cluster_visibility_changed.connect(w2.on_cluster_visibility_changed)
            w2.cluster_visibility_changed.connect(w1.on_cluster_visibility_changed)
        

        docks = {}

        docks['WaveformViewer'] = QtGui.QDockWidget('WaveformViewer',self)
        docks['WaveformViewer'].setWidget(self.WaveformViewer)
        #~ self.tabifyDockWidget(docks['ndscatter'], docks['WaveformViewer'])
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['WaveformViewer'])
        
        docks['traceviewer'] = QtGui.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        #~ self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['traceviewer'])
        self.tabifyDockWidget(docks['WaveformViewer'], docks['traceviewer'])
        
        docks['peaklist'] = QtGui.QDockWidget('peaklist',self)
        docks['peaklist'].setWidget(self.peaklist)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['peaklist'])
        
        docks['clusterlist'] = QtGui.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.splitDockWidget(docks['peaklist'], docks['clusterlist'], QtCore.Qt.Horizontal)
        
        docks['ndscatter'] = QtGui.QDockWidget('ndscatter',self)
        docks['ndscatter'].setWidget(self.ndscatter)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['ndscatter'])
        
        self.create_actions()
        self.create_toolbar()
        
        self.spikesorter.refresh_colors()
        
    def create_actions(self):
        self.act_save = QtGui.QAction(u'Save catalogue', self,checkable = False, icon=QtGui.QIcon.fromTheme("document-save"))
        self.act_save.triggered.connect(self.save_catalogue)

        self.act_refresh = QtGui.QAction(u'Refresh', self,checkable = False, icon=QtGui.QIcon.fromTheme("view-refresh"))
        self.act_refresh.triggered.connect(self.refresh)

        self.act_decimate = QtGui.QAction(u'Random decimate', self,checkable = False, icon=QtGui.QIcon.fromTheme("roll"))
        self.act_decimate.triggered.connect(self.random_decimate)

        self.act_setting = QtGui.QAction(u'Settings', self,checkable = False, icon=QtGui.QIcon.fromTheme("preferences-other"))
        self.act_setting.triggered.connect(self.open_settings)

    
    
    
    def create_toolbar(self):
        self.toolbar = QtGui.QToolBar('Tools')
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QtCore.QSize(60, 40))
        
        self.toolbar.addAction(self.act_save)
        self.toolbar.addAction(self.act_refresh)
        self.toolbar.addAction(self.act_decimate)
        self.toolbar.addAction(self.act_setting)
    
    @classmethod
    def from_classes(cls, peakdetector, waveformextractor, clustering, dataio =None):
        if dataio is None:
            name = 'test_tri_des_clous_'+datetime.datetime.now().strftime('%A, %d. %B %Y %Ih%M%pm%S')
            print('Create DataIO : ', name)
            dataio = DataIO(name)
            dataio.append_signals(peakdetector.sigs,  seg_num=0, signal_type = 'filtered')
        spikesorter = SpikeSorter(dataio = dataio)
        
        spikesorter.threshold = peakdetector.threshold
        spikesorter.peak_selection = pd.Series(name = 'selected', index = clustering.labels.index, dtype = bool)
        spikesorter.peak_selection[:] = False
        spikesorter.all_waveforms  = waveformextractor.get_ajusted_waveforms()
        spikesorter.clustering = clustering
        
        spikesorter.on_new_cluster()
        spikesorter.refresh_colors()
        
        return CatalogueWindow(spikesorter)

    def save_catalogue(self):
        self.spikesorter.dataio.save_catalogue(self.spikesorter.clustering.catalogue)
    
    def refresh(self):
        for w in self.all_view:
            w.refresh()
    
    def random_decimate(self):
        pass
    
    def open_settings(self):
        pass