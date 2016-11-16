import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from .cataloguecontroller import CatalogueController
from .traceviewer import CatalogueTraceViewer
from .peaklists import PeakList, ClusterPeakList
from .ndscatter import NDScatter
from .waveformviewer import WaveformViewer

import itertools
import datetime

class CatalogueWindow(QtGui.QMainWindow):
    def __init__(self, catalogueconstructor):
        QtGui.QMainWindow.__init__(self)
        
        self.catalogueconstructor = catalogueconstructor
        self.controller = CatalogueController(catalogueconstructor=catalogueconstructor)
        
        self.traceviewer = CatalogueTraceViewer(controller=self.controller)
        self.peaklist = PeakList(controller=self.controller)
        self.clusterlist = ClusterPeakList(controller=self.controller)
        self.ndscatter = NDScatter(controller=self.controller)
        self.waveformviewer = WaveformViewer(controller=self.controller)
        
        docks = {}

        docks['waveformviewer'] = QtGui.QDockWidget('waveformviewer',self)
        docks['waveformviewer'].setWidget(self.waveformviewer)
        #self.tabifyDockWidget(docks['ndscatter'], docks['waveformviewer'])
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['waveformviewer'])
        
        docks['traceviewer'] = QtGui.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        #self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['traceviewer'])
        self.tabifyDockWidget(docks['waveformviewer'], docks['traceviewer'])
        
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
        
        
    def create_actions(self):
        self.act_save = QtGui.QAction(u'Save catalogue', self,checkable = False, icon=QtGui.QIcon.fromTheme("document-save"))
        self.act_save.triggered.connect(self.save_catalogue)

        self.act_refresh = QtGui.QAction(u'Refresh', self,checkable = False, icon=QtGui.QIcon.fromTheme("view-refresh"))
        self.act_refresh.triggered.connect(self.refresh)

        self.act_setting = QtGui.QAction(u'Settings', self,checkable = False, icon=QtGui.QIcon.fromTheme("preferences-other"))
        self.act_setting.triggered.connect(self.open_settings)

    def create_toolbar(self):
        self.toolbar = QtGui.QToolBar('Tools')
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QtCore.QSize(60, 40))
        
        self.toolbar.addAction(self.act_save)
        self.toolbar.addAction(self.act_refresh)
        self.toolbar.addAction(self.act_setting)
    

    def save_catalogue(self):
        self.catalogueconstructor.save_catalogue()
    
    def refresh(self):
        for w in self.controller.all_view:
            w.refresh()
    
    def open_settings(self):
        #TODO
        pass

    