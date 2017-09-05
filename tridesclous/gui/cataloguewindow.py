import numpy as np

from .myqt import QT
import pyqtgraph as pg

from .cataloguecontroller import CatalogueController
from .traceviewer import CatalogueTraceViewer
from .peaklists import PeakList, ClusterPeakList
from .ndscatter import NDScatter
from .waveformviewer import WaveformViewer
from .similarity import SimilarityView

from .tools import ParamDialog


import itertools
import datetime

class CatalogueWindow(QT.QMainWindow):
    def __init__(self, catalogueconstructor):
        QT.QMainWindow.__init__(self)
        
        self.catalogueconstructor = catalogueconstructor
        self.controller = CatalogueController(catalogueconstructor=catalogueconstructor)
        
        self.traceviewer = CatalogueTraceViewer(controller=self.controller)
        self.peaklist = PeakList(controller=self.controller)
        self.clusterlist = ClusterPeakList(controller=self.controller)
        self.ndscatter = NDScatter(controller=self.controller)
        self.waveformviewer = WaveformViewer(controller=self.controller)
        self.similarityview = SimilarityView(controller=self.controller)
        
        docks = {}

        docks['waveformviewer'] = QT.QDockWidget('waveformviewer',self)
        docks['waveformviewer'].setWidget(self.waveformviewer)
        #self.tabifyDockWidget(docks['ndscatter'], docks['waveformviewer'])
        self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['waveformviewer'])
        
        docks['similarityview'] = QT.QDockWidget('similarityview',self)
        docks['similarityview'].setWidget(self.similarityview)
        self.tabifyDockWidget(docks['waveformviewer'], docks['similarityview'])
        
        docks['traceviewer'] = QT.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        #self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['traceviewer'])
        self.tabifyDockWidget(docks['similarityview'], docks['traceviewer'])
        
        docks['peaklist'] = QT.QDockWidget('peaklist',self)
        docks['peaklist'].setWidget(self.peaklist)
        self.addDockWidget(QT.Qt.LeftDockWidgetArea, docks['peaklist'])
        
        docks['clusterlist'] = QT.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.splitDockWidget(docks['peaklist'], docks['clusterlist'], QT.Qt.Horizontal)
        
        docks['ndscatter'] = QT.QDockWidget('ndscatter',self)
        docks['ndscatter'].setWidget(self.ndscatter)
        self.addDockWidget(QT.Qt.LeftDockWidgetArea, docks['ndscatter'])
        
        self.create_actions()
        self.create_toolbar()
        
        
    def create_actions(self):
        self.act_save = QT.QAction(u'Save catalogue', self,checkable = False, icon=QT.QIcon.fromTheme("document-save"))
        self.act_save.triggered.connect(self.save_catalogue)

        self.act_refresh = QT.QAction(u'Refresh', self,checkable = False, icon=QT.QIcon.fromTheme("view-refresh"))
        self.act_refresh.triggered.connect(self.refresh)

        self.act_setting = QT.QAction(u'Settings', self,checkable = False, icon=QT.QIcon.fromTheme("preferences-other"))
        self.act_setting.triggered.connect(self.open_settings)

        self.act_new_waveforms = QT.QAction(u'New waveforms', self,checkable = False, icon=QT.QIcon.fromTheme("TODO"))
        self.act_new_waveforms.triggered.connect(self.new_waveforms)

    def create_toolbar(self):
        self.toolbar = QT.QToolBar('Tools')
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.Qt.RightToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))
        
        self.toolbar.addAction(self.act_save)
        self.toolbar.addAction(self.act_refresh)
        self.toolbar.addAction(self.act_setting)
        #TODO with correct settings (left and right)
        self.toolbar.addAction(self.act_new_waveforms)
    

    def save_catalogue(self):
        self.catalogueconstructor.save_catalogue()
    
    def refresh(self):
        for w in self.controller.views:
            w.refresh()
    
    def open_settings(self):
        _params = [{'name' : 'nb_waveforms', 'type' : 'int', 'value' : 10000}]
        dialog1 = ParamDialog(_params, title = 'Settings', parent = self)
        if not dialog1.exec_():
            return None, None
        
        self.settings = dialog1.get()
    
    def new_waveforms(self):
        pass
        #~ self.catalogueconstructor.extract_some_waveforms(n_left=-12, n_right=15, mode='rand', nb_max=10000)
        #~ self.controller.on_new_cluster()
        #~ self.refresh()

