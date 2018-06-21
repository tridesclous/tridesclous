import numpy as np

from .myqt import QT
import pyqtgraph as pg

from .cataloguecontroller import CatalogueController
from .traceviewer import CatalogueTraceViewer
from .peaklists import PeakList, ClusterPeakList
from .ndscatter import NDScatter
from .waveformviewer import WaveformViewer
from .similarity import SpikeSimilarityView, ClusterSimilarityView, ClusterRatioSimilarityView
from .pairlist import PairList
from .silhouette import Silhouette
from .waveformhistviewer import WaveformHistViewer
from .featuretimeviewer import FeatureTimeViewer

from .tools import ParamDialog, open_dialog_methods

from . import gui_params

from . import icons


import itertools
import datetime
import time
import webbrowser

class CatalogueWindow(QT.QMainWindow):
    new_catalogue = QT.pyqtSignal(int)
    def __init__(self, catalogueconstructor):
        QT.QMainWindow.__init__(self)
        
        self.setWindowIcon(QT.QIcon(':/main_icon.png'))
        
        self.catalogueconstructor = catalogueconstructor
        self.controller = CatalogueController(catalogueconstructor=catalogueconstructor)
        
        self.traceviewer = CatalogueTraceViewer(controller=self.controller)
        self.peaklist = PeakList(controller=self.controller)
        self.clusterlist = ClusterPeakList(controller=self.controller)
        self.ndscatter = NDScatter(controller=self.controller)
        self.waveformviewer = WaveformViewer(controller=self.controller)
        self.spikesimilarityview = SpikeSimilarityView(controller=self.controller)
        self.clustersimilarityview = ClusterSimilarityView(controller=self.controller)
        self.clusterratiosimilarityview = ClusterRatioSimilarityView(controller=self.controller)
        self.pairlist = PairList(controller=self.controller)
        self.silhouette = Silhouette(controller=self.controller)
        self.waveformhistviewer = WaveformHistViewer(controller=self.controller)
        self.featuretimeviewer = FeatureTimeViewer(controller=self.controller)
        
        
        docks = {}

        docks['waveformviewer'] = QT.QDockWidget('waveformviewer',self)
        docks['waveformviewer'].setWidget(self.waveformviewer)
        #self.tabifyDockWidget(docks['ndscatter'], docks['waveformviewer'])
        self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['waveformviewer'])
        

        docks['waveformhistviewer'] = QT.QDockWidget('waveformhistviewer',self)
        docks['waveformhistviewer'].setWidget(self.waveformhistviewer)
        self.tabifyDockWidget(docks['waveformviewer'], docks['waveformhistviewer'])

        docks['featuretimeviewer'] = QT.QDockWidget('featuretimeviewer',self)
        docks['featuretimeviewer'].setWidget(self.featuretimeviewer)
        self.tabifyDockWidget(docks['waveformhistviewer'], docks['featuretimeviewer'])
        
        
        docks['traceviewer'] = QT.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        #self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['traceviewer'])
        self.tabifyDockWidget(docks['waveformviewer'], docks['traceviewer'])
        
        docks['peaklist'] = QT.QDockWidget('peaklist',self)
        docks['peaklist'].setWidget(self.peaklist)
        self.addDockWidget(QT.Qt.LeftDockWidgetArea, docks['peaklist'])
        
        docks['pairlist'] = QT.QDockWidget('pairlist',self)
        docks['pairlist'].setWidget(self.pairlist)
        self.splitDockWidget(docks['peaklist'], docks['pairlist'], QT.Qt.Horizontal)
        
        docks['clusterlist'] = QT.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.tabifyDockWidget(docks['pairlist'], docks['clusterlist'])
        
        #on bottom left
        docks['spikesimilarityview'] = QT.QDockWidget('spikesimilarityview',self)
        docks['spikesimilarityview'].setWidget(self.spikesimilarityview)
        self.addDockWidget(QT.Qt.LeftDockWidgetArea, docks['spikesimilarityview'])

        docks['clustersimilarityview'] = QT.QDockWidget('clustersimilarityview',self)
        docks['clustersimilarityview'].setWidget(self.clustersimilarityview)
        self.tabifyDockWidget(docks['spikesimilarityview'], docks['clustersimilarityview'])

        docks['clusterratiosimilarityview'] = QT.QDockWidget('clusterratiosimilarityview',self)
        docks['clusterratiosimilarityview'].setWidget(self.clusterratiosimilarityview)
        self.tabifyDockWidget(docks['spikesimilarityview'], docks['clusterratiosimilarityview'])
        

        docks['silhouette'] = QT.QDockWidget('silhouette',self)
        docks['silhouette'].setWidget(self.silhouette)
        self.tabifyDockWidget(docks['spikesimilarityview'], docks['silhouette'])
        
        
        docks['ndscatter'] = QT.QDockWidget('ndscatter',self)
        docks['ndscatter'].setWidget(self.ndscatter)
        self.tabifyDockWidget(docks['spikesimilarityview'], docks['ndscatter'])
        
        self.create_actions()
        self.create_toolbar()
        
        
    def create_actions(self):

        self.act_make_catalogue = QT.QAction('Make catalogue for peeler', self,checkable = False, icon=QT.QIcon(":/document-save.svg"))
        self.act_make_catalogue.triggered.connect(self.make_catalogue_for_peeler)

        self.act_savepoint = QT.QAction('Savepoint', self,checkable = False, icon=QT.QIcon(":/document-save.svg"))
        self.act_savepoint.triggered.connect(self.create_savepoint)
        
        #~ self.act_refresh = QT.QAction('Refresh', self,checkable = False, icon=QT.QIcon.fromTheme("view-refresh"))
        self.act_refresh = QT.QAction('Refresh', self,checkable = False, icon=QT.QIcon(":/view-refresh.svg"))
        self.act_refresh.triggered.connect(self.refresh_with_reload)
        
        self.act_redetect_peak = QT.QAction('New peaks', self,checkable = False, icon=QT.QIcon(":/configure-shortcuts.svg"))
        self.act_redetect_peak.triggered.connect(self.redetect_peak)

        self.act_new_waveforms = QT.QAction('New waveforms', self,checkable = False, icon=QT.QIcon(":/configure-shortcuts.svg"))
        self.act_new_waveforms.triggered.connect(self.new_waveforms)

        self.act_clean_waveforms = QT.QAction('Clean waveforms', self,checkable = False, icon=QT.QIcon(":/configure-shortcuts.svg"))
        self.act_clean_waveforms.triggered.connect(self.clean_waveforms)

        self.act_new_noise_snippet = QT.QAction('New noise snippet', self,checkable = False, icon=QT.QIcon(":/configure-shortcuts.svg"))
        self.act_new_noise_snippet.triggered.connect(self.new_noise_snippet)

        self.act_new_features = QT.QAction('New features', self,checkable = False, icon=QT.QIcon(":/configure-shortcuts.svg"))
        self.act_new_features.triggered.connect(self.new_features)

        self.act_new_cluster = QT.QAction('New cluster', self,checkable = False, icon=QT.QIcon(":/configure-shortcuts.svg"))
        self.act_new_cluster.triggered.connect(self.new_cluster)

        self.act_compute_metrics = QT.QAction('Compute metrics', self,checkable = False, icon=QT.QIcon(":/configure-shortcuts.svg"))
        self.act_compute_metrics.triggered.connect(self.compute_metrics)


        
        self.help_act = QT.QAction('Help', self,checkable = False, icon=QT.QIcon(":main_icon.png"))
        self.help_act.triggered.connect(self.open_webbrowser_help)
        


    def create_toolbar(self):
        self.toolbar = QT.QToolBar('Tools')
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.Qt.RightToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))
        
        self.toolbar.addAction(self.act_make_catalogue)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_refresh)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_redetect_peak)
        self.toolbar.addAction(self.act_new_waveforms)
        self.toolbar.addAction(self.act_clean_waveforms)
        self.toolbar.addAction(self.act_new_noise_snippet)
        self.toolbar.addAction(self.act_new_features)
        self.toolbar.addAction(self.act_new_cluster)
        self.toolbar.addAction(self.act_compute_metrics)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.help_act)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_savepoint)
    

    def warn(self, title, text):
        mb = QT.QMessageBox.warning(self, title,text, QT.QMessageBox.Ok ,  QT.QMessageBox.NoButton)
    
    def open_webbrowser_help(self):
        url = "http://tridesclous.readthedocs.io/en/latest/catalogue_window.html"
        webbrowser.open(url, new=2)

    def make_catalogue_for_peeler(self):
        self.catalogueconstructor.make_catalogue_for_peeler()
        self.new_catalogue.emit(self.catalogueconstructor.chan_grp)
    
    def create_savepoint(self):
        try:
            copy_path = self.catalogueconstructor.create_savepoint()
        except:
            copy_path = None
        
        if copy_path is None:
            txt = 'Savepoint FAIL!!!'
        else:
            txt = 'Savepoint done here {}'.format(copy_path)
        self.warn('savepoint', txt)
        
    
    def refresh_with_reload(self):
        self.controller.reload_data()
        self.refresh()
    
    def refresh(self):
        self.controller.check_plot_attributes()
        for w in self.controller.views:
            #TODO refresh only visible but need catch on visibility changed
            #~ print(w)
            #~ t1 = time.perf_counter()
            w.refresh()
            #~ t2 = time.perf_counter()
            #~ print('refresh',w,  t2-t1)
    
    def redetect_peak(self):
        dia = ParamDialog(gui_params.peak_detector_params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            self.catalogueconstructor.re_detect_peak(**d)
            self.controller.init_plot_attributes()
        self.refresh()
    
    def new_waveforms(self):
        dia = ParamDialog(gui_params.waveforms_params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            self.catalogueconstructor.extract_some_waveforms(**d)
            
            self.refresh()

    def clean_waveforms(self):
        dia = ParamDialog(gui_params.clean_waveforms_params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            self.catalogueconstructor.clean_waveforms(**d)
            self.refresh()

    def new_noise_snippet(self):
        dia = ParamDialog(gui_params.noise_snippet_params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            self.catalogueconstructor.extract_some_noise(**d)
        self.refresh()

    def new_features(self):
        method, kargs = open_dialog_methods(gui_params.features_params_by_methods, self)
        if method is not None:
            self.catalogueconstructor.extract_some_features(method=method, **kargs)
            self.refresh()


    def new_cluster(self):
        method, kargs = open_dialog_methods(gui_params.cluster_params_by_methods, self)
        if method is not None:
            self.catalogueconstructor.find_clusters(method=method, **kargs)
            self.refresh()
    
    def compute_metrics(self):
        dia = ParamDialog(gui_params.metrics_params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            self.catalogueconstructor.compute_spike_waveforms_similarity(method=d['spike_waveforms_similarity'], size_max=d['size_max'])
            self.catalogueconstructor.compute_cluster_similarity(method=d['cluster_similarity'])
            self.catalogueconstructor.compute_cluster_ratio_similarity(method=d['cluster_ratio_similarity'])
            self.catalogueconstructor.compute_spike_silhouette(size_max=d['size_max'])
            #TODO refresh only metrics concerned
            self.refresh()
        
        
    

