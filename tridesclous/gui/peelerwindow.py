import numpy as np
import pandas as pd

from .myqt import QT
import pyqtgraph as pg


from .peelercontroller import PeelerController
from .traceviewer import PeelerTraceViewer
from .spikelists import SpikeList, ClusterSpikeList
from .waveformviewer import PeelerWaveformViewer
from .waveformhistviewer import WaveformHistViewer
from .isiviewer import ISIViewer
from .crosscorrelogramviewer import CrossCorrelogramViewer

from . import icons

import itertools
import datetime



    

class PeelerWindow(QT.QMainWindow):
    def __init__(self, parent=None, dataio=None, catalogue=None):
        QT.QMainWindow.__init__(self, parent=None)
        
        self.setWindowIcon(QT.QIcon(':/main_icon.png'))
        
        self.controller = PeelerController(dataio=dataio, catalogue=catalogue)
        
        self.traceviewer = PeelerTraceViewer(controller=self.controller)
        self.spikelist = SpikeList(controller=self.controller)
        self.clusterlist = ClusterSpikeList(controller=self.controller)
        self.waveformviewer = PeelerWaveformViewer(controller=self.controller)
        self.waveformhistviewer = WaveformHistViewer(controller=self.controller)
        self.isiviewer = ISIViewer(controller=self.controller)
        self.crosscorrelogramviewer = CrossCorrelogramViewer(controller=self.controller)
        
        all = [self.traceviewer, self.spikelist, self.clusterlist, self.isiviewer]
        
        docks = {}


        docks['isiviewer'] = QT.QDockWidget('isiviewer',self)
        docks['isiviewer'].setWidget(self.isiviewer)
        self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['isiviewer'])

        docks['crosscorrelogramviewer'] = QT.QDockWidget('crosscorrelogramviewer',self)
        docks['crosscorrelogramviewer'].setWidget(self.crosscorrelogramviewer)
        self.tabifyDockWidget(docks['isiviewer'], docks['crosscorrelogramviewer'])

        docks['waveformviewer'] = QT.QDockWidget('waveformviewer',self)
        docks['waveformviewer'].setWidget(self.waveformviewer)
        #~ self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['waveformviewer'])
        self.tabifyDockWidget(docks['crosscorrelogramviewer'], docks['waveformviewer'])
        
        docks['waveformhistviewer'] = QT.QDockWidget('waveformhistviewer',self)
        docks['waveformhistviewer'].setWidget(self.waveformhistviewer)
        #~ self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['waveformviewer'])
        self.tabifyDockWidget(docks['waveformviewer'], docks['waveformhistviewer'])
        
        
        docks['traceviewer'] = QT.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        #~ self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['traceviewer'])
        self.tabifyDockWidget(docks['waveformhistviewer'], docks['traceviewer'])

        docks['spikelist'] = QT.QDockWidget('spikelist',self)
        docks['spikelist'].setWidget(self.spikelist)
        self.addDockWidget(QT.Qt.LeftDockWidgetArea, docks['spikelist'])

        docks['clusterlist'] = QT.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.splitDockWidget(docks['spikelist'], docks['clusterlist'], QT.Qt.Horizontal)


