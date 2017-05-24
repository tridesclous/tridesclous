import numpy as np
import pandas as pd

from .myqt import QT
import pyqtgraph as pg


from .peelercontroller import PeelerController
from .traceviewer import PeelerTraceViewer
from .spikelists import SpikeList, ClusterSpikeList

import itertools
import datetime



    

class PeelerWindow(QT.QMainWindow):
    def __init__(self, parent=None, dataio=None, catalogue=None):
        QT.QMainWindow.__init__(self, parent=None)
        
        self.controller = PeelerController(dataio=dataio, catalogue=catalogue)
        
        self.traceviewer = PeelerTraceViewer(controller=self.controller)
        self.spikelist = SpikeList(controller=self.controller)
        self.clusterlist = ClusterSpikeList(controller=self.controller)
        
        all = [self.traceviewer, self.spikelist, self.clusterlist]
        
        docks = {}

        
        docks['traceviewer'] = QT.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        self.addDockWidget(QT.Qt.RightDockWidgetArea, docks['traceviewer'])

        docks['spikelist'] = QT.QDockWidget('spikelist',self)
        docks['spikelist'].setWidget(self.spikelist)
        self.addDockWidget(QT.Qt.LeftDockWidgetArea, docks['spikelist'])

        docks['clusterlist'] = QT.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.splitDockWidget(docks['spikelist'], docks['clusterlist'], QT.Qt.Horizontal)


