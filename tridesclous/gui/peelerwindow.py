import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


from .peelercontroller import PeelerController
from .traceviewer import PeelerTraceViewer
from .spikelists import SpikeList, ClusterSpikeList

import itertools
import datetime



    

class PeelerWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, dataio=None, chan_grp=0, catalogue=None):
        QtGui.QMainWindow.__init__(self, parent=None)
        
        self.controller = PeelerController(dataio=dataio,chan_grp=chan_grp, catalogue=catalogue)
        
        self.traceviewer = PeelerTraceViewer(controller=self.controller)
        self.spikelist = SpikeList(controller=self.controller)
        self.clusterlist = ClusterSpikeList(controller=self.controller)
        
        all = [self.traceviewer, self.spikelist, self.clusterlist]
        
        docks = {}

        
        docks['traceviewer'] = QtGui.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['traceviewer'])

        docks['spikelist'] = QtGui.QDockWidget('spikelist',self)
        docks['spikelist'].setWidget(self.spikelist)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['spikelist'])

        docks['clusterlist'] = QtGui.QDockWidget('clusterlist',self)
        docks['clusterlist'].setWidget(self.clusterlist)
        self.splitDockWidget(docks['spikelist'], docks['clusterlist'], QtCore.Qt.Horizontal)


