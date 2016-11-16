import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


from .peelercontroller import PeelerController
from .traceviewer import PeelerTraceViewer
from .spikelists import SpikeList

import itertools
import datetime



    

class PeelerWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, dataio=None,catalogue=None):
        QtGui.QMainWindow.__init__(self, parent=None)
        
        self.controller = PeelerController(dataio=dataio,catalogue=catalogue)
        
        self.traceviewer = PeelerTraceViewer(controller=self.controller)
        self.spikelist = SpikeList(controller=self.controller)
        
        all = [self.traceviewer, self.spikelist]
        
        #~ for w1, w2 in itertools.combinations(all,2):
            #~ w1.peak_selection_changed.connect(w2.on_peak_selection_changed)
            #~ w2.peak_selection_changed.connect(w1.on_peak_selection_changed)
            
            #~ w1.peak_cluster_changed.connect(w2.on_peak_cluster_changed)
            #~ w2.peak_cluster_changed.connect(w1.on_peak_cluster_changed)

            #~ w1.colors_changed.connect(w2.on_colors_changed)
            #~ w2.colors_changed.connect(w1.on_colors_changed)

            #~ w1.cluster_visibility_changed.connect(w2.on_cluster_visibility_changed)
            #~ w2.cluster_visibility_changed.connect(w1.on_cluster_visibility_changed)
        

        docks = {}

        
        docks['traceviewer'] = QtGui.QDockWidget('traceviewer',self)
        docks['traceviewer'].setWidget(self.traceviewer)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['traceviewer'])

        docks['spikelist'] = QtGui.QDockWidget('spikelist',self)
        docks['spikelist'].setWidget(self.spikelist)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['spikelist'])


