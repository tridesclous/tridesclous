"""
This should be rewritte with vispy but I don't have time now...
"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .tools import TimeSeeker
from ..tools import median_mad



class MyViewBox(pg.ViewBox):
    doubleclicked = QtCore.pyqtSignal()
    #~ gain_zoom = QtCore.pyqtSignal(float)
    #~ xsize_zoom = QtCore.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #~ self.disableAutoRange()
    def mouseClickEvent(self, ev):
        ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.ignore()
    def wheelEvent(self, ev):
        #~ if ev.modifiers() == QtCore.Qt.ControlModifier:
            #~ z = 10 if ev.delta()>0 else 1/10.
        #~ else:
            #~ z = 1.3 if ev.delta()>0 else 1/1.3
        #~ self.gain_zoom.emit(z)
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.accept()
        #~ self.xsize_zoom.emit((ev.pos()-ev.lastPos()).x())



class NDScatter(QtGui.QWidget):
    
    peak_selection_changed = QtCore.pyqtSignal()
    
    def __init__(self, spikesorter = None, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.spikesorter = spikesorter
        
        self.layout = QtGui.QHBoxLayout()
        self.setLayout(self.layout)
        
        self.create_toolbar()
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.viewBox = MyViewBox()
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()

        _params = [{'name': 'refresh_interval', 'type': 'float', 'value': 100 },
                           {'name': 'nb_step', 'type': 'int', 'value':  10, 'limits' : [5, 100] },]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for NDScatter')
        self.tree_params.setWindowFlags(QtCore.Qt.Window)

        
        self.timer_tour = QtCore.QTimer(interval = 50)
        self.timer_tour.timeout.connect(self.new_tour_step)
        self.initialize()
        self.refresh()
    
    def create_toolbar(self):
        
        tb = self.toolbar = QtGui.QVBoxLayout()
        self.layout.addLayout(tb)
        but = QtGui.QPushButton('Random')
        tb.addWidget(but)
        but.clicked.connect(self.random_projection)
        but = QtGui.QPushButton('Random tour', checkable = True)
        tb.addWidget(but)
        but.clicked.connect(self.start_stop_tour)
        but = QtGui.QPushButton('settings')
        but.clicked.connect(self.open_settings)
        tb.addWidget(but)
        

    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()        
    
    @property
    def data(self):
        return self.spikesorter.clustering.features
        
    @property
    def metadata(self):
        return self.spikesorter.all_peaks
    
    def initialize(self):
        self.scatters = {}
        color = QtGui.QColor( 'magenta')
        color.setAlpha(220)
        self.scatters['sel'] = pg.ScatterPlotItem(pen=None, brush=color, size=8, pxMode = True)
        self.plot.addItem(self.scatters['sel'])
        self.scatters['sel'].setZValue(1000)
        
        ndim = self.data.shape[1]
        self.projection = np.zeros( (ndim, 2))
        self.projection[0,0] = .25
        self.projection[1,1] = .25
        
    
    def get_one_random_projection(self):
        ndim = self.data.shape[1]
        projection = np.random.rand(ndim,2)*2-1.
        m = np.sqrt(np.sum(self.projection**2, axis=0))
        projection /= m
        return projection
    
    def random_projection(self):
        self.projection = self.get_one_random_projection()
        if self.timer_tour.isActive():
            self.tour_step == 0
        self.refresh()
    
    def refresh(self):
        for k, scatter in self.scatters.items():
            #~ if k not in visible_labels:
            scatter.setData([], [])
        
        visible_labels = np.unique(self.metadata['label'].values)
        for k in visible_labels:
            data = self.data[self.metadata['label']==k].values
            projected = np.dot(data, self.projection )
            #~ print(data.shape)
            #~ print(projected.shape)
            #~ print(projected)
            
            if k not in self.scatters:
                #TODO color
                color = QtGui.QColor( 'cyan')
                self.scatters[k] = pg.ScatterPlotItem(pen=None, brush=color, size=2, pxMode = True)
                self.plot.addItem(self.scatters[k])
                self.scatters[k].sigClicked.connect(self.item_clicked)
            
            self.scatters[k].setData(projected[:,0], projected[:,1])
        
        data = self.data[self.metadata['selected']]
        projected = np.dot(data, self.projection )
        self.scatters['sel'].setData(projected[:,0], projected[:,1])
    
    def start_stop_tour(self, checked):
        if checked:
            self.tour_step = 0
            self.timer_tour.start()
        else:
            self.timer_tour.stop()
    
    def new_tour_step(self):
        nb_step = self.params['nb_step']
        ndim = self.data.shape[1]
        
        if self.tour_step == 0:
            self.tour_steps = np.empty( (ndim , 2 ,  nb_step))
            arrival = self.get_one_random_projection()
            for i in range(ndim):
                for j in range(2):
                    self.tour_steps[i,j , : ] = np.linspace(self.projection[i,j] , arrival[i,j] , nb_step)
            m = np.sqrt(np.sum(self.tour_steps**2, axis=0))
            m = m[np.newaxis, : ,  :]
            self.tour_steps /= m
        
        self.projection = self.tour_steps[:,:,self.tour_step]
        
        self.tour_step+=1
        if self.tour_step>=nb_step:
            self.tour_step = 0
            
        self.refresh()
    

    def on_peak_selection_changed(self):
        self.refresh()
    
    
    def item_clicked(self):
        pass
