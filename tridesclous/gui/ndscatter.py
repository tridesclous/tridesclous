"""
This should be rewritte with vispy but I don't have time now...
"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

import itertools

from .base import WidgetBase
from ..tools import median_mad



class MyViewBox(pg.ViewBox):
    doubleclicked = QtCore.pyqtSignal()
    gain_zoom = QtCore.pyqtSignal(float)
    #~ xsize_zoom = QtCore.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.disableAutoRange()
    def mouseClickEvent(self, ev):
        ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.ignore()
    def wheelEvent(self, ev):
        if ev.modifiers() == QtCore.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.accept()
        #~ self.xsize_zoom.emit((ev.pos()-ev.lastPos()).x())



class NDScatter(WidgetBase):
    def __init__(self, catalogueconstructor=None, parent=None):
        WidgetBase.__init__(self, parent)
        
        self.cc = self.catalogueconstructor = catalogueconstructor
        
        self.layout = QtGui.QHBoxLayout()
        self.setLayout(self.layout)
        
        self.create_toolbar()
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)

        self.toolbar.addStretch()
        self.graphicsview2 = pg.GraphicsView()
        self.toolbar.addWidget(self.graphicsview2)

        _params = [{'name': 'refresh_interval', 'type': 'float', 'value': 100 },
                           {'name': 'nb_step', 'type': 'int', 'value':  10, 'limits' : [5, 100] },]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for NDScatter')
        self.tree_params.setWindowFlags(QtCore.Qt.Window)

        
        self.timer_tour = QtCore.QTimer(interval = 100)
        self.timer_tour.timeout.connect(self.new_tour_step)
        
        if self.data is not None:
            self.initialize()
            self.refresh()
    
    def create_toolbar(self):
        
        tb = self.toolbar = QtGui.QVBoxLayout()
        self.layout.addLayout(tb)
        but = QtGui.QPushButton('next face')
        tb.addWidget(but)
        but.clicked.connect(self.next_face)
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
        if hasattr(self.cc, 'features'):
            return self.cc.features
    
    def initialize(self):
        self.viewBox = MyViewBox()
        self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        self.scatters = {}
        brush = QtGui.QColor( 'magenta')
        brush.setAlpha(180)
        pen = QtGui.QColor( 'yellow')
        self.scatters['sel'] = pg.ScatterPlotItem(pen=pen, brush=brush, size=11, pxMode = True)
        self.plot.addItem(self.scatters['sel'])
        self.scatters['sel'].setZValue(1000)
        
        #~ m = np.max(np.abs(self.data.values))
        med, mad = median_mad(self.data)
        m = 4.*np.max(mad)
        self.limit = m
        self.plot.setXRange(-m, m)
        self.plot.setYRange(-m, m)
        
        ndim = self.data.shape[1]
        self.projection = np.zeros( (ndim, 2))
        self.projection[0,0] = 1.
        self.projection[1,1] = 1.
        
        self.plot2 = pg.PlotItem(viewBox=MyViewBox(lockAspect=True))
        self.graphicsview2.setCentralItem(self.plot2)
        self.plot2.hideButtons()
        angles = np.arange(0,360, .1)
        self.circle = pg.PlotCurveItem(x=np.cos(angles), y=np.sin(angles), pen=(255,255,255))
        self.plot2.addItem(self.circle)
        self.direction_lines = pg.PlotCurveItem(x=[], y=[], pen=(255,255,255))
        self.direction_data = np.zeros( (ndim*2, 2))
        self.plot2.addItem(self.direction_lines)
        self.plot2.setXRange(-1, 1)
        self.plot2.setYRange(-1, 1)
        self.proj_labels = []
        for i in range(ndim):
            text = 'PC{}'.format(i)
            label = pg.TextItem(text, color=(1,1,1), anchor=(0.5, 0.5), border=None, fill=pg.mkColor((128,128,128, 180)))
            self.proj_labels.append(label)
            self.plot2.addItem(label)
        
        self.graphicsview2.setMaximumSize(200, 200)
        
        #~ self.hyper_faces = list(itertools.product(range(ndim), range(ndim)))
        self.hyper_faces = list(itertools.permutations(range(ndim), 2))
        self.n_face = -1
        
    
    def next_face(self):
        self.n_face += 1
        self.n_face = self.n_face%len(self.hyper_faces)
        ndim = self.data.shape[1]
        self.projection = np.zeros( (ndim, 2))
        i, j = self.hyper_faces[self.n_face]
        self.projection[i,0] = 1.
        self.projection[j,1] = 1.
        if self.timer_tour.isActive():
            self.tour_step == 0
        self.refresh()
        
    def get_one_random_projection(self):
        ndim = self.data.shape[1]
        projection = np.random.rand(ndim,2)*2-1.
        m = np.sqrt(np.sum(projection**2, axis=0))
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
        
        if self.data.shape[1] != self.projection.shape[0]:
            self.initialize()
        
        for k in self.cc.cluster_labels:
            color = self.cc.qcolors.get(k, QtGui.QColor( 'white'))
            if k not in self.scatters:
                self.scatters[k] = pg.ScatterPlotItem(pen=None, brush=color, size=2, pxMode = True)
                self.plot.addItem(self.scatters[k])
                self.scatters[k].sigClicked.connect(self.item_clicked)
            else:
                self.scatters[k].setBrush(color)
            
            if self.cc.cluster_visible[k]:
                data = self.data[(self.cc.peak_labels==k) & self.cc.peak_visible]
                projected = np.dot(data, self.projection )
                self.scatters[k].setData(projected[:,0], projected[:,1])
            else:
                self.scatters[k].setData([], [])
        
        data = self.data[self.cc.peak_selection]
        projected = np.dot(data, self.projection )
        self.scatters['sel'].setData(projected[:,0], projected[:,1])
        
        self.direction_data[::, :] =0
        self.direction_data[::2, :] = self.projection
        self.direction_lines.setData(self.direction_data[:,0], self.direction_data[:,1])
        
        for i, label in enumerate(self.proj_labels):
            label.setPos(self.projection[i,0], self.projection[i,1])
    
    def start_stop_tour(self, checked):
        if checked:
            self.tour_step = 0
            self.timer_tour.setInterval(self.params['refresh_interval'])
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

    def gain_zoom(self, factor):
        self.limit /= factor
        self.plot.setXRange(-self.limit, self.limit)
        self.plot.setYRange(-self.limit, self.limit)
    
    def item_clicked(self):
        pass
