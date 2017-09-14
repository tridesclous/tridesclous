"""
This view is from taken from sklearn examples.
See http://scikit-learn.org: plot-kmeans-silhouette-analysis-py



"""
from .myqt import QT
import pyqtgraph as pg

import numpy as np
import matplotlib.cm
import matplotlib.colors



#~ import sklearn.metrics.pairwise
from sklearn.metrics import silhouette_samples, silhouette_score

from .base import WidgetBase
from .tools import ParamDialog

class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()


class Silhouette(WidgetBase):
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        h = QT.QHBoxLayout()
        self.layout.addLayout(h)
        h.addWidget(QT.QLabel('<b>Silhouette</b>') )

        but = QT.QPushButton('settings')
        but.clicked.connect(self.open_settings)
        h.addWidget(but)

        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        
        self.alpha = 60
        
        self.create_settings()
        self.initialize_plot()
        self.compute_slihouette()
        self.refresh()
        

    def create_settings(self):
        _params = [
                          {'name': 'data', 'type': 'list', 'values' : ['waveforms', 'features', ] },
                          
                          ]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        
        self.params.sigTreeStateChanged.connect(self.refresh)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for waveforms viewer')
        self.tree_params.setWindowFlags(QT.Qt.Window)
        
        self.params.sigTreeStateChanged.connect(self.on_params_change)  
        
    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()
            
    def on_params_change(self):
        self.compute_slihouette()
        self.refresh()

    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
    def compute_slihouette(self):
        if self.params['data']=='waveforms':
            wf = self.controller.some_waveforms
            data = wf.reshape(wf.shape[0], -1)
        if self.params['data']=='features':
            data = self.controller.some_features

        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep = labels>=0
        labels = labels[keep]
        data = data[keep]        
        
        labels_list = np.unique(labels)
        if labels_list.size<=1:
            self.silhouette_avg = None
            return
        
        self.silhouette_avg = silhouette_score(data, labels)
        silhouette_values = silhouette_samples(data, labels)
        
        self.silhouette_by_labels = {}
        for k in labels_list:
            v = silhouette_values[k==labels]
            v.sort()
            self.silhouette_by_labels[k] = v
    
    def refresh(self):
        self.plot.clear()
        if self.silhouette_avg is None:
            return
        self.vline = pg.InfiniteLine(pos=self.silhouette_avg, angle = 90, movable = False, pen = '#FF0000')
        self.plot.addItem(self.vline)
        
        y_lower = 10
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in self.controller.cluster_visible.items() if v and c>=0]
        
        for k in visibles:
            v = self.silhouette_by_labels[k]
            
            color = self.controller.qcolors[k]
            color2 = QT.QColor(color)
            color2.setAlpha(self.alpha)
            
            y_upper = y_lower + v.size
            y_vect = np.arange(y_lower, y_upper)
            curve1 = pg.PlotCurveItem(np.zeros(v.size), y_vect, pen=color)
            curve2 = pg.PlotCurveItem(v, y_vect, pen=color)
            self.plot.addItem(curve1)
            self.plot.addItem(curve2)
            fill = pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=color2)
            self.plot.addItem(fill)
            
            txt = pg.TextItem( text='{}'.format(k), color='#FFFFFF', anchor=(0, 0.5), border=None)#, fill=pg.mkColor((128,128,128, 180)))
            self.plot.addItem(txt)
            txt.setPos(0, (y_upper+y_lower)/2.)
            
            y_lower = y_upper + 10

        
        self.plot.setXRange(-.5, 1.)
        self.plot.setYRange(0,y_lower)


    def on_spike_selection_changed(self):
        pass

    def on_spike_label_changed(self):
        self.compute_slihouette()
        self.refresh()
    
    def on_colors_changed(self):
        self.refresh()
    
    def on_cluster_visibility_changed(self):
        self.refresh()
        
