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
    pass

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
        
        self.refresh()
        

    def create_settings(self):
        _params = [
                          #~ {'name': 'colormap', 'type': 'list', 'values' : ['viridis', 'jet', 'gray', 'hot', ] },
                          {'name': 'similarity_metric', 'type': 'list', 'values' : [ 'cosine_similarity',  'linear_kernel',
                                                                    'polynomial_kernel', 'sigmoid_kernel', 'rbf_kernel', 'laplacian_kernel' ] },
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
        pass

    def initialize_plot(self):
        self.viewBox = MyViewBox()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        #~ self.image = pg.ImageItem()
        #~ self.plot.addItem(self.image)
    
    def refresh(self):
        self.plot.clear()
        
        if self.params['data']=='waveforms':
            wf = self.controller.some_waveforms
            data = wf.reshape(wf.shape[0], -1)
        if self.params['data']=='features':
            data = self.controller.some_features
        
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in self.controller.cluster_visible.items() if v and c>=0]
        
        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep = labels>=0
        labels = labels[keep]
        data = data[keep]
        
        if np.unique(labels).size<=1:
            return
        
        silhouette_avg = silhouette_score(data, labels)

        self.vline = pg.InfiniteLine(pos=silhouette_avg, angle = 90, movable = False, pen = '#FF0000')
        self.plot.addItem(self.vline)

        
        silhouette_values = silhouette_samples(data, labels)
        
        y_lower = 10
        for k in visibles:
            v = silhouette_values[k==labels]
            #~ print(k, v.size)
            v.sort()
            
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
            
            
            y_lower = y_upper + 10

            #~ color2 = QT.QColor(color)
            #~ color2.setAlpha(self.alpha)
        
        self.plot.setXRange(-.5,1.)
        self.plot.setYRange(0,y_lower)
            
            
        
        
    
        #~ ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        #~ ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    