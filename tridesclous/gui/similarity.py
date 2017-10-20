from .myqt import QT
import pyqtgraph as pg

import numpy as np
import matplotlib.cm
import matplotlib.colors


import sklearn.metrics.pairwise
#~ from sklearn.metrics.pairwise import (linear_kernel, cosine_similarity, 
                #~ polynomial_kernel, sigmoid_kernel, rbf_kernel, laplacian_kernel)

from .base import WidgetBase
from .tools import ParamDialog

class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()

class SimilarityView(WidgetBase):
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        #~ h = QT.QHBoxLayout()
        #~ self.layout.addLayout(h)
        #~ h.addWidget(QT.QLabel('<b>Similarity</b>') )

        #~ but = QT.QPushButton('settings')
        #~ but.clicked.connect(self.open_settings)
        #~ h.addWidget(but)
        
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        
        self.create_settings()
        
        self.initialize_plot()
        self.similarity = None
        
        self.on_params_change()#this do refresh
        #~ self.refresh()

    def create_settings(self):
        _params = [
                          {'name': 'colormap', 'type': 'list', 'values' : ['viridis', 'jet', 'gray', 'hot', ] },
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
        N = 512
        cmap_name = self.params['colormap']
        cmap = matplotlib.cm.get_cmap(cmap_name , N)
        
        lut = []
        for i in range(N):
            r,g,b,_ =  matplotlib.colors.ColorConverter().to_rgba(cmap(i))
            lut.append([r*255,g*255,b*255])
        self.lut = np.array(lut, dtype='uint8')
        
        self.compute_similarity()
        #~ self.similarity = None
        
        self.refresh()
    
    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        
        self._text_items = []
    
    def compute_similarity(self):
        #~ print('compute_similarity')
        if self.params['data']=='waveforms':
            wf = self.controller.some_waveforms
            if wf is not None:
                feat = wf.reshape(wf.shape[0], -1)
            else:
                feat = None
        if self.params['data']=='features':
            feat = self.controller.some_features

        if feat is None:
            self.similarity = None
            return
        
        
        if feat.size>1e6:
            print('compute_similarity: TOO BIG')
            #~ print(feat.size)
            self.similarity = None
        else:
            func = getattr(sklearn.metrics.pairwise, self.params['similarity_metric'])
            
            self.similarity = func(feat)
            self._max = np.max(self.similarity)
            #~ print('compute_similarity DONE')
    
    
    def refresh(self):
        if self.similarity is None:
            self.image.hide()
            return
            
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in self.controller.cluster_visible.items() if v and c>=0]
        
        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep_ind,  = np.nonzero(np.in1d(labels, visibles))
        keep_label = labels[keep_ind]
        order = np.argsort(keep_label)
        keep_ind = keep_ind[order]
        
        if keep_ind.size>0:
            s = self.similarity[keep_ind,:][:, keep_ind]
            self.image.setImage(s, lut=self.lut, levels=[0, self._max])
            self.image.show()
            self.plot.setXRange(0, s.shape[0])
            self.plot.setYRange(0, s.shape[1])
            
            for item in self._text_items:
                self.plot.removeItem(item)
            
            pos = 0
            for k in np.sort(visibles):
                n = np.sum(keep_label==k)
                for i in range(2):
                    item = pg.TextItem(text='{}'.format(k), color='#FFFFFF', anchor=(0.5, 0.5), border=None)
                    self.plot.addItem(item)
                    if i==0:
                        item.setPos(pos+n/2., 0)
                    else:
                        item.setPos(0, pos+n/2.)
                    self._text_items.append(item)
                pos += n
                
        else:
            self.image.hide()
        
    def on_spike_selection_changed(self):
        pass
    
    def on_spike_label_changed(self):
        self.compute_similarity()
        #~ self.similarity = None
        self.refresh()
    
    def on_colors_changed(self):
        pass
    
    def on_cluster_visibility_changed(self):
        self.refresh()
