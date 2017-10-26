from .myqt import QT
import pyqtgraph as pg

import numpy as np
import matplotlib.cm
import matplotlib.colors

from .base import WidgetBase
from .tools import ParamDialog

class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()

class SimilarityView(WidgetBase):

    _params = [
                      {'name': 'colormap', 'type': 'list', 'values' : ['viridis', 'jet', 'gray', 'hot', ] },
        ]
    
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
        
        self.initialize_plot()
        
        self.on_params_changed()#this do refresh

    def on_params_changed(self):
        N = 512
        cmap_name = self.params['colormap']
        cmap = matplotlib.cm.get_cmap(cmap_name , N)
        
        lut = []
        for i in range(N):
            r,g,b,_ =  matplotlib.colors.ColorConverter().to_rgba(cmap(i))
            lut.append([r*255,g*255,b*255])
        self.lut = np.array(lut, dtype='uint8')
        
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
    
    @property
    def similarity(self):
        return self.controller.spike_waveforms_similarity
    
    def refresh(self):
        if self.similarity is None:
            self.image.hide()
            return
        
        _max = np.max(self.similarity)
        
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in self.controller.cluster_visible.items() if v and c>=0]
        
        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep_ind,  = np.nonzero(np.in1d(labels, visibles))
        keep_label = labels[keep_ind]
        order = np.argsort(keep_label)
        keep_ind = keep_ind[order]
        
        if keep_ind.size>0:
            s = self.similarity[keep_ind,:][:, keep_ind]
            self.image.setImage(s, lut=self.lut, levels=[0, _max])
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
        self.refresh()
    
    def on_colors_changed(self):
        pass
    
    def on_cluster_visibility_changed(self):
        self.refresh()
