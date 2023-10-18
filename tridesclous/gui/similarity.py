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
    def raiseContextMenu(self, ev):
        #for some reasons enableMenu=False is not taken (bug ????)
        pass



class BaseSimilarityView(WidgetBase):
    _params = [
                      {'name': 'colormap', 'type': 'list', 'limits' : ['viridis', 'jet', 'gray', 'hot', ] },
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
        
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        
        self._text_items = []


    @property
    def similarity(self):
        raise(NotImplementedError)

    def on_spike_selection_changed(self):
        pass
    
    def on_spike_label_changed(self):
        self.refresh()
    
    def on_colors_changed(self):
        pass
    
    def on_cluster_visibility_changed(self):
        self.refresh()

class SpikeSimilarityView(BaseSimilarityView):
    """
    **Spike similarity view** dispplay the spike-to-spike similarity. Only visible
    cluster are shown.
    
    If nothing appear means : metrics are not computed yet or the size of the 
    similarity is too big (over **max__size**).
    """
    
    @property
    def similarity(self):
        if self.controller.spike_waveforms_similarity is not None:
            return self.controller.spike_waveforms_similarity.copy()

    def refresh(self):
        if self.similarity is None:
            self.image.hide()
            return
        
        _max = np.max(self.similarity)
        
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in self.controller.cluster_visible.items() if v and c>=0]
        
        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep_ind,  = np.nonzero(np.isin(labels, visibles))
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
        


class BaseClusterSimilarityView(BaseSimilarityView):
    def refresh(self):
        if self.similarity is None:
            self.image.hide()
            return
        
        _max = np.max(self.similarity)

        s = self.similarity
        #~ _max = np.max(s)
        _max = 1
        
        self.image.setImage(s, lut=self.lut, levels=[0, _max])
        self.image.show()
        self.plot.setXRange(0, s.shape[0])
        self.plot.setYRange(0, s.shape[1])

        for item in self._text_items:
            self.plot.removeItem(item)
        
        for pos, k in enumerate(self.controller.positive_cluster_labels):
            for i in range(2):
                item = pg.TextItem(text='{}'.format(k), color='#FFFFFF', anchor=(0.5, 0.5), border=None)
                self.plot.addItem(item)
                if i==0:
                    item.setPos(pos+0.5, 0)
                else:
                    item.setPos(0, pos+0.5)
                self._text_items.append(item)


class ClusterSimilarityView(BaseClusterSimilarityView):
    """
    **Cluster similarity view** dispplay the clsuter-to-cluster similarity.
    
    If nothing appear means : metrics are not computed yet.
    """
    
    @property
    def similarity(self):
        if self.controller.cluster_similarity is not None:
            return self.controller.cluster_similarity.copy()

class ClusterRatioSimilarityView(BaseClusterSimilarityView):
    """
    **Cluster similarity ratio view** dispplay the clsuter-to-cluster ratio similarity.
    
    If nothing appear means : metrics are not computed yet.
    """
    
    @property
    def similarity(self):
        if self.controller.cluster_ratio_similarity is not None:
            return self.controller.cluster_ratio_similarity.copy()

