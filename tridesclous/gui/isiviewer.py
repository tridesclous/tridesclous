from .myqt import QT
import pyqtgraph as pg

import numpy as np
import pandas as pd

from .base import WidgetBase



class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def raiseContextMenu(self, ev):
        #for some reasons enableMenu=False is not taken (bug ????)
        pass



class ISIViewer(WidgetBase):
    _params = [
                      {'name': 'bin_min', 'type': 'float', 'value' : 0. },
                      {'name': 'bin_max', 'type': 'float', 'value' : 100. },
                      {'name': 'bin_size', 'type': 'float', 'value' : 1.0 },
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
        
        #~ self.on_params_changed()#this do refresh    


    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        #~ self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        #ISI are computed on demand
        self.all_isi = {}
    
    def _compute_isi(self, k):
        spikes = self.controller.spikes
        
        isi = []
        for seg_num in range(self.controller.dataio.nb_segment):
            sel = (spikes['segment'] == seg_num) & (spikes['cluster_label'] == k)
            isi.append(np.diff(spikes[sel]['index'])/self.controller.dataio.sample_rate)
        isi = np.concatenate(isi)
        self.all_isi[k] = isi * 1000. #ms

    def refresh(self):
        self.plot.clear()
        
        n = 0
        for k in self.controller.positive_cluster_labels:
            if not self.controller.cluster_visible[k]:
                continue
            
            if k not in self.all_isi:
                self._compute_isi(k)
            
            isi = self.all_isi[k]
            if len(isi) ==0:
                return
            
            bins = np.arange(self.params['bin_min'], self.params['bin_max'], self.params['bin_size'])
            count, bins = np.histogram(isi, bins=bins)
            
            qcolor = self.controller.qcolors[k]
            curve = pg.PlotCurveItem(bins[:-1], count, pen=pg.mkPen(qcolor, width=3))
            self.plot.addItem(curve)


