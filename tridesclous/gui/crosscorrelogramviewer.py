from .myqt import QT
import pyqtgraph as pg

import numpy as np
import pandas as pd

from .base import WidgetBase
from ..tools import compute_cross_correlograms


class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def raiseContextMenu(self, ev):
        #for some reasons enableMenu=False is not taken (bug ????)
        pass



class CrossCorrelogramViewer(WidgetBase):
    _params = [
                      {'name': 'window_size_ms', 'type': 'float', 'value' : 100. },
                      {'name': 'bin_size_ms', 'type': 'float', 'value' : 1.0 },
                      {'name': 'symmetrize', 'type': 'bool', 'value' : True },
                      {'name': 'display_axis', 'type': 'bool', 'value' : True },
                      {'name': 'max_visible', 'type': 'int', 'value' : 8 },
                      {'name': 'check_sorted', 'type': 'bool', 'value' : False },
        ]
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        h = QT.QHBoxLayout()
        self.layout.addLayout(h)

        but = QT.QPushButton('settings')
        but.clicked.connect(self.open_settings)
        h.addWidget(but)

        but = QT.QPushButton('compute')
        but.clicked.connect(self.compute_ccg)
        h.addWidget(but)
        
        self.grid = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.grid)
        
        self.ccg = None


    def on_params_changed(self):
        self.ccg = None
        self.refresh()
    
    def compute_ccg(self):
        cluster_labels = self.controller.positive_cluster_labels
        spikes = self.controller.spikes
        try:
            # this prevent some bug like this one https://github.com/tridesclous/tridesclous/issues/69
            self.ccg, self.bins = compute_cross_correlograms(spikes['index'], spikes['cluster_label'],
                        spikes['segment'], cluster_labels, self.controller.dataio.sample_rate,
                        window_size=self.params['window_size_ms']/1000.,
                        bin_size = self.params['bin_size_ms']/1000.,
                        symmetrize=self.params['symmetrize'],
                        check_sorted=self.params['check_sorted'],
                        )
        except:
            self.ccg, self.bins = None, None
        self.refresh()

    def refresh(self):
        self.grid.clear()
        
        if self.ccg is None:
            return
        
        visibles = [ ]
        for k in self.controller.positive_cluster_labels:
            if self.controller.cluster_visible[k]:
                visibles.append(k)
        
        visibles = visibles[:self.params['max_visible']]
        
        n = len(visibles)
        
        bins = self.bins * 1000. #to ms
        
        labels = self.controller.positive_cluster_labels.tolist()
        
        for r in range(n):
            for c in range(r, n):
                
                i = labels.index(visibles[r])
                j = labels.index(visibles[c])
                
                count = self.ccg[i, j, :]
                
                plot = pg.PlotItem()
                if not self.params['display_axis']:
                    plot.hideAxis('bottom')
                    plot.hideAxis('left')
                
                if r==c:
                    k = visibles[r]
                    color = self.controller.qcolors[k]
                else:
                    color = (120,120,120,120)
                
                curve = pg.PlotCurveItem(bins, count, stepMode=True, fillLevel=0, brush=color, pen=color)
                plot.addItem(curve)
                self.grid.addItem(plot, row=r, col=c)


#~ plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))

