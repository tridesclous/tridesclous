"""
This view is from taken from sklearn examples.
See http://scikit-learn.org: plot-kmeans-silhouette-analysis-py



"""
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



class Silhouette(WidgetBase):
    """
    **Silhouette**  display the silhouette score.
    
    Implemented with sklearn. Must compute metrics first.
    
    See:
      * `Silhouette wikipedia <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
      * `Silhouette sklearn <http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py>`_
    """
    
    _params = [
        ]
    
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
        
        self.initialize_plot()
        
        self.refresh()
        
    def on_params_changed(self):
        self.compute_slihouette()
        self.refresh()

    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
    
    def refresh(self):
        self.plot.clear()
        silhouette_values = self.controller.spike_silhouette
        if silhouette_values is None:
            return
        
        if silhouette_values.shape != self.controller.spike_label.shape:
            return
        
        silhouette_avg = np.mean(silhouette_values)
        silhouette_by_labels = {}
        labels = self.controller.spike_label
        labels_list = np.unique(labels)
        for k in labels_list:
            v = silhouette_values[k==labels]
            v.sort()
            silhouette_by_labels[k] = v
        
        
        self.vline = pg.InfiniteLine(pos=silhouette_avg, angle = 90, movable = False, pen = '#FF0000')
        self.plot.addItem(self.vline)
        
        y_lower = 10
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in self.controller.cluster_visible.items() if v and c>=0]
        
        for k in visibles:
            if k not in silhouette_by_labels:
                continue
            v = silhouette_by_labels[k]
            
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
        #~ self.compute_slihouette()
        self.refresh()
    
    def on_colors_changed(self):
        self.refresh()
    
    def on_cluster_visibility_changed(self):
        self.refresh()
        
