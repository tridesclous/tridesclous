import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .base import WidgetBase


class MyViewBox(pg.ViewBox):
    pass

class CatalogueViewer(WidgetBase):
    
    def __init__(self, spikesorter = None, parent=None):
        WidgetBase.__init__(self, parent)
    
        self.spikesorter = spikesorter
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
        
        self.alpha = 60
        self.refresh()
        
    
    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        self.plot.showAxis('left', False)
        
        #~ self.viewBox.gain_zoom.connect(self.gain_zoom)
        #~ self.viewBox.xsize_zoom.connect(self.xsize_zoom)    
    
    @property
    def catalogue(self):
        return self.spikesorter.clustering.catalogue
    
    def refresh(self):
        self.plot.clear()
        
        #lines
        nb_channel = self.spikesorter.dataio.nb_channel
        samples = self.spikesorter.all_waveforms.columns.levels[1]
        n_left, n_right = min(samples)+2, max(samples)-1
        white = pg.mkColor(255, 255, 255, 20)
        width = n_right - n_left
        for i in range(nb_channel):
            if i%2==1:
                region = pg.LinearRegionItem([width*i, width*(i+1)-1], movable = False, brush = white)
                self.plot.addItem(region, ignoreBounds=True)
                for l in region.lines:
                    l.setPen(white)
                
            vline = pg.InfiniteLine(pos = -n_left + width*i, angle=90, movable=False, pen = pg.mkPen('w'))
            self.plot.addItem(vline)
        
        #waveforms
        for i,k in enumerate(self.catalogue):
            if not self.spikesorter.cluster_visible[k]:
                continue
            wf0 = self.catalogue[k]['center']
            mad = self.catalogue[k]['mad']
            
            color = self.spikesorter.qcolors.get(k, QtGui.QColor( 'white'))
            curve = pg.PlotCurveItem(np.arange(wf0.size), wf0, pen=pg.mkPen(color, width=2))
            self.plot.addItem(curve)
            
            color2 = QtGui.QColor(color)
            color2.setAlpha(self.alpha)
            curve1 = pg.PlotCurveItem(np.arange(wf0.size), wf0+mad, pen=color2)
            curve2 = pg.PlotCurveItem(np.arange(wf0.size), wf0-mad, pen=color2)
            self.plot.addItem(curve1)
            self.plot.addItem(curve2)
            fill = pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=color2)
            self.plot.addItem(fill)


        

        
            
            
            
            #~ ax.axvline(-n_left + width*i, alpha = .05, color = 'k')
        
        

    def on_peak_selection_changed(self):
        pass
        #TODO peak the selected peak if only one
