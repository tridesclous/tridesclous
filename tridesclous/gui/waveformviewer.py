import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .base import WidgetBase


class MyViewBox(pg.ViewBox):
    pass

class WaveformViewer(WidgetBase):
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
        
        self.alpha = 60
        self.refresh()
        
    
    def initialize_plot(self):
        self.viewBox1 = MyViewBox()
        #~ self.viewBox.disableAutoRange()

        grid = pg.GraphicsLayout(border=(100,100,100))
        self.graphicsview.setCentralItem(grid)
        
        self.plot1 = grid.addPlot(row=0, col=0, rowspan=2, viewBox=self.viewBox1)
        self.plot1.hideButtons()
        self.plot1.showAxis('left', True)
        
        grid.nextRow()
        grid.nextRow()
        self.viewBox2 = MyViewBox()
        self.plot2 = grid.addPlot(row=2, col=0, rowspan=1, viewBox=self.viewBox2)
        self.plot2.hideButtons()
        self.plot2.showAxis('left', True)
        
        self.viewBox2.setXLink(self.viewBox1)
        
        
        #~ self.viewBox.gain_zoom.connect(self.gain_zoom)
        #~ self.viewBox.xsize_zoom.connect(self.xsize_zoom)    
    
    
    def refresh(self):
        self.plot1.clear()
        self.plot2.clear()
        
        if self.controller.spike_index ==[]:
            return
        
        #lines
        def addSpan(plot):
        
            nb_channel = self.controller.dataio.nb_channel
            #~ n_left, n_right = min(samples)+2, max(samples)-1
            
            d = self.controller.info['params_waveformextractor']
            n_left, n_right = d['n_left'], d['n_right']
            white = pg.mkColor(255, 255, 255, 20)
            width = n_right - n_left
            for i in range(nb_channel):
                if i%2==1:
                    region = pg.LinearRegionItem([width*i, width*(i+1)-1], movable = False, brush = white)
                    plot.addItem(region, ignoreBounds=True)
                    for l in region.lines:
                        l.setPen(white)
                    
                vline = pg.InfiniteLine(pos = -n_left + width*i, angle=90, movable=False, pen = pg.mkPen('w'))
                plot.addItem(vline)
        
        addSpan(self.plot1)
        addSpan(self.plot2)
        
        #waveforms
        for i,k in enumerate(self.controller.centroids):
            if not self.controller.cluster_visible[k]:
                continue
            wf0 = self.controller.centroids[k]['median'].T.flatten()
            mad = self.controller.centroids[k]['mad'].T.flatten()
            
            color = self.controller.qcolors.get(k, QtGui.QColor( 'white'))
            curve = pg.PlotCurveItem(np.arange(wf0.size), wf0, pen=pg.mkPen(color, width=2))
            self.plot1.addItem(curve)
            
            color2 = QtGui.QColor(color)
            color2.setAlpha(self.alpha)
            curve1 = pg.PlotCurveItem(np.arange(wf0.size), wf0+mad, pen=color2)
            curve2 = pg.PlotCurveItem(np.arange(wf0.size), wf0-mad, pen=color2)
            self.plot1.addItem(curve1)
            self.plot1.addItem(curve2)
            fill = pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=color2)
            self.plot1.addItem(fill)
            
            curve = pg.PlotCurveItem(np.arange(wf0.size), mad, pen=color)
            self.plot2.addItem(curve)

        

        
            
            
            
            #~ ax.axvline(-n_left + width*i, alpha = .05, color = 'k')
        
        

    def on_peak_selection_changed(self):
        pass
        #TODO peak the selected peak if only one
