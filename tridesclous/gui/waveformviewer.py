import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .base import WidgetBase


class MyViewBox(pg.ViewBox):
    doubleclicked = QtCore.pyqtSignal()
    gain_zoom = QtCore.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #~ self.disableAutoRange()
    def mouseClickEvent(self, ev):
        ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    #~ def mouseDragEvent(self, ev):
        #~ ev.ignore()
    def wheelEvent(self, ev):
        if ev.modifiers() == QtCore.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()


class WaveformViewer(WidgetBase):
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.create_settings()
        
        self.create_toolbar()
        self.layout.addWidget(self.toolbar)

        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
        
        self.alpha = 60
        self.refresh()

    def create_settings(self):
        _params = [{'name': 'plot_selected_spike', 'type': 'bool', 'value': True },
                          {'name': 'plot_limit_for_flatten', 'type': 'bool', 'value': True },
                          {'name': 'metrics', 'type': 'list', 'values': ['median/mad', 'mean/std'] },
                          {'name': 'fillbetween', 'type': 'bool', 'value': True },
                          
                          ]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        
        if self.controller.nb_channel>16:
            self.params['fillbetween'] = False
            self.params['plot_limit_for_flatten'] = False
        
        self.params.sigTreeStateChanged.connect(self.refresh)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for waveforms viewer')
        self.tree_params.setWindowFlags(QtCore.Qt.Window)
        
    
    def create_toolbar(self):
        tb = self.toolbar = QtGui.QToolBar()
        
        #Mode flatten or geometry
        self.combo_mode = QtGui.QComboBox()
        tb.addWidget(self.combo_mode)
        #~ self.mode = 'flatten'
        #~ self.combo_mode.addItems([ 'flatten', 'geometry'])
        self.mode = 'geometry'
        self.combo_mode.addItems([ 'geometry', 'flatten'])
        self.combo_mode.currentIndexChanged.connect(self.on_combo_mode_changed)
        tb.addSeparator()
        
        
        but = QtGui.QPushButton('settings')
        but.clicked.connect(self.open_settings)
        tb.addWidget(but)

        but = QtGui.QPushButton('refresh')
        but.clicked.connect(self.refresh)
        tb.addWidget(but)
    
        
    
    def on_combo_mode_changed(self):
        self.mode = str(self.combo_mode.currentText())
        self.initialize_plot()
        self.refresh()
    
    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()

    
    def initialize_plot(self):
        self.viewBox1 = MyViewBox()
        self.viewBox1.disableAutoRange()

        grid = pg.GraphicsLayout(border=(100,100,100))
        self.graphicsview.setCentralItem(grid)
        
        
        self.plot1 = grid.addPlot(row=0, col=0, rowspan=2, viewBox=self.viewBox1)
        self.plot1.hideButtons()
        self.plot1.showAxis('left', True)
        
        if self.mode=='flatten':
            grid.nextRow()
            grid.nextRow()
            self.viewBox2 = MyViewBox()
            self.viewBox2.disableAutoRange()
            self.plot2 = grid.addPlot(row=2, col=0, rowspan=1, viewBox=self.viewBox2)
            self.plot2.hideButtons()
            self.plot2.showAxis('left', True)
            self.viewBox2.setXLink(self.viewBox1)
            self.factor_y = 1.
            
        elif self.mode=='geometry':
            self.plot2 = None
            
            chan_grp = self.controller.chan_grp
            channel_group = self.controller.dataio.channel_groups[chan_grp]
            #~ print(channel_group['geometry'])
            if channel_group['geometry'] is None:
                print('no geometry')
                self.xvect = None
            else:

                shape = list(self.controller.centroids.values())[0]['median'].shape
                width = shape[0]
                
                self.xvect = np.zeros(shape[0]*shape[1], dtype='float32')

                self.arr_geometry = []
                for i, chan in enumerate(channel_group['channels']):
                    x, y = channel_group['geometry'][chan]
                    self.arr_geometry.append([x, y])
                self.arr_geometry = np.array(self.arr_geometry)
                
                xpos = self.arr_geometry[:,0]
                ypos = self.arr_geometry[:,1]
                
                if np.unique(xpos).size>1:
                    print('yep')
                    self.delta_x = np.min(np.diff(np.sort(np.unique(xpos))))
                else:
                    self.delta_x = np.unique(xpos)[0]
                if np.unique(ypos).size>1:
                    self.delta_y = np.min(np.diff(np.sort(np.unique(ypos))))
                else:
                    self.delta_y = np.unique(ypos)[0]
                self.factor_y = .3
                #~ print(np.sort(np.unique(xpos)))
                #~ print(np.sort(np.unique(ypos)))
                #~ print('self.delta_x',self.delta_x )
                #~ print('self.delta_y',self.delta_y)
                if self.delta_x>0.:
                    espx = self.delta_x/2. *.95
                else:
                    espx = .5
                for i, chan in enumerate(channel_group['channels']):
                    x, y = channel_group['geometry'][chan]
                    self.xvect[i*width:(i+1)*width] = np.linspace(x-espx, x+espx, num=width)
                self.arr_geometry = np.array(self.arr_geometry)

                
                
            
        
        #~ for k, centroid in self.controller.centroids.items():
            #~ print(k)
            #~ print(centroid['mad'])
            
        self.wf_min = min(np.min(centroid['median']) for centroid in self.controller.centroids.values())
        self.wf_max = max(np.max(centroid['median']) for centroid in self.controller.centroids.values())

        
        self.viewBox1.gain_zoom.connect(self.gain_zoom)
        
        self.viewBox1.doubleclicked.connect(self.open_settings)
        
        #~ self.viewBox.xsize_zoom.connect(self.xsize_zoom)    
    

    def gain_zoom(self, factor_ratio):
        self.factor_y *= factor_ratio
        
        self.refresh()
    
    def refresh(self):
        if self.mode=='flatten':
            self.refresh_mode_flatten()
        elif self.mode=='geometry':
            self.refresh_mode_geometry()
        
        self.curve_one_waveform = None
    
    def refresh_mode_flatten(self):
        self.plot1.clear()
        self.plot2.clear()
        
        if self.controller.spike_index ==[]:
            return
        
        #lines
        def addSpan(plot):
        
            nb_channel = self.controller.nb_channel
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
        
        if self.params['plot_limit_for_flatten']:
            addSpan(self.plot1)
            addSpan(self.plot2)
        
        #waveforms
        
        if self.params['metrics']=='median/mad':
            key1, key2 = 'median', 'mad'
        elif self.params['metrics']=='mean/std':
            key1, key2 = 'mean', 'std'
        
        
        shape = list(self.controller.centroids.values())[0]['median'].shape
        xvect = np.arange(shape[0]*shape[1])
        
        for i,k in enumerate(self.controller.centroids):
            if not self.controller.cluster_visible[k]:
                continue
            
            wf0 = self.controller.centroids[k][key1].T.flatten()
            mad = self.controller.centroids[k][key2].T.flatten()
            
            color = self.controller.qcolors.get(k, QtGui.QColor( 'white'))
            curve = pg.PlotCurveItem(np.arange(wf0.size), wf0, pen=pg.mkPen(color, width=2))
            self.plot1.addItem(curve)
            
            
            if self.params['fillbetween']:
                color2 = QtGui.QColor(color)
                color2.setAlpha(self.alpha)
                curve1 = pg.PlotCurveItem(xvect, wf0+mad, pen=color2)
                curve2 = pg.PlotCurveItem(xvect, wf0-mad, pen=color2)
                self.plot1.addItem(curve1)
                self.plot1.addItem(curve2)
                
                fill = pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=color2)
                self.plot1.addItem(fill)
            
            curve = pg.PlotCurveItem(xvect, mad, pen=color)
            self.plot2.addItem(curve)        
        
        self.plot1.setXRange(xvect[0], xvect[-1], padding = 0.0)
        self.plot2.setXRange(xvect[0], xvect[-1], padding = 0.0)
        
        self.plot1.setYRange(self.wf_min*1.1, self.wf_max*1.1, padding = 0.0)
        self.plot2.setYRange(0., 5., padding = 0.0)

        

    def refresh_mode_geometry(self):
        self.plot1.clear()
        
        if self.xvect is None:
            return
        
        if self.params['metrics']=='median/mad':
            key1, key2 = 'median', 'mad'
        elif self.params['metrics']=='mean/std':
            key1, key2 = 'mean', 'std'

        ypos = self.arr_geometry[:,1]
        
        for i,k in enumerate(self.controller.centroids):
            if not self.controller.cluster_visible[k]:
                continue
            
            wf = self.controller.centroids[k][key1]
            wf = wf*self.factor_y*self.delta_y + ypos[None, :]
            wf[0,:] = np.nan
            wf = wf.T.reshape(-1)
            
            color = self.controller.qcolors.get(k, QtGui.QColor( 'white'))
            curve = pg.PlotCurveItem(self.xvect, wf, pen=pg.mkPen(color, width=2), connect='finite')
            self.plot1.addItem(curve)
            
        
        self.plot1.setXRange(np.min(self.xvect), np.max(self.xvect), padding = 0.0)
        self.plot1.setYRange(np.min(ypos)-self.delta_y*2, np.max(ypos)+self.delta_y*2, padding = 0.0)
        

    def on_spike_selection_changed(self):
        pass
        #TODO peak the selected peak if only one


        n_selected = np.sum(self.controller.spike_selection)
        if n_selected!=1 or not self.params['plot_selected_spike']: 
            
            if self.curve_one_waveform is not None:
                self.plot1.remove(self.curve_one_waveform)
            
            return
        
        if self.curve_one_waveform is None:
            self.curve_one_waveform = pg.PlotCurveItem([], [], pen=pg.mkPen(QtGui.QColor( 'white'), width=1), connect='finite')
            self.plot1.addItem(self.curve_one_waveform)
        
        ind, = np.nonzero(self.controller.spike_selection)
        ind = ind[0]
        seg_num = self.controller.spike_segment[ind]
        peak_ind = self.controller.spike_index[ind]
        
        d = self.controller.info['params_waveformextractor']
        n_left, n_right = d['n_left'], d['n_right']
        
        wf = self.controller.dataio.get_signals_chunk(seg_num=seg_num, chan_grp=self.controller.chan_grp,
                i_start=peak_ind+n_left, i_stop=peak_ind+n_right,
                signal_type='processed', return_type='raw_numpy')
        
        
        
        if self.mode=='flatten':
            wf = wf.T.flatten()
            xvect = np.arange(wf.size)
            self.curve_one_waveform.setData(xvect, wf)
        elif self.mode=='geometry':
            ypos = self.arr_geometry[:,1]
            wf = wf*self.factor_y*self.delta_y + ypos[None, :]
            wf[0,:] = np.nan
            wf = wf.T.reshape(-1)
            self.curve_one_waveform.setData(self.xvect, wf)
            

        
        

        



