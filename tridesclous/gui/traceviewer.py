import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .tools import TimeSeeker
from ..tools import median_mad




class MyViewBox(pg.ViewBox):
    doubleclicked = QtCore.pyqtSignal()
    gain_zoom = QtCore.pyqtSignal(float)
    xsize_zoom = QtCore.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #~ self.disableAutoRange()
    def mouseClickEvent(self, ev):
        ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.ignore()
    def wheelEvent(self, ev):
        if ev.modifiers() == QtCore.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.accept()
        self.xsize_zoom.emit((ev.pos()-ev.lastPos()).x())


class TraceViewer(QtGui.QWidget):
    
    peak_selection_changed = QtCore.pyqtSignal()
    
    def __init__(self, spikesorter = None, shared_view_with = [], 
                    mode = 'memory', parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.spikesorter = spikesorter
        self.dataio = self.spikesorter.dataio
        self.mode = mode
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        # Can share view with other trace viewer : simultanetly view filter + full band
        self.shared_view_with = shared_view_with

        self.create_toolbar()
        self.layout.addWidget(self.toolbar)
        
        # create graphic view and plot item
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
            
        
        #handle time by segments
        self.time_by_seg = pd.Series(self.dataio.segments['t_start'].copy(), name = 'time', index = self.dataio.segments.index)
        
        
        _params = [{'name': 'auto_zoom_on_select', 'type': 'bool', 'value': True },
                           {'name': 'zoom_size', 'type': 'float', 'value':  0.08, 'step' : 0.001 },
                          {'name': 'plot_threshold', 'type': 'bool', 'value':  True },]        
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for signal viewer')
        self.tree_params.setWindowFlags(QtCore.Qt.Window)        
        
        self.change_segment(0)
        self.refresh()
    
    def create_toolbar(self):
        tb = self.toolbar = QtGui.QToolBar()
        
        #Segment selection
        #~ but = QtGui.QPushButton('<')
        #~ but.clicked.connect(self.prev_segment)
        #~ tb.addWidget(but)
        self.combo = QtGui.QComboBox()
        tb.addWidget(self.combo)
        self.combo.addItems([ 'Segment {}'.format(seg_num) for seg_num in self.dataio.segments.index ])
        #~ but = QtGui.QPushButton('>')
        #~ but.clicked.connect(self.next_segment)
        #~ tb.addWidget(but)
        self._seg_pos = 0
        self.seg_num = self.dataio.segments.index[self._seg_pos]
        self.combo.currentIndexChanged.connect(self.on_combo_changed)
        tb.addSeparator()

        # time slider
        self.timeseeker = TimeSeeker()
        tb.addWidget(self.timeseeker)
        self.timeseeker.time_changed.connect(self.seek)
        
        # winsize
        self.xsize = .5
        tb.addWidget(QtGui.QLabel(u'X size (s)'))
        self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 10.], suffix = 's', siPrefix = True, step = 0.1, dec = True)
        #~ self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 10.]) # step = 0.1, dec = True)
        self.spinbox_xsize.sigValueChanged.connect(self.xsize_changed)
        tb.addWidget(self.spinbox_xsize)
        tb.addSeparator()
        self.spinbox_xsize.sigValueChanged.connect(self.refresh)
        
        #
        but = QtGui.QPushButton('auto scale')
        but.clicked.connect(self.auto_scale)
        tb.addWidget(but)
        but = QtGui.QPushButton('settings')
        but.clicked.connect(self.open_settings)
        tb.addWidget(but)
        self.select_button = QtGui.QPushButton('select', checkable = True)
        tb.addWidget(self.select_button)
    

    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        self.plot.showAxis('left', False)
        
        self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.viewBox.xsize_zoom.connect(self.xsize_zoom)
        
        
        self.curves = []
        self.channel_labels = []
        self.threshold_lines =[]
        self.scatters = []
        for c in range(self.dataio.nb_channel):
            color = '#7FFF00'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves.append(curve)
            label = pg.TextItem(str(self.dataio.info['channels'][c]), color=color, anchor=(0, 0.5), border=None, fill=pg.mkColor((128,128,128, 180)))
            self.plot.addItem(label)
            self.channel_labels.append(label)
            
            tc = pg.InfiniteLine(angle = 0, movable = False, pen = 'g')
            tc.setPos(0.)
            self.threshold_lines.append(tc)
            self.plot.addItem(tc)
            tc.hide()
            
            self.scatters.append({})
        
        self.gains = None
        self.offsets = None

    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()
        

    def prev_segment(self):
        self.change_segment(self._seg_pos - 1)
        
    def next_segment(self):
        self.change_segment(self._seg_pos + 1)

    def change_segment(self, seg_pos):
        self._seg_pos  =  seg_pos
        if self._seg_pos<0:
            self._seg_pos = self.dataio.segments.shape[0]-1
        if self._seg_pos == self.dataio.segments.shape[0]:
            self._seg_pos = 0
        self.seg_num = self.dataio.segments.index[self._seg_pos]
        self.combo.setCurrentIndex(self._seg_pos)
        
        lims = self.dataio.segments.loc[self.seg_num] 
        self.timeseeker.set_start_stop(lims['t_start'], lims['t_stop'], seek = False)
        self.timeseeker.seek(self.time_by_seg[self.seg_num], emit = False)
        
        if self.mode == 'memory':
            #TODO filtered or not
            self.sigs = self.dataio.get_signals(seg_num = self.seg_num, t_start = lims['t_start'], 
                            t_stop = lims['t_stop'], filtered = True)
        elif self.mode == 'file':
            self.sigs = None

        if self.spikesorter.all_peaks is None:
            self.spikesorter.load_all_peaks()
        self.seg_peaks = self.spikesorter.all_peaks.xs(self.seg_num)
            
        if self.isVisible():
            self.refresh()

    
    def on_combo_changed(self):
        s =  self.combo.currentIndex()
        for otherviewer in self.shared_view_with:
            otherviewer.combo.setCurrentIndex(s)
        self.change_segment(s)
    
    def xsize_changed(self):
        self.xsize = self.spinbox_xsize.value()
        for otherviewer in self.shared_view_with:
            otherviewer.spinbox_xsize.setValue(self.xsize)
        if self.isVisible():
            self.refresh()
    
    def seek(self, t, cascade=True):
        if cascade:
            for otherviewer in self.shared_view_with:
                otherviewer.seek(t, cascade = False)
        else:
            self.timeseeker.seek(t, emit = False)
            
        self.time_by_seg[self.seg_num] = t
        t1,t2 = t-self.xsize/3. , t+self.xsize*2/3.
        #~ print(t1, t2)
        
        if self.gains is None:
            self.estimate_auto_scale()

        
        #signal chunk
        if self.mode == 'memory':
            chunk = self.sigs.loc[t1:t2]
        elif self.mode == 'file':
            chunk = self.dataio.get_signals(seg_num = self.seg_num, t_start = t1, t_stop = t2, filtered = True)
        
        for c in range(self.dataio.nb_channel):
            self.curves[c].setData(chunk.index.values, chunk.iloc[:, c].values*self.gains[c]+self.offsets[c])
            self.channel_labels[c].setPos(t1, self.dataio.nb_channel-c-1)
        
        inwin = self.seg_peaks.loc[t1:t2,:]
        visible_labels = np.unique(inwin['label'].values)
        for c in range(self.dataio.nb_channel):
            #reset scatters
            for k, scatter in self.scatters[c].items():
                if k not in visible_labels:
                    scatter.setData([], [])
            # plotted selected
            if 'sel' not in self.scatters[c]:
                color = QtGui.QColor( 'magenta')
                color.setAlpha(180)
                self.scatters[c]['sel'] = pg.ScatterPlotItem(pen=None, brush=color, size=20, pxMode = True)
                self.plot.addItem(self.scatters[c]['sel'])
            sel = inwin[inwin['selected']]
            p = chunk.loc[sel.index]
            self.scatters[c]['sel'].setData(p.index.values, p.iloc[:,c].values*self.gains[c]+self.offsets[c])
        
        for k in visible_labels:
            for c in range(self.dataio.nb_channel):
                sel = inwin['label']==k
                p = chunk.loc[sel.index]
                
                if k not in self.scatters[c]:
                    #TODO color
                    color = QtGui.QColor( 'cyan')
                    self.scatters[c][k] = pg.ScatterPlotItem(pen=None, brush=color, size=10, pxMode = True)
                    self.plot.addItem(self.scatters[c][k])
                    self.scatters[c][k].sigClicked.connect(self.item_clicked)
                
                self.scatters[c][k].setData(p.index.values, p.iloc[:,c].values*self.gains[c]+self.offsets[c])
                
        
        self.plot.setXRange( t1, t2, padding = 0.0)
        self.plot.setYRange(-.5, self.dataio.nb_channel-.5, padding = 0.0)
        
    def refresh(self):
        self.seek(self.time_by_seg[self.seg_num], cascade = False)

    def xsize_zoom(self, xmove):
        factor = xmove/100.
        newsize = self.xsize*(factor+1.)
        limits = self.spinbox_xsize.opts['bounds']
        if newsize>0. and newsize<limits[1]:
            self.spinbox_xsize.setValue(newsize)
    
    def auto_scale(self):
        self.estimate_auto_scale()
        self.refresh()
    
    def estimate_auto_scale(self):
        if self.mode == 'memory':
            self.med, self.mad = median_mad(self.sigs, axis = 0)
        elif self.mode == 'file':
            lims = self.dataio.segments.loc[self.seg_num]
            chunk = self.dataio.get_signals(seg_num = self.seg_num, t_start = lims['t_start'], t_stop = lims['t_start']+60., filtered = True)
            self.med, self.mad = median_mad(chunk, axis = 0)
        
        self.med, self.mad = self.med.values, self.mad.values
        n = self.dataio.nb_channel
        factor = 15
        self.gains = np.ones(n, dtype=float) * 1./(factor*max(self.mad))
        self.offsets = np.arange(n)[::-1] - self.med*self.gains
    
    def gain_zoom(self, factor):
        self.offsets = self.offsets + self.med*self.gains - self.med*self.gains*factor
        self.gains = self.gains*factor
        self.refresh()
    
    def on_peak_selection_changed(self):
        selected_peaks = self.spikesorter.all_peaks[self.spikesorter.all_peaks['selected']]
        if self.params['auto_zoom_on_select'] and selected_peaks.shape[0]==1:
            seg_num, time= selected_peaks.index[0]
            if seg_num != self.seg_num:
                seg_pos = self.dataio.segments.index.tolist().index(seg_num)
                self.combo.setCurrentIndex(seg_pos)
            self.spinbox_xsize.setValue(self.params['zoom_size'])
            self.seek(time)
        else:
            self.refresh()
        
    def item_clicked(self, plot, points):
        if self.select_button.isChecked()and len(points)==1:
            x = points[0].pos().x()
            self.spikesorter.all_peaks['selected'] = False
            self.spikesorter.all_peaks.loc[(self.seg_num, x), 'selected'] = True
            
            self.peak_selection_changed.emit()
            self.refresh()

    