import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from ..spikesorter import SpikeSorter
from .tools import TimeSeeker




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
    def __init__(self, spikesorter = None, shared_view_with = [], 
                    mode = 'memory', parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.spikesorter = spikesorter
        assert isinstance(self.spikesorter, SpikeSorter)
        self.dataio = self.spikesorter.dataio
        self.mode = mode
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        # Can share view with other trace viewer : simultanetly view filter + full band
        self.shared_view_with = shared_view_with

        self.createToolBar()
        self.layout.addWidget(self.toolbar)
        
        # create graphic view and plot item
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
            
        
        #handle time by segments
        self.time_by_seg = pd.Series(self.dataio.segments['t_start'].copy(), name = 'time', index = self.dataio.segments.index)
        
        self.change_segment(0)
    
    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        self.curves = []
        self.channel_labels = []
        self.threshold_lines =[]
        for c in range(self.dataio.nb_channel):
            color = '#7FFF00'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves.append(curve)
            label = pg.TextItem(str(self.dataio.info['channels'][c]), color=color, anchor=(0.5, 0.5), border=None, fill=pg.mkColor((128,128,128, 200)))
            self.plot.addItem(label)
            self.channel_labels.append(label)
            
            tc = pg.InfiniteLine(angle = 0, movable = False, pen = 'g')
            tc.setPos(0.)
            self.threshold_lines.append(tc)
            self.plot.addItem(tc)
            tc.hide()

    def createToolBar(self):
        tb = self.toolbar = QtGui.QToolBar()
        
        #Segment selection
        but = QtGui.QPushButton('<')
        but.clicked.connect(self.prev_segment)
        tb.addWidget(but)
        self.combo = QtGui.QComboBox()
        tb.addWidget(self.combo)
        self.combo.addItems([ 'Segment {}'.format(seg_num) for seg_num in self.dataio.segments.index ])
        but = QtGui.QPushButton('>')
        but.clicked.connect(self.next_segment)
        tb.addWidget(but)
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
        #~ self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 10.], suffix = 's', siPrefix = True, step = 0.1, dec = True)
        self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 10.]) # step = 0.1, dec = True)
        self.spinbox_xsize.sigValueChanged.connect(self.xsize_changed)
        tb.addWidget(self.spinbox_xsize)
        tb.addSeparator()
        self.spinbox_xsize.sigValueChanged.connect(self.refresh)


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

        if self.isVisible():
            self.refresh()

    
    def on_combo_changed(self):
        s =  self.combo.currentIndex()
        for otherviewer in self.shared_view_with:
            otherviewer.combo.setCurrentIndex(s)
    
    def xsize_changed(self):
        self.xsize = self.spinbox_xsize.value()
        for otherviewer in self.shared_view_with:
            otherviewer.spinbox_xsize.setValue(self.xsize)
        if self.isVisible():
            self.refresh()
            
    
    def seek(self, t, cascade = True):
        if cascade:
            for otherviewer in self.shared_view_with:
                otherviewer.seek(t, cascade = False)
        else:
            self.timeseeker.seek(t, emit = False)
            
        self.time_by_seg[self.seg_num] = t
        t1,t2 = t-self.xsize/3. , t+self.xsize*2/3.
        #~ print(t1, t2)
        
        #signal chunk
        if self.mode == 'memory':
            chunk = self.sigs.loc[t1:t2]
        elif self.mode == 'file':
            chunk = self.dataio.get_signals(seg_num = self.seg_num, t_start = t1, t_stop = t2, filtered = True)
        
        for c in range(self.dataio.nb_channel):
            self.curves[c].setData(chunk.index.values, chunk.iloc[:, c].values)

        #TODO spikes by clusters
        
        self.plot.setXRange( t1, t2, padding = 0.0)
        #~ self.plot.setYRange( *self.ylims , padding = 0.0)
        
        
        
        
        
        
        
        
    
    def refresh(self):
        self.seek(self.time_by_seg[self.seg_num], cascade = False)


    