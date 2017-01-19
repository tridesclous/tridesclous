import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import time

from .base import WidgetBase
from .tools import TimeSeeker
from ..tools import median_mad
from ..dataio import _signal_types
from ..peeler import make_prediction_signals


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


class BaseTraceViewer(WidgetBase):
    def __init__(self,controller=None, signal_type='initial', parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
    
        self.dataio = controller.dataio
        self.signal_type = signal_type
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.create_toolbar()
        self.layout.addWidget(self.toolbar)
        
        # create graphic view and 2 scroll bar
        g = QtGui.QGridLayout()
        self.layout.addLayout(g)
        self.scroll_chan = QtGui.QScrollBar()
        g.addWidget(self.scroll_chan, 0,0)
        self.scroll_chan.valueChanged.connect(self.on_scroll_chan)
        self.graphicsview = pg.GraphicsView()
        g.addWidget(self.graphicsview, 0,1)
        self.initialize_plot()
        self.layout.addLayout(g)
        self.scroll_time = QtGui.QScrollBar(orientation=QtCore.Qt.Horizontal)
        g.addWidget(self.scroll_time, 1,1)
        self.scroll_time.valueChanged.connect(self.on_scroll_time)

        #handle time by segments
        self.time_by_seg = np.array([0.]*self.dataio.nb_segment, dtype='float64')
        
        _params = [{'name': 'auto_zoom_on_select', 'type': 'bool', 'value': True },
                           {'name': 'zoom_size', 'type': 'float', 'value':  0.08, 'step' : 0.001 },
                          {'name': 'plot_threshold', 'type': 'bool', 'value':  True },
                          ]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        self.params.param('plot_threshold').sigValueChanged.connect(self.refresh)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for signal viewer')
        self.tree_params.setWindowFlags(QtCore.Qt.Window)
        
        self.change_segment(0)
        self.refresh()
    
    _default_color = QtGui.QColor( 'white')
    
    def create_toolbar(self):
        tb = self.toolbar = QtGui.QToolBar()
        
        #Segment selection
        self.combo_seg = QtGui.QComboBox()
        tb.addWidget(self.combo_seg)
        self.combo_seg.addItems([ 'Segment {}'.format(seg_num) for seg_num in range(self.dataio.nb_segment) ])
        self._seg_pos = 0
        self.seg_num = self._seg_pos
        self.combo_seg.currentIndexChanged.connect(self.on_combo_seg_changed)
        tb.addSeparator()
        
        self.combo_type = QtGui.QComboBox()
        tb.addWidget(self.combo_type)
        self.combo_type.addItems([ signal_type for signal_type in _signal_types ])
        self.combo_type.setCurrentIndex(_signal_types.index(self.signal_type))
        self.combo_type.currentIndexChanged.connect(self.on_combo_type_changed)

        # time slider
        self.timeseeker = TimeSeeker(show_slider=False)
        tb.addWidget(self.timeseeker)
        self.timeseeker.time_changed.connect(self.seek)
        
        # winsize
        self.xsize = .5
        tb.addWidget(QtGui.QLabel(u'X size (s)'))
        self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 4.], suffix = 's', siPrefix = True, step = 0.1, dec = True)
        self.spinbox_xsize.sigValueChanged.connect(self.on_xsize_changed)
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
        
        self._create_toolbar()
    

    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        self.plot.showAxis('left', False)
        
        self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.viewBox.xsize_zoom.connect(self.xsize_zoom)
        
        self.visible_channels = np.zeros(self.controller.nb_channel, dtype='bool')
        self.max_channel = min(16, self.controller.nb_channel)
        #~ self.max_channel = min(5, self.controller.nb_channel)
        if self.controller.nb_channel>self.max_channel:
            self.visible_channels[:self.max_channel] = True
            self.scroll_chan.show()
            self.scroll_chan.setMinimum(0)
            self.scroll_chan.setMaximum(self.controller.nb_channel-self.max_channel)
            self.scroll_chan.setPageStep(self.max_channel)
        else:
            self.visible_channels[:] = True
            self.scroll_chan.hide()
            
        self.signals_curve = pg.PlotCurveItem(pen='#7FFF00', connect='finite')
        self.plot.addItem(self.signals_curve)

        self.scatter = pg.ScatterPlotItem(size=10, pxMode = True)
        self.plot.addItem(self.scatter)
        self.scatter.sigClicked.connect(self.scatter_item_clicked)
        
        self.channel_labels = []
        self.threshold_lines =[]
        for c in range(self.controller.nb_channel):
            #TODO label channels
            label = pg.TextItem('chan{}'.format(c), color='#FFFFFF', anchor=(0, 0.5), border=None, fill=pg.mkColor((128,128,128, 180)))
            self.plot.addItem(label)
            self.channel_labels.append(label)
        
        
        for i in range(self.max_channel):
            tc = pg.InfiniteLine(angle = 0., movable = False, pen = pg.mkPen(color=(128,128,128, 120)))
            tc.setPos(0.)
            self.threshold_lines.append(tc)
            self.plot.addItem(tc)
            tc.hide()
        
        pen = pg.mkPen(color=(128,0,128, 120), width=3, style=QtCore.Qt.DashLine)
        self.selection_line = pg.InfiniteLine(pos = 0., angle=90, movable=False, pen = pen)
        self.plot.addItem(self.selection_line)
        self.selection_line.hide()
        
        self._initialize_plot()
        
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
        #TODO: dirty because now seg_pos IS seg_num
        self._seg_pos  =  seg_pos
        if self._seg_pos<0:
            self._seg_pos = self.dataio.nb_segment-1
        if self._seg_pos == self.dataio.nb_segment:
            self._seg_pos = 0
        self.seg_num = self._seg_pos
        self.combo_seg.setCurrentIndex(self._seg_pos)
        
        length = self.dataio.get_segment_length(self.seg_num)
        t_start=0.
        t_stop = length/self.dataio.sample_rate
        self.timeseeker.set_start_stop(t_start, t_stop, seek = False)

        self.scroll_time.setMinimum(0)
        self.scroll_time.setMaximum(length)
        
        if self.isVisible():
            self.refresh()
    
    def on_combo_seg_changed(self):
        s =  self.combo_seg.currentIndex()
        self.change_segment(s)
    
    def on_combo_type_changed(self):
        s =  self.combo_type.currentIndex()
        self.signal_type = _signal_types[s]
        self.estimate_auto_scale()
        self.change_segment(self._seg_pos)
    

    
    def on_xsize_changed(self):
        self.xsize = self.spinbox_xsize.value()
        if self.isVisible():
            self.refresh()
    
    def refresh(self):
        self.seek(self.time_by_seg[self.seg_num])

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

        if self.signal_type=='initial':
            i_stop = min(int(60.*self.dataio.sample_rate), self.dataio.get_segment_shape(self.seg_num)[0])
            sigs = self.dataio.get_signals_chunk(seg_num=self.seg_num, i_start=0, i_stop=i_stop, signal_type=self.signal_type,
                return_type='raw_numpy')
            self.med, self.mad = median_mad(sigs.astype('float32'), axis = 0)

        elif self.signal_type=='processed':
            #in that case it should be already normalize
            self.med = np.zeros(self.controller.nb_channel, dtype='float32')
            self.mad = np.ones(self.controller.nb_channel, dtype='float32')
        
        self.factor = 1.
        self.gain_zoom(15.)
    
    def gain_zoom(self, factor_ratio):
        self.factor *= factor_ratio
        self.gains = np.zeros(self.controller.nb_channel, dtype='float32')
        self.offsets = np.zeros(self.controller.nb_channel, dtype='float32')
        n = np.sum(self.visible_channels)
        self.gains[self.visible_channels] = np.ones(n, dtype=float) * 1./(self.factor*max(self.mad))
        self.offsets[self.visible_channels] = np.arange(n)[::-1] - self.med[self.visible_channels]*self.gains[self.visible_channels]
        self.refresh()

    def on_scroll_time(self, val):
        sr = self.controller.dataio.sample_rate
        self.timeseeker.seek(val/sr)
    
    def on_scroll_chan(self, val):
        self.visible_channels[:] = False
        self.visible_channels[val:val+self.max_channel] = True
        self.gain_zoom(1)
        self.refresh()
    
    def center_scrollbar_on_channel(self, c):
        c = c - self.max_channel//2
        c = min(max(c, 0), self.controller.nb_channel-self.max_channel)
        self.scroll_chan.valueChanged.disconnect(self.on_scroll_chan)
        self.scroll_chan.setValue(c)
        self.scroll_chan.valueChanged.connect(self.on_scroll_chan)
        
        self.visible_channels[:] = False
        self.visible_channels[c:c+self.max_channel] = True
        self.gain_zoom(1)
    
    def scatter_item_clicked(self, plot, points):
        if self.select_button.isChecked()and len(points)==1:
            x = points[0].pos().x()
            self.controller.spike_selection[:] = False
            
            pos_click = int(x*self.dataio.sample_rate )
            mask = self.controller.spikes['segment']==self.seg_num
            ind_nearest = np.argmin(np.abs(self.controller.spikes[mask]['index'] - pos_click))
            
            ind_clicked = np.nonzero(mask)[0][ind_nearest]
            self.controller.spike_selection[ind_clicked] = True
            
            self.spike_selection_changed.emit()
            self.refresh()
    
    def on_spike_selection_changed(self):
        ind_selected, = np.nonzero(self.controller.spike_selection)
        n_selected = ind_selected.size
        if self.params['auto_zoom_on_select'] and n_selected==1:
            ind_selected, = np.nonzero(self.controller.spike_selection)
            ind = ind_selected[0]
            peak_ind = self.controller.spikes[ind]['index']
            seg_num = self.controller.spikes[ind]['segment']
            peak_time = peak_ind/self.dataio.sample_rate
            
            if seg_num != self.seg_num:
                self.combo_seg.setCurrentIndex(seg_num)
            
            self.spinbox_xsize.sigValueChanged.disconnect(self.on_xsize_changed)
            self.spinbox_xsize.setValue(self.params['zoom_size'])
            self.xsize = self.params['zoom_size']
            self.spinbox_xsize.sigValueChanged.connect(self.on_xsize_changed)
            
            label = self.controller.spikes[ind]['label']
            if label>=0:
                c = self.controller.get_max_on_channel(label)
            else:
                wf = self.controller.dataio.get_signals_chunk(seg_num=seg_num, chan_grp=self.controller.chan_grp,
                        i_start=peak_ind, i_stop=peak_ind+1,
                        signal_type='processed', return_type='raw_numpy')
                c = np.argmax(np.abs(wf))
            
            self.center_scrollbar_on_channel(c)
            
            self.seek(peak_time)
            
        else:
            self.refresh()
    
    def seek(self, t):
        #~ tp1 = time.perf_counter()
        
        if self.sender() is not self.timeseeker:
            self.timeseeker.seek(t, emit = False)
        
        self.time_by_seg[self.seg_num] = t
        t1,t2 = t-self.xsize/3. , t+self.xsize*2/3.
        t_start = 0.
        sr = self.dataio.sample_rate

        self.scroll_time.valueChanged.disconnect(self.on_scroll_time)
        self.scroll_time.setValue(int(sr*t))
        self.scroll_time.setPageStep(int(sr*self.xsize))
        self.scroll_time.valueChanged.connect(self.on_scroll_time)
        
        ind1 = max(0, int((t1-t_start)*sr))
        ind2 = int((t2-t_start)*sr)

        sigs_chunk = self.dataio.get_signals_chunk(seg_num=self.seg_num, chan_grp=self.controller.chan_grp,
                i_start=ind1, i_stop=ind2, signal_type=self.signal_type,
                return_type='raw_numpy')
        
        if sigs_chunk is None: 
            return
        
        if self.gains is None:
            self.estimate_auto_scale()

        
        nb_visible = np.sum(self.visible_channels)
        
        data_curves = sigs_chunk[:, self.visible_channels].T.copy()
        if data_curves.dtype!='float32':
            data_curves = data_curves.astype('float32')
        
        data_curves *= self.gains[self.visible_channels, None]
        data_curves += self.offsets[self.visible_channels, None]
        data_curves[:,0] = np.nan
        data_curves = data_curves.flatten()
        times_chunk = np.arange(sigs_chunk.shape[0], dtype='float32')/self.dataio.sample_rate+max(t1, 0)
        times_chunk_tile = np.tile(times_chunk, nb_visible)
        self.signals_curve.setData(times_chunk_tile, data_curves)
        
        
        #channel labels
        i = 1
        for c in range(self.controller.nb_channel):
            if self.visible_channels[c]:
                self.channel_labels[c].setPos(t1, nb_visible-i)
                self.channel_labels[c].show()
                i +=1
            else:
                self.channel_labels[c].hide()

        n = np.sum(self.visible_channels)
        index_visible, = np.nonzero(self.visible_channels)
        for i, c in enumerate(index_visible):
            if self.params['plot_threshold']:
                threshold = self.controller.get_threshold()
                self.threshold_lines[i].setPos(n-i-1 + self.gains[c]*threshold)
                self.threshold_lines[i].show()
            else:
                self.threshold_lines[i].hide()        
        
        
        # plot peak on signal
        all_spikes = self.controller.spikes
        if len(all_spikes)>0:
            keep = (all_spikes['segment']==self.seg_num) & (all_spikes['index']>=ind1) & (all_spikes['index']<ind2)
            spikes_chunk = np.array(all_spikes[keep], copy=True)
            spikes_chunk['index'] -= ind1
            inwindow_ind = spikes_chunk['index']
            inwindow_label = spikes_chunk['label']
            inwindow_selected = np.array(self.controller.spike_selection[keep])

            self.scatter.clear()
            all_x = []
            all_y = []
            all_brush = []
            for k in self.controller.cluster_labels:

                if not self.controller.cluster_visible[k]: continue
                mask = inwindow_label==k
                if np.sum(mask)==0: continue
                
                color = self.controller.qcolors.get(k, self._default_color)
                
                x = times_chunk[inwindow_ind[mask]]
                
                sigs_chunk_in = sigs_chunk[inwindow_ind[mask], :]
                if k >=0:
                    c = self.controller.get_max_on_channel(k)
                    if c is not None:
                        #~ print('k', k, 'c', c)
                        c = np.array([c]*np.sum(mask), dtype='int64')
                    else:
                        c = np.argmax(np.abs(sigs_chunk_in), axis=1)
                else:
                    c = np.argmax(np.abs(sigs_chunk_in), axis=1)
                keep = self.visible_channels[c]
                if np.sum(keep)==0: continue
                c = c[keep]
                x = x[keep]
                y = sigs_chunk_in[np.arange(c.size), c]*self.gains[c]+self.offsets[c]
                #~ print('c', c)
                #~ print('x', x)
                #~ print('y', y)
                #~ self.scatter.addPoints(x=x, y=y,pen=pg.mkPen(None), brush=color)
                all_x.append(x)
                all_y.append(y)
                all_brush.append(np.array([pg.mkBrush(color)]*len(x)))
            if len(all_x):
                all_x = np.concatenate(all_x)
                all_y = np.concatenate(all_y)
                all_brush = np.concatenate(all_brush)
                self.scatter.setData(x=all_x, y=all_y, brush=all_brush)
            
            
            if np.sum(inwindow_selected)==1:
                self.selection_line.setPos(times_chunk[inwindow_ind[inwindow_selected]])
                self.selection_line.show()
            else:
                self.selection_line.hide()            
        else:
            spikes_chunk = None
        
        # plot prediction or residuals ...
        self._plot_specific_items(sigs_chunk, times_chunk, spikes_chunk)
        
        #ranges
        self.plot.setXRange( t1, t2, padding = 0.0)
        self.plot.setYRange(-.5, nb_visible-.5, padding = 0.0)
        
        #TODO : do some thing here
        #~ self.graphicsview.repaint()

        #~ tp2 = time.perf_counter()
        #~ print('seek', tp2-tp1)
        

class CatalogueTraceViewer(BaseTraceViewer):
    def __init__(self, controller=None, signal_type = 'processed', parent=None):
        BaseTraceViewer.__init__(self, controller=controller, signal_type=signal_type, parent=parent)
    
    def _create_toolbar(self):
        pass
        
    def _initialize_plot(self):
        pass
    
    def _plot_specific_items(self, sigs_chunk, times_chunk, spikes_chunk):
        pass


class PeelerTraceViewer(BaseTraceViewer):
    def __init__(self, controller=None, signal_type = 'processed', parent=None):
        BaseTraceViewer.__init__(self, controller=controller, signal_type=signal_type, parent=parent)
    
    def _create_toolbar(self):
        self.plot_buttons = {}
        for name in ['signals', 'prediction', 'residual']:
            self.plot_buttons[name] = but = QtGui.QPushButton(name,  checkable = True)
            but.clicked.connect(self.refresh)
            self.toolbar.addWidget(but)
            
            if name in ['signals', 'prediction']:
                but.setChecked(True)
    
    def _initialize_plot(self):
        self.curve_predictions = pg.PlotCurveItem(pen='#FF00FF', connect='finite')
        self.plot.addItem(self.curve_predictions)
        self.curve_residuals = pg.PlotCurveItem(pen='#FFFF00', connect='finite')
        self.plot.addItem(self.curve_residuals)
   
    def _plot_specific_items(self, sigs_chunk, times_chunk, spikes_chunk):
        if spikes_chunk is None: return
        
        #prediction
        #TODO make prediction only on visible!!!! 
        prediction = make_prediction_signals(spikes_chunk, sigs_chunk.dtype, sigs_chunk.shape, self.controller.catalogue)
        residuals = sigs_chunk - prediction
        
        # plots
        nb_visible = np.sum(self.visible_channels)
        times_chunk_tile = np.tile(times_chunk, nb_visible)
        
        def plot_curves(curve, data):
            data = data[:, self.visible_channels].T.copy()
            data *= self.gains[self.visible_channels, None]
            data += self.offsets[self.visible_channels, None]
            data[:,0] = np.nan
            data = data.flatten()
            curve.setData(times_chunk_tile, data)
        
        if self.plot_buttons['prediction'].isChecked():
            plot_curves(self.curve_predictions, prediction)
        else:
            self.curve_predictions.setData([], [])

        if self.plot_buttons['residual'].isChecked():
            plot_curves(self.curve_residuals, residuals)
        else:
            self.curve_residuals.setData([], [])
        
        if not self.plot_buttons['signals'].isChecked():
            self.signals_curve.setData([], [])

