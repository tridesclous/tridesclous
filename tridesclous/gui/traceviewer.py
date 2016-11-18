import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np

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
        
        # create graphic view and plot item
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
            
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
        self.timeseeker = TimeSeeker()
        tb.addWidget(self.timeseeker)
        self.timeseeker.time_changed.connect(self.seek)
        
        # winsize
        self.xsize = .5
        tb.addWidget(QtGui.QLabel(u'X size (s)'))
        self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 4.], suffix = 's', siPrefix = True, step = 0.1, dec = True)
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
        
        self._create_toolbar()
    

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
        self.scatters = {}
        for c in range(self.dataio.nb_channel):
            color = '#7FFF00'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves.append(curve)
            #~ label = pg.TextItem(str(self.dataio.info['channels'][c]), color=color, anchor=(0, 0.5), border=None, fill=pg.mkColor((128,128,128, 180)))
            #TODO label channels
            label = pg.TextItem('chan{}'.format(c), color=color, anchor=(0, 0.5), border=None, fill=pg.mkColor((128,128,128, 180)))
            self.plot.addItem(label)
            self.channel_labels.append(label)
            
            #~ tc = pg.InfiniteLine(angle = 0., movable = False, pen = pg.mkPen('w'))
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
        

        t_start=0.
        t_stop = self.dataio.get_segment_shape(self.seg_num)[0]/self.dataio.sample_rate
        self.timeseeker.set_start_stop(t_start, t_stop, seek = False)
        
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
    
    def xsize_changed(self):
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
            self.med = np.zeros(self.dataio.nb_channel, dtype='float32')
            self.mad = np.ones(self.dataio.nb_channel, dtype='float32')
        
        self.factor = 1.
        self.gain_zoom(15.)
    
    def gain_zoom(self, factor_ratio):
        self.factor *= factor_ratio
        n = self.dataio.nb_channel
        self.gains = np.ones(n, dtype=float) * 1./(self.factor*max(self.mad))
        self.offsets = np.arange(n)[::-1] - self.med*self.gains
        self.refresh()

    def seek(self, t):
        if self.sender() is not self.timeseeker:
            self.timeseeker.seek(t, emit = False)
        
        self.time_by_seg[self.seg_num] = t
        t1,t2 = t-self.xsize/3. , t+self.xsize*2/3.
        t_start = 0.
        sr = self.dataio.sample_rate
        ind1 = max(0, int((t1-t_start)*sr))
        ind2 = int((t2-t_start)*sr)

        sigs_chunk = self.dataio.get_signals_chunk(seg_num=self.seg_num, i_start=ind1, i_stop=ind2, signal_type=self.signal_type,
                return_type='raw_numpy')
        
        if sigs_chunk is None: 
            return
        
        if self.gains is None:
            self.estimate_auto_scale()

        #signal chunk
        times_chunk = np.arange(sigs_chunk.shape[0], dtype='float32')/self.dataio.sample_rate+max(t1, 0)
        for c in range(self.dataio.nb_channel):
            self.curves[c].setData(times_chunk, sigs_chunk[:, c]*self.gains[c]+self.offsets[c])
            self.channel_labels[c].setPos(t1, self.dataio.nb_channel-c-1)
        
        # plot peaks or spikes or prediction or residuals ...
        self._plot_specific_items(ind1, ind2, sigs_chunk, times_chunk)
        
        #ranges
        self.plot.setXRange( t1, t2, padding = 0.0)
        self.plot.setYRange(-.5, self.dataio.nb_channel-.5, padding = 0.0)



class CatalogueTraceViewer(BaseTraceViewer):
    def __init__(self, controller=None, signal_type = 'processed', parent=None):
        BaseTraceViewer.__init__(self, controller=controller, signal_type=signal_type, parent=parent)
    
    def _create_toolbar(self):
        pass
        
    def _initialize_plot(self):
        pass
    
    def _plot_specific_items(self, ind1, ind2, sigs_chunk, times_chunk):
        # plot peaks

        if self.controller.spike_index ==[]:
            return
        
        keep = (self.controller.spike_segment==self.seg_num) & (self.controller.spike_index>=ind1) & (self.controller.spike_index<ind2)
        
        inwindow_ind = np.array(self.controller.spike_index[keep] - ind1)
        inwindow_label = np.array(self.controller.spike_label[keep])
        inwindow_selected = np.array(self.controller.spike_selection[keep])
        
        for k in list(self.scatters.keys()):
            if not k in self.controller.cluster_labels:
                scatter = self.scatters.pop(k)
                self.plot.removeItem(scatter)
            
        if np.sum(inwindow_selected)==1:
            self.selection_line.setPos(times_chunk[inwindow_ind[inwindow_selected]])
            self.selection_line.show()
        else:
            self.selection_line.hide()
        
        for k in self.controller.cluster_labels:
            color = self.controller.qcolors.get(k, self._default_color)
            if k not in self.scatters:
                self.scatters[k] = pg.ScatterPlotItem(pen=None, brush=color, size=10, pxMode = True)
                self.plot.addItem(self.scatters[k])
                self.scatters[k].sigClicked.connect(self.item_clicked)
            
            if not self.controller.cluster_visible[k]:
                self.scatters[k].setData([], [])
            else:
                mask = inwindow_label==k
                times_chunk_in = times_chunk[inwindow_ind[mask]]
                sigs_chunk_in = sigs_chunk[inwindow_ind[mask], :]
                c = self.controller.centroids[k]['max_on_channel']
                self.scatters[k].setBrush(color)
                self.scatters[k].setData(times_chunk_in, sigs_chunk_in[:, c]*self.gains[c]+self.offsets[c])

        n = self.dataio.nb_channel
        for c in range(n):
            if self.params['plot_threshold']:
                threshold = self.controller.get_threshold()
                self.threshold_lines[c].setPos(n-c-1 + self.gains[c]*self.mad[c]*threshold)
                self.threshold_lines[c].show()
            else:
                self.threshold_lines[c].hide()
    
    def on_spike_selection_changed(self):
        n_selected = np.sum(self.controller.spike_selection)
        if self.params['auto_zoom_on_select'] and n_selected==1:
            ind, = np.nonzero(self.controller.spike_selection)
            ind = ind[0]
            seg_num = self.controller.spike_segment[ind]
            peak_time = self.controller.spike_index[ind]/self.dataio.sample_rate

            if seg_num != self.seg_num:
                seg_pos = seg_num
                self.combo_seg.setCurrentIndex(seg_pos)
            self.spinbox_xsize.setValue(self.params['zoom_size'])
            self.seek(peak_time)
        else:
            self.refresh()
    
    def item_clicked(self, plot, points):
        if self.select_button.isChecked()and len(points)==1:
            x = points[0].pos().x()
            self.controller.spike_selection[:] = False
            
            pos_click = int(x*self.dataio.sample_rate )
            mask = self.controller.spike_segment==self.seg_num
            ind_nearest = np.argmin(np.abs(self.controller.spike_index[mask] - pos_click))
            
            ind_clicked = np.nonzero(mask)[0][ind_nearest]
            #~ ind_clicked
            self.controller.spike_selection[ind_clicked] = True
            #~ print(ind_nearest)
            
            #~ self.controller.spike_selection.loc[(self.seg_num, x)] = True
            
            self.spike_selection_changed.emit()
            self.refresh()
    


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
        self.curves_prediction = []
        self.curves_residuals = []
        for c in range(self.controller.dataio.nb_channel):
            color = '#FF00FF'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves_prediction.append(curve)

            color = '#FFFF00'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves_residuals.append(curve)

   
    def _plot_specific_items(self, ind1, ind2, sigs_chunk, times_chunk):
        
        #~ all_spikes = self.controller.dataio.get_spikes(seg_num=self.seg_num, i_start=None, i_stop=None)
        all_spikes = self.controller.spikes
        
        keep = (all_spikes['segment']==self.seg_num) & (all_spikes['index']>=ind1) & (all_spikes['index']<ind2)
        spikes = np.array(all_spikes[keep], copy=True)

        spikes['index'] -= ind1
        
        inwindow_ind = spikes['index']
        inwindow_label = spikes['label']
        inwindow_selected = spikes['selected']
        
        for k in list(self.scatters.keys()):
            if not k in self.controller.cluster_labels:
                scatter = self.scatters.pop(k)
                self.plot.removeItem(scatter)
        
        if np.sum(inwindow_selected)==1:
            self.selection_line.setPos(times_chunk[inwindow_ind[inwindow_selected]])
            self.selection_line.show()
        else:
            self.selection_line.hide()

        for k in self.controller.cluster_labels:
            if k<0:
                continue
            color = self.controller.qcolors.get(k,  self._default_color)
            
            if k not in self.scatters:
                self.scatters[k] = pg.ScatterPlotItem(pen=None, brush=color, size=10, pxMode = True)
                self.plot.addItem(self.scatters[k])
                self.scatters[k].sigClicked.connect(self.item_clicked)
            
            if not self.controller.cluster_visible[k]:
                self.scatters[k].setData([], [])
            else:
                mask = inwindow_label==k
                times_chunk_in = times_chunk[inwindow_ind[mask]]
                sigs_chunk_in = sigs_chunk[inwindow_ind[mask], :]
                cluster_idx = self.controller.catalogue['label_to_index'][k]
                c = self.controller.catalogue['max_on_channel'][cluster_idx]
                
                self.scatters[k].setBrush(color)
                self.scatters[k].setData(times_chunk_in, sigs_chunk_in[:, c]*self.gains[c]+self.offsets[c])

        n = self.controller.dataio.nb_channel
        for c in range(n):
            if self.params['plot_threshold']:
                threshold = self.controller.get_threshold()
                self.threshold_lines[c].setPos(n-c-1 + self.gains[c]*self.mad[c]*threshold)
                self.threshold_lines[c].show()
            else:
                self.threshold_lines[c].hide()

        #prediction
        prediction = make_prediction_signals(spikes, sigs_chunk.dtype, sigs_chunk.shape, self.controller.catalogue)
        residuals = sigs_chunk - prediction
        
       
        for c in range(self.controller.dataio.nb_channel):
            if self.plot_buttons['prediction'].isChecked():
                self.curves_prediction[c].setData(times_chunk, prediction[:, c]*self.gains[c]+self.offsets[c])
            else:
                self.curves_prediction[c].setData([], [])
            
            if self.plot_buttons['residual'].isChecked():
                self.curves_residuals[c].setData(times_chunk, residuals[:, c]*self.gains[c]+self.offsets[c])
            else:
                self.curves_residuals[c].setData([], [])
                
            if not self.plot_buttons['signals'].isChecked():
                self.curves[c].setData([], [])

    def on_spike_selection_changed(self):
        spikes = self.controller.spikes
        selected = spikes['selected']
        n_selected = np.sum(selected)
        if self.params['auto_zoom_on_select'] and n_selected==1:
            ind, = np.nonzero(selected)
            ind = ind[0]
            seg_num = spikes[ind]['segment']
            peak_time = spikes[ind]['index']/self.controller.dataio.sample_rate

            if seg_num != self.seg_num:
                self.combo_seg.setCurrentIndex(seg_num)
            self.spinbox_xsize.setValue(self.params['zoom_size'])
            self.seek(peak_time)
        else:
            self.refresh()

    
    def item_clicked(self, plot, points):
        #TODO
        pass
        #~ if self.select_button.isChecked()and len(points)==1:
            #~ x = points[0].pos().x()
            #~ self.spikesorter.peak_selection[:] = False
            #~ self.spikesorter.peak_selection.loc[(self.seg_num, x)] = True
            
            #~ self.spike_selection_changed.emit()
            #~ self.refresh()

