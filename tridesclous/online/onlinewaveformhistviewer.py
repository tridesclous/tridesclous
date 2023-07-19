import numpy as np

import matplotlib.cm
import matplotlib.colors

from ..gui import QT
import pyqtgraph as pg
from pyqtgraph.util.mutex import Mutex

import pyacq
from pyacq import WidgetNode,ThreadPollInput, StreamConverter, InputStream


#~ _dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]
from ..peeler import _dtype_spike
from ..tools import make_color_dict
from ..labelcodes import LABEL_UNCLASSIFIED

import time

class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    gain_zoom = QT.pyqtSignal(float)
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() == QT.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()
    def raiseContextMenu(self, ev):
        #for some reasons enableMenu=False is not taken (bug ????)
        pass



    
class OnlineWaveformHistViewer(WidgetNode):
    
    _input_specs = {'signals': dict(streamtype='signals'),
                                'spikes': dict(streamtype='events', shape = (-1, ),  dtype=_dtype_spike),
                                    }
    
    _params = [
                      {'name': 'colormap', 'type': 'list', 'limits' : ['hot', 'viridis', 'jet', 'gray',  ] },
                      {'name': 'bin_min', 'type': 'float', 'value' : -20. },
                      {'name': 'bin_max', 'type': 'float', 'value' : 8. },
                      {'name': 'bin_size', 'type': 'float', 'value' : .1 },
                      {'name': 'refresh_interval', 'type': 'int', 'value': 100, 'limits':[5, 1000]},
                      ]

    
    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)

        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        
        h = QT.QHBoxLayout()
        self.layout.addLayout(h)
        
        self.combobox = QT.QComboBox()
        h.addWidget(self.combobox)
        
        but = QT.QPushButton('clear')
        h.addWidget(but)
        but.clicked.connect(self.on_clear)
        
        self.label = QT.QLabel('')
        h.addWidget(self.label)
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)

        self.params = pg.parametertree.Parameter.create( name='settings', type='group', children=self._params)
        self.tree_params = pg.parametertree.ParameterTree(parent=self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle('Options for waveforms hist viewer')
        self.tree_params.setWindowFlags(QT.Qt.Window)
        self.params.sigTreeStateChanged.connect(self.on_params_changed)
        
        self.initialize_plot()
        
        self.mutex = Mutex()

        
        
    def _configure(self, peak_buffer_size=100000, catalogue=None, **kargs):
        self.peak_buffer_size = peak_buffer_size
        self.catalogue = catalogue
    
    
    def _initialize(self, **kargs):

        self.sample_rate =  self.inputs['signals'].params['sample_rate']
        self.wf_dtype =  self.inputs['signals'].params['dtype']
        
        self.inputs['spikes'].set_buffer(size=self.peak_buffer_size, double=False)
        buffer_sigs_size = int(self.sample_rate*3.)
        self.inputs['signals'].set_buffer(size=buffer_sigs_size, double=False)

        # poller
        self.poller_sigs = ThreadPollInput(input_stream=self.inputs['signals'], return_data=True)
        self.poller_spikes = ThreadPollInput(input_stream=self.inputs['spikes'], return_data=True)

        self.histogram_2d = {}
        self.last_waveform = {}
        self.change_catalogue(self.catalogue)
        
        self.timer = QT.QTimer(interval=100)
        self.timer.timeout.connect(self.refresh)
    
    def _start(self, **kargs):
        self.last_head_sigs = None
        self.last_head_spikes = None
        self.timer.start()
        self.inputs['signals'].empty_queue()
        self.inputs['spikes'].empty_queue()
        self.poller_sigs.start()
        self.poller_spikes.start()

    def _stop(self, **kargs):
        self.timer.stop()
        self.poller_sigs.stop()
        self.poller_sigs.wait()
        self.poller_spikes.stop()
        self.poller_spikes.wait()

    def _close(self, **kargs):
        pass

    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()
    
    def on_params_changed(self, params, changes):
        self.change_lut()
        self.change_catalogue(self.catalogue)
        self.timer.setInterval(self.params['refresh_interval'])

    def initialize_plot(self):
        
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        
        self.curve_spike = pg.PlotCurveItem()
        self.plot.addItem(self.curve_spike)

        self.curve_limit = pg.PlotCurveItem()
        self.plot.addItem(self.curve_limit)
        
        self.change_lut()


    def change_lut(self):
        N = 512
        cmap_name = self.params['colormap']
        cmap = matplotlib.cm.get_cmap(cmap_name , N)
        lut = []
        for i in range(N):
            r,g,b,_ =  matplotlib.colors.ColorConverter().to_rgba(cmap(i))
            lut.append([r*255,g*255,b*255])
        self.lut = np.array(lut, dtype='uint8')

    def change_catalogue(self, catalogue):
        self.params.blockSignals(True)
        with self.mutex:
            
            self.catalogue = catalogue
            
            colors = make_color_dict(self.catalogue['clusters'])
            self.qcolors = {}
            for k, color in colors.items():
                r, g, b = color
                self.qcolors[k] = QT.QColor(int(r*255), int(g*255), int(b*255))

            self.all_plotted_labels = self.catalogue['cluster_labels'].tolist() + [LABEL_UNCLASSIFIED]
            
            centers0 = self.catalogue['centers0']
            if centers0.shape[0]>0:
                self.params['bin_min'] = np.min(centers0)*1.5
                self.params['bin_max'] = np.max(centers0)*1.5
            
            
            bin_min, bin_max = self.params['bin_min'], self.params['bin_max']
            bin_size = self.params['bin_size']
            self.bins = np.arange(bin_min, bin_max, self.params['bin_size'])

            self.combobox.clear()
            self.combobox.addItems([str(k) for k in self.all_plotted_labels])
        
        self.on_clear()
        self._max = 10
        
        _, peak_width, nb_chan = self.catalogue['centers0'].shape
        x, y = [], []
        for c in range(1, nb_chan):
            x.extend([c*peak_width, c*peak_width, np.nan])
            y.extend([-1000, 1000, np.nan])
        
        self.curve_limit.setData(x=x, y=y, connect='finite')
        self.params.blockSignals(False)
        
        

    def on_clear(self):
        with self.mutex:
            shape = self.catalogue['centers0'].shape
            
            self.indexes0 = np.arange(shape[1]*shape[2], dtype='int64')
            
            self.histogram_2d = {}
            self.last_waveform = {}
            self.nb_spikes = {}
            for k in self.all_plotted_labels:
                self.histogram_2d[k] = np.zeros((shape[1]*shape[2], self.bins.size), dtype='int64')
                self.last_waveform[k] = np.zeros((shape[1]*shape[2],), dtype=self.wf_dtype)
                self.nb_spikes[k] = 0
        
        self.plot.setXRange(0, self.indexes0[-1]+1)
        self.plot.setYRange(self.params['bin_min'], self.params['bin_max'])
    
    def auto_scale(self):
        pass
        

    def gain_zoom(self, v):
        self._max *= v
        self.image.setLevels([0, self._max], update=True)
    
    def refresh(self):
        #~ print('refresh')
        #~ t0 = time.perf_counter()
        
        head_sigs = self.poller_sigs.pos()
        head_spikes = self.poller_spikes.pos()
        
        if self.last_head_sigs is None:
            self.last_head_sigs = head_sigs
        
        if self.last_head_spikes is None:
            self.last_head_spikes = head_spikes
        
        if self.last_head_spikes is None or self.last_head_sigs is None:
            return
        
        # update image
        n_right, n_left = self.catalogue['n_right'],self.catalogue['n_left']
        bin_min, bin_max, bin_size = self.params['bin_min'], self.params['bin_max'],self.params['bin_size']
        
        
        # check peak_buffer_size here
        if (head_spikes-self.last_head_spikes)>=(0.9*self.peak_buffer_size):
            self.last_head_spikes = head_spikes - int(0.9*self.peak_buffer_size)
        new_spikes = self.inputs['spikes'].get_data(self.last_head_spikes, head_spikes)
        
        right_indexes = new_spikes['index'] + n_right
        if np.any(right_indexes > head_sigs):
            # the buffer of signals is available for some spikes yet
            # so remove then for this loop and get back on head_spikes
            first_out = np.nonzero(right_indexes)[0][0]
            head_spikes = head_spikes - (new_spikes.size - first_out)
            new_spikes = new_spikes[:first_out]
        
        for k in self.all_plotted_labels:
            mask = new_spikes['cluster_label'] == k
            indexes = new_spikes[mask]['index']
            for ind in indexes:
                wf = self.inputs['signals'].get_data(ind+n_left, ind+n_right)
                wf = wf.T.reshape(-1)
                wf_bined = np.floor((wf-bin_min)/bin_size).astype('int32')
                wf_bined = wf_bined.clip(0, self.bins.size-1)
                
                with self.mutex:
                    self.histogram_2d[k][self.indexes0, wf_bined] += 1
                    self.last_waveform[k] = wf
                    self.nb_spikes[k] += 1
        
        self.last_head_sigs = head_sigs
        self.last_head_spikes = head_spikes

        
        if self.combobox.currentIndex() == -1:
            return

        if self.visibleRegion().isEmpty():
            # when several tabs not need to refresh
            return
        
        # refresh plot , update image
        k = self.all_plotted_labels[self.combobox.currentIndex()]
        hist2d = self.histogram_2d[k]

        self.image.setImage(hist2d, lut=self.lut, levels=[0, self._max])
        self.image.setRect(QT.QRectF(-0.5, bin_min, hist2d.shape[0], bin_max-bin_min))
        self.image.show()
        
        self.curve_spike.setData(x=self.indexes0, y=self.last_waveform[k],
                                            pen=pg.mkPen(self.qcolors[k], width=1.5))
        
        
        txt = 'nbs_pike = {}'.format(self.nb_spikes[k])
        self.label.setText(txt)
        
        #~ t1 = time.perf_counter()
        #~ print('refresh time', t1-t0)
        
        
    
