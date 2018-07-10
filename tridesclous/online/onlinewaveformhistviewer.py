import numpy as np

from ..gui import QT
import pyqtgraph as pg
from pyqtgraph.util.mutex import Mutex

import pyacq
from pyacq import WidgetNode,ThreadPollInput, StreamConverter, InputStream


#~ _dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]
from ..peeler import _dtype_spike
from ..tools import make_color_dict
from ..labelcodes import LABEL_UNCLASSIFIED

class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    #~ gain_zoom = QT.pyqtSignal(float)
    #~ xsize_zoom = QT.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.disableAutoRange()
    #~ def mouseClickEvent(self, ev):
        #~ ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    #~ def mouseDragEvent(self, ev):
        #~ ev.ignore()
    #~ def wheelEvent(self, ev, axis=None):
        #~ if ev.modifiers() == QtCore.Qt.ControlModifier:
            #~ z = 10 if ev.delta()>0 else 1/10.
        #~ else:
            #~ z = 1.3 if ev.delta()>0 else 1/1.3
        #~ self.gain_zoom.emit(z)
        #~ ev.accept()
    #~ def mouseDragEvent(self, ev):
        #~ ev.accept()
        #~ self.xsize_zoom.emit((ev.pos()-ev.lastPos()).x())


    
class OnlineWaveformHistViewer(WidgetNode):
    
    _input_specs = {'signals': dict(streamtype='signals'),
                                'spikes': dict(streamtype='events', shape = (-1, ),  dtype=_dtype_spike),
                                    }
    
    _params = [
                      {'name': 'colormap', 'type': 'list', 'values' : ['hot', 'viridis', 'jet', 'gray',  ] },
                      {'name': 'data', 'type': 'list', 'values' : ['waveforms', 'features', ] },
                      {'name': 'bin_min', 'type': 'float', 'value' : -20. },
                      {'name': 'bin_max', 'type': 'float', 'value' : 8. },
                      {'name': 'bin_size', 'type': 'float', 'value' : .1 },
                      {'name': 'display_threshold', 'type': 'bool', 'value' : True },
                      {'name': 'max_label', 'type': 'int', 'value' : 2 },
                      ]

    
    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)

        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)

        self.params = pg.parametertree.Parameter.create( name='settings', type='group', children=self._params)
        self.tree_params = pg.parametertree.ParameterTree(parent=self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle('Options for waveforms hist viewer')
        self.tree_params.setWindowFlags(QT.Qt.Window)
        #~ self.params.sigTreeStateChanged.connect(self.on_params_changed)
        
        self.initialize_plot()
        
        self.mutex = Mutex()

        
        
    def _configure(self, peak_buffer_size = 100000, catalogue=None, **kargs):
        self.peak_buffer_size = peak_buffer_size
        self.catalogue = catalogue
    
    
    def _initialize(self, **kargs):
        
        self.inputs['spikes'].set_buffer(size=self.peak_buffer_size, double=False)
        self.sample_rate =  self.inputs['signals'].params['sample_rate']
        self.wf_dtype =  self.inputs['signals'].params['dtype']
        print(self.sample_rate)


        # poller onpeak
        self._last_peak = 0
        self.poller_peak = ThreadPollInput(input_stream=self.inputs['spikes'], return_data=True)
        #~ self.poller_peak.new_data.connect(self._on_new_peak)

        self.histogram_2d = {}
        self.last_waveform = {}
        self.change_catalogue(self.catalogue)
        
    
    def _start(self, **kargs):
        self.poller_peak.start()

    def _stop(self, **kargs):
        self.poller_peak.stop()
        self.poller_peak.wait()

    def _close(self, **kargs):
        pass




    def initialize_plot(self):
        
        self.viewBox = MyViewBox()
        #~ self.viewBox.doubleclicked.connect(self.open_settings)
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        
        self.curve_spike = pg.PlotCurveItem()
        self.plot.addItem(self.curve_spike)

    def change_catalogue(self, catalogue):
        with self.mutex:
            
            self.catalogue = catalogue

            colors = make_color_dict(self.catalogue['clusters'])
            self.qcolors = {}
            for k, color in colors.items():
                r, g, b = color
                self.qcolors[k] = QT.QColor(r*255, g*255, b*255)

            self.all_plotted_labels = self.catalogue['cluster_labels'].tolist()# + [LABEL_UNCLASSIFIED]
            
            centers0 = self.catalogue['centers0']
            if centers0.shape[0]>0:
                self.params['bin_min'] = np.min(centers0)*1.5
                self.params['bin_max'] = np.max(centers0)*1.5
            
            
            bin_min, bin_max = self.params['bin_min'], self.params['bin_max']
            bin_size = self.params['bin_size']
            bins = np.arange(bin_min, bin_max, self.params['bin_size'])

            shape = centers0.shape
            
            self.indexes0 = np.arange(shape[1]*shape[2], dtype='int64')
            
            self.histogram_2d = {}
            self.last_waveform = {}
            
            for k in self.all_plotted_labels:
                self.histogram_2d[k] = np.zeros((shape[1]*shape[2], bins.size), dtype='int64')
                self.last_waveform[k] = np.zeros((shape[1]*shape[2],), dtype=self.wf_dtype)

