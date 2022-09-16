from pprint import pprint
import time
import weakref

from matplotlib.cm import get_cmap
import numpy as np

from pyacq.core import WidgetNode, ThreadPollInput, OutputStream
from pyacq.viewers import QOscilloscope

import pyqtgraph as pg
#~ from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.util.mutex import Mutex

from tridesclous.gui import QT
from tridesclous.tools import open_prb, median_mad
from tridesclous.signalpreprocessor import signalpreprocessor_engines
from tridesclous.peakdetector import get_peak_detector_class



noise_estimation_duration = 2.

class ProbeActivityThread(ThreadPollInput):
    def __init__(self, input_stream, viewer, in_group_channels, geometry, 
                        oscope_stream,
                        signalpreprocessor_params,
                        peakdetector_params,
                        timeout=200, parent=None):
        
        ThreadPollInput.__init__(self, input_stream,  timeout=timeout, return_data=True, parent = parent)
        
        self.viewer = viewer
        
        self.in_group_channels = in_group_channels
        self.geometry = geometry
        
        self.oscope_stream = oscope_stream
        self.signalpreprocessor_params = signalpreprocessor_params
        self.peakdetector_params = peakdetector_params
        
        
        self.chunksize = self.viewer.chunksize
        
        self.sample_rate = input_stream.params['sample_rate']
        self.nb_channel = len(self.in_group_channels)
        
        self.noise_estimated = False
        self.is_noise_estimating = True
        self.nb_sample_for_noise = 0
        self.chunk_for_noise = []
        
        self.mutex_noise = Mutex()
    
    def start_new_noise_estimation(self):
        p = dict(self.signalpreprocessor_params)
        engine = p.pop('engine', 'numpy')
        SignalPreprocessor_class = signalpreprocessor_engines[engine]
        source_dtype = 'float32'
        self.signalpreprocessor_for_noise = SignalPreprocessor_class(self.sample_rate, self.nb_channel, self.chunksize, source_dtype)
        p['normalize'] = False  # not computed yet
        p['signals_medians'] = None
        p['signals_mads'] = None
        self.signalpreprocessor_for_noise.change_params(**p)

        with self.mutex_noise:
            self.is_noise_estimating = True
            self.nb_sample_for_noise = 0
            self.chunk_for_noise = []
    
    def on_noise_estimated(self):
        with self.mutex_noise:
            noise = np.concatenate(self.chunk_for_noise)
            signals_medians, signals_mads = median_mad(noise, axis=0)

            p = dict(self.signalpreprocessor_params)
            engine = p.pop('engine', 'numpy')
            SignalPreprocessor_class = signalpreprocessor_engines[engine]
            
            source_dtype = 'float32'
            self.signalpreprocessor = SignalPreprocessor_class(self.sample_rate, self.nb_channel, self.chunksize, source_dtype)
            p['normalize'] = True
            p['signals_medians'] = signals_medians
            p['signals_mads'] = signals_mads
            self.signalpreprocessor.change_params(**p)
            print('estimate noise done')
            print(signals_medians)
            print(signals_mads)
            
            p = dict(self.peakdetector_params)
            engine = p.pop('engine')
            method = p.pop('method')
            PeakDetector_class = get_peak_detector_class(method, engine)
            self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                            self.chunksize, source_dtype, self.geometry)
            self.peakdetector.change_params(**p)

            self.noise_estimated = True
            self.is_noise_estimating = False
            self.signalpreprocessor_for_noise = None
            
            self.timer_autoscale = QT.QTimer(singleShot=True, interval=500)
            self.timer_autoscale.timeout.connect(self.viewer.on_roi_change)
            self.timer_autoscale.start()        
    
    def process_data(self, pos, data):
        self._head = pos
        
        data_in_group = data[:, self.in_group_channels]
        
        #~ print()
        #~ print(data_in_group.shape)
        #~ print(data_in_group.shape[0]/self.sample_rate)
        
        #~ return
        # preprocessor
        if self.noise_estimated:
            #~ t0 = time.perf_counter()
            out_pos, preprocessed_data = self.signalpreprocessor.process_data(pos, data_in_group)
            #~ t1 = time.perf_counter()
            #~ print('signalpreprocessor', t1-t0)
        
        with self.mutex_noise:
            if self.is_noise_estimating:
                out_pos, preprocessed_data = self.signalpreprocessor_for_noise.process_data(pos, data_in_group)
                if preprocessed_data is not None:
                    self.chunk_for_noise.append(preprocessed_data)
                    self.nb_sample_for_noise += preprocessed_data.shape[0]
        
        if self.is_noise_estimating and self.nb_sample_for_noise > int(self.sample_rate * noise_estimation_duration):
            # TODO do this in background with a thread
            self.on_noise_estimated()
    
    
        if  self.noise_estimated:
            # send to oscope
            self.oscope_stream.send(preprocessed_data, index=out_pos)
            
            # peak detector
            #~ t0 = time.perf_counter()
            time_ind_peaks, chan_ind_peaks, peak_val_peaks = self.peakdetector.process_data(out_pos, preprocessed_data)
            #~ t1 = time.perf_counter()
            #~ print('peakdetector', t1-t0)
            
            if time_ind_peaks is None:
                return
            
            # update count
            with self.viewer.mutex:
                self.viewer.count_per_channel[chan_ind_peaks, self.viewer.bin_index] += 1




class ProbeActivityViewerController(QT.QWidget):
    def __init__(self, parent=None, viewer=None):
        QT.QWidget.__init__(self, parent)
        
        self._viewer = weakref.ref(viewer)
        
        # layout
        self.mainlayout = QT.QVBoxLayout()
        self.setLayout(self.mainlayout)
        t = 'Options for {}'.format(viewer.name)
        self.setWindowTitle(t)
        self.mainlayout.addWidget(QT.QLabel('<b>'+t+'<\b>'))
        
        h = QT.QHBoxLayout()
        self.mainlayout.addLayout(h)

        self.tree_params = pg.parametertree.ParameterTree()
        self.tree_params.setParameters(viewer.params, showTop=True)
        self.tree_params.header().hide()
        h.addWidget(self.tree_params)
    
    #~ @property
    #~ def viewer(self):
        #~ return self._viewer()


class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #~ self.disableAutoRange()
        
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()


class ProbeActivityViewer(WidgetNode):
    """
    
    """
    _input_specs = {'signals': dict()}
    
    _default_params = [
            {'name': 'max_rate', 'type': 'float', 'value': 10., 'step': 0.5},
            {'name': 'colormap', 'type': 'list', 'value': 'inferno', 'limits': ['inferno', 'summer', 'viridis', 'jet'] },
            {'name': 'show_channel_num', 'type': 'bool', 'value': True},
            {'name': 'spacing_factor', 'type': 'float', 'value': 20., 'step': 1.0},
    ]
    
    
    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)
        
        self.layout = QT.QHBoxLayout()
        self.setLayout(self.layout)
        
        v = QT.QVBoxLayout()
        self.layout.addLayout(v, 1)
        
        but = QT.QPushButton('Estimate noise')
        v.addWidget(but)
        but.clicked.connect(self.start_new_noise_estimation)
        
        self.graphicsview = pg.GraphicsView()
        #~ self.layout.addWidget(self.graphicsview, 1)
        v.addWidget(self.graphicsview)
        
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.show_params_controller)
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.plot.getViewBox().disableAutoRange()
        self.graphicsview.setCentralItem(self.plot)
        self.plot.getViewBox().setAspectLocked(lock=True, ratio=1)
        self.plot.hideButtons()
        self.plot.showAxis('left', False)
        self.plot.showAxis('bottom', False)
        
        
        self.params = pg.parametertree.Parameter.create(name='Global options',
                                                    type='group', children=self._default_params)
        self.params.sigTreeStateChanged.connect(self.on_param_change)
        self.params_controller = ProbeActivityViewerController(parent=self, viewer=self)
        self.params_controller.setWindowFlags(QT.Qt.Window)
       
        
        
        self.mutex = Mutex()
        
        # stream between processing and oscope
        self.oscope_stream = OutputStream()
        
        self.oscope = QOscilloscope()
        self.layout.addWidget(self.oscope, 3)

    def show_params_controller(self):
        self.params_controller.show()

    def _configure(self, prb_filename=None, chan_grp = 0,
                                        chunksize=1000,
                                        refresh_interval=200, rate_interval=1000, 
                                        signalpreprocessor_params=None,
                                        peakdetector_params=None,
                                        ):
        
        self.prb_filename = prb_filename
        self.chan_grp = chan_grp
        self.chunksize = chunksize
        
        assert signalpreprocessor_params is not None
        assert peakdetector_params is not None
        
        self.signalpreprocessor_params = signalpreprocessor_params
        self.peakdetector_params = peakdetector_params
        
        self.probe = open_prb(prb_filename)
        self.in_group_channels = np.array(self.probe[self.chan_grp]['channels'], dtype='int64')
        
        self.geometry = np.array([self.probe[self.chan_grp]['geometry'][c] for c in self.in_group_channels], dtype='float64')

        self.nb_channel = len(self.in_group_channels)
        
        
        self.scatter = pg.ScatterPlotItem(pos=self.geometry, pxMode=False, size=10, brush='w')
        self.plot.addItem(self.scatter)
        
        x, y = self.geometry.mean(axis=0)
        #~ x, y = self.geometry[0, :]
        self.roi = pg.CircleROI([x-10, y-10], [20, 20], pen=(4,9))
        #~ self.plot.addItem(self.roi)
        
        self.roi.sigRegionChanged.connect(self.on_roi_change)
        
        xlim0 = np.min(self.geometry[:, 0]) - 20
        xlim1 = np.max(self.geometry[:, 0]) + 20
        ylim0 = np.min(self.geometry[:, 1]) - 20
        ylim1 = np.max(self.geometry[:, 1]) + 20
        self.plot.setXRange(xlim0, xlim1)
        self.plot.setYRange(ylim0, ylim1)
        
        self.reset_lut()
        
        
        self.refresh_interval = int(refresh_interval)  # ms
        self.rate_interval = int(rate_interval)  # ms
        self.nb_bin = self.rate_interval // self.refresh_interval
        #~ print('self.nb_bin'self.nb_bin, self.nb_bin)
        
    
    def _initialize(self):
        in_params = self.input.params

        # 
        self.timer = QT.QTimer(singleShot=False)
        self.timer.setInterval(self.refresh_interval)
        self.timer.timeout.connect(self._refresh)

        source_dtype = 'float32'
        
        self.sample_rate = self.input.params['sample_rate']
        
        # signal processor
        #~ p = dict(self.signalpreprocessor_params)
        #~ engine = p.pop('engine', 'numpy')
        #~ SignalPreprocessor_class = signalpreprocessor_engines[engine]
        #~ self.signalpreprocessor = SignalPreprocessor_class(self.sample_rate, self.nb_channel, self.chunksize, source_dtype)
        #~ p['normalize'] = False  # not computed yet
        #~ p['signals_medians'] = None
        #~ p['signals_mads'] = None
        #~ self.signalpreprocessor.change_params(**p)
        
        # peak detector
        #~ p = dict(self.peakdetector_params)
        #~ engine = p.pop('engine')
        #~ method = p.pop('method')
        #~ PeakDetector_class = get_peak_detector_class(method, engine)
        #~ self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        #~ self.chunksize, source_dtype, self.geometry)
        #~ self.peakdetector.change_params(**p)
        
        self.channel_names = [self.input.params['channel_info'][c]['name'] for c in self.in_group_channels]
        
        self.init_plot()
        

        # configure / initialize internal stream for oscope
        max_xsize = 10.
        stream_params = dict(
            protocol='tcp',
            interface='127.0.0.1',
            port='*',
            transfermode='sharedmem',
            streamtype='analogsignal',
            dtype='float32',
            shape=(-1, self.nb_channel),
            buffer_size=int((max_xsize+1) * self.sample_rate),
            sample_rate=float(self.sample_rate),
        )
        
        self.oscope_stream.configure(**stream_params)
        self.oscope_stream.params['channel_info'] = [{'name': name} for name in self.channel_names]
        
        self.oscope.configure(with_user_dialog=True, max_xsize=max_xsize)
        self.oscope.input.connect(self.oscope_stream)
        self.oscope.initialize()
        self.oscope.params['scale_mode'] = 'by_channel'
        self.oscope.params['display_labels'] = True
        
        self.thread = ProbeActivityThread(self.input, self, self.in_group_channels, self.geometry,
                    self.oscope_stream, self.signalpreprocessor_params, self.peakdetector_params)


    def _start(self):
        self.thread.noise_estimated = False
        #~ self.thread.nb_sample_for_noise = 0
        #~ self.thread.chunk_for_noise = []
        self.thread.start_new_noise_estimation()
        
        self.rate_per_channel = np.zeros(self.nb_channel, dtype='float64')
        self.count_per_channel = np.zeros((self.nb_channel, self.nb_bin), dtype='int64')
        self.bin_index = 0
        
        self.thread.start()
        self.timer.start()
        self.oscope.start()

    def _stop(self):
        self.thread.stop()
        self.thread.wait()
        self.timer.stop()
        self.oscope.stop()
    
    def _close(self):
        pass

    def _refresh(self):
        
        with self.mutex:
            self.rate_per_channel[:] = self.count_per_channel.sum(axis=1) / self.rate_interval * 1000
            self.bin_index = (self.bin_index + 1) % self.nb_bin
            self.count_per_channel[:, self.bin_index] = 0
        
        self.plot.removeItem(self.scatter)
        
        rate = self.rate_per_channel.copy()
        rate[rate>=self.params['max_rate']] = self.params['max_rate']
        rate_index = (rate / self.params['max_rate'] * 255).astype(int)
        
        colors = [ self.color_lut[ind] for ind in rate_index]

        self.scatter = pg.ScatterPlotItem(pos=self.geometry, pxMode=False, size=10, brush=colors)
        self.plot.addItem(self.scatter)

    def on_param_change(self, params, changes):
        for param, change, data in changes:
            if change != 'value': continue
            if param.name()=='max_rate':
                # in refersh
                pass
            if param.name()=='colormap':
                self.reset_lut()
            if param.name()=='show_channel_num':
                self.init_plot()
            if param.name()=='spacing_factor':
                self.oscope.auto_scale(spacing_factor=self.params['spacing_factor'])
    
    def init_plot(self):
        self.plot.clear()
        
        if self.params['show_channel_num']:
            for i in range(self.nb_channel):
                name = self.channel_names[i]
                x, y = self.geometry[i, :]
                itemtxt = pg.TextItem('{}: {}'.format(i, name), anchor=(.5,.5), color='#FFFF00')
                #~ itemtxt.setFont(QT.QFont('', pointSize=12))
                self.plot.addItem(itemtxt)
                itemtxt.setPos(x, y)
        
        self.plot.addItem(self.roi)
    
    def reset_lut(self):
        colormap = self.params['colormap']
        cmap = get_cmap(colormap, 256)
        self.color_lut = [cmap(i) for i in range(256)]
        self.color_lut = [ (int(r*255), int(g*255), int(b*255), int(a*255)) for r, g, b, a in self.color_lut]

    def on_roi_change(self):
        
        r = self.roi.state['size'][0] / 2
        x = self.roi.state['pos'].x() + r
        y = self.roi.state['pos'].y() + r
        
        #~ roi_shape = self.roi.shape()
        #~ print(roi_shape)
        
        
        mask_visible = np.sqrt(np.sum((self.geometry - np.array([[x, y]]))**2, axis=1)) < r
        
        for c in range(self.nb_channel):
            self.oscope.by_channel_params[f'ch{c}', 'visible'] = mask_visible[c]
        
        self.oscope.auto_scale(spacing_factor=self.params['spacing_factor'])

    def start_new_noise_estimation(self):
        self.thread.start_new_noise_estimation()


