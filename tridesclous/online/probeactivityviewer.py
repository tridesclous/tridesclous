
from pprint import pprint

from matplotlib.cm import get_cmap
import numpy as np

from pyacq.core import WidgetNode, ThreadPollInput, OutputStream
from pyacq.viewers import QOscilloscope

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.util.mutex import Mutex

import tridesclous as tdc
from tridesclous.tools import open_prb, median_mad
from tridesclous.signalpreprocessor import signalpreprocessor_engines
from tridesclous.peakdetector import get_peak_detector_class



class ProbeActivityViewer(WidgetNode):
    """
    
    """
    _input_specs = {'signals': dict()}
    
    _default_params = [
            {'name': 'max_rate', 'type': 'float', 'value': 6., 'step': 0.5},
            {'name': 'show_channel_num', 'type': 'bool', 'value': True},
            
            
    ]
    
    
    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)
        
        self.layout = QtGui.QHBoxLayout()
        self.setLayout(self.layout)
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview, 1)
        
        self.plot = pg.PlotItem()
        self.plot.getViewBox().disableAutoRange()
        self.graphicsview.setCentralItem(self.plot)
        self.plot.getViewBox().setAspectLocked(lock=True, ratio=1)
        self.plot.hideButtons()
        self.plot.showAxis('left', False)
        self.plot.showAxis('bottom', False)
        
        
        self.params = pg.parametertree.Parameter.create(name='Global options',
                                                    type='group', children=self._default_params)
        
        self.mutex = Mutex()
        
        # stream between processing and oscope
        self.internal_stream = OutputStream()
        
        self.oscope = QOscilloscope()
        self.layout.addWidget(self.oscope, 3)

        
        


    def _configure(self, prb_filename=None, chan_grp = 0,
                                        refresh_interval=200, rate_interval=1000, 
                                        signalpreprocessor_params=None,
                                        peakdetector_params=None,
                                        ):
        
        self.prb_filename = prb_filename
        self.chan_grp = chan_grp
        
        assert signalpreprocessor_params is not None
        assert peakdetector_params is not None
        
        self.signalpreprocessor_params = signalpreprocessor_params
        self.peakdetector_params = peakdetector_params        
        
        
        
        
        self.probe = open_prb(prb_filename)
        self.in_group_channels = np.array(self.probe[self.chan_grp]['channels'], dtype='int64')
        
        
        
        self.geometry = np.array([self.probe[self.chan_grp]['geometry'][c] for c in self.in_group_channels], dtype='float64')

        
        
        self.nb_channel = self.geometry.shape[0]
        
        self.scatter = pg.ScatterPlotItem(pos=self.geometry, pxMode=False, size=10, brush='w')
        self.plot.addItem(self.scatter)
        
        #~ x, y = self.geometry.mean(axis=0)
        x, y = self.geometry[0, :]
        self.roi = pg.CircleROI([x-10, y-10], [20, 20], pen=(4,9))
        self.plot.addItem(self.roi)
        
        self.roi.sigRegionChanged.connect(self.on_roi_change)
        
        xlim0 = np.min(self.geometry[:, 0]) - 20
        xlim1 = np.max(self.geometry[:, 0]) + 20
        ylim0 = np.min(self.geometry[:, 1]) - 20
        ylim1 = np.max(self.geometry[:, 1]) + 20
        self.plot.setXRange(xlim0, xlim1)
        self.plot.setYRange(ylim0, ylim1)
        
        
        cmap = get_cmap('inferno', 256)
        self.color_lut = [cmap(i) for i in range(256)]
        #~ print(self.color_lut)
        self.color_lut = [ (int(r*255), int(g*255), int(b*255), int(a*255)) for r, g, b, a in self.color_lut]
        #~ print(self.color_lut)
        
        
        self.refresh_interval = int(refresh_interval)  # ms
        self.rate_interval = int(rate_interval)  # ms
        self.nb_bin = self.rate_interval // self.refresh_interval
        #~ print('self.nb_bin'self.nb_bin, self.nb_bin)
        
        self.chunksize = 1024



        
        
        
        
    
    def _initialize(self):
        in_params = self.input.params

        # poller
        self.poller = ThreadPollInput(input_stream=self.inputs['signals'], return_data=True)
        self.poller.new_data.connect(self._on_new_data)

        # 
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.setInterval(self.refresh_interval)
        self.timer.timeout.connect(self._refresh)

        source_dtype = 'float32'
        
        # signal processor
        p = dict(self.signalpreprocessor_params)
        engine = p.pop('engine', 'numpy')
        SignalPreprocessor_class = signalpreprocessor_engines[engine]
        self.sample_rate = self.input.params['sample_rate']
        self.signalpreprocessor = SignalPreprocessor_class(self.sample_rate, self.nb_channel, self.chunksize, source_dtype)
        p['normalize'] = False  # not computed yet
        p['signals_medians'] = None
        p['signals_mads'] = None
        self.signalpreprocessor.change_params(**p)
        
        # peak detector
        p = dict(self.peakdetector_params)
        engine = p.pop('engine')
        method = p.pop('method')
        PeakDetector_class = get_peak_detector_class(method, engine)
        self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        self.chunksize, source_dtype, self.geometry)
        self.peakdetector.change_params(**p)
        
        self.channel_names = [self.input.params['channel_info'][c]['name'] for c in self.in_group_channels]
        
        if self.params['show_channel_num']:
            for i in range(self.nb_channel):
                name = self.channel_names[i]
                x, y = self.geometry[i, :]
                itemtxt = pg.TextItem('{}: {}'.format(i, name), anchor=(.5,.5), color='#FFFF00')
                #~ itemtxt.setFont(QT.QFont('', pointSize=12))
                self.plot.addItem(itemtxt)
                itemtxt.setPos(x, y)

        # configure / initialize internal stream
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
        
        self.internal_stream.configure(**stream_params)
        self.internal_stream.params['channel_info'] = [{'name': name} for name in self.channel_names]
        
        self.oscope.configure(with_user_dialog=True, max_xsize=max_xsize)
        self.oscope.input.connect(self.internal_stream)
        self.oscope.initialize()
        self.oscope.params['scale_mode'] = 'by_channel'
        self.oscope.params['display_labels'] = True


    def _start(self):
        self.noise_estimated = False
        self.nb_sample_for_noise = 0
        self.chunk_for_noise = []
        
        self.rate_per_channel = np.zeros(self.nb_channel, dtype='float64')
        self.count_per_channel = np.zeros((self.nb_channel, self.nb_bin), dtype='int64')
        self.bin_index = 0
        
        self.poller.start()
        self.timer.start()
        self.oscope.start()

    def _stop(self):
        self.poller.stop()
        self.timer.stop()
        self.oscope.stop()
    
    def _close(self):
        pass
    
    def _on_new_data(self, pos, data):
        self._head = pos
        #~ print('_on_new_data', pos, data.shape)
        
        data_in_group = data[:, self.in_group_channels]
        
        # preprocessor
        if self.noise_estimated:
            out_pos, preprocessed_data = self.signalpreprocessor.process_data(pos, data_in_group)
            
        else:
            out_pos, preprocessed_data = self.signalpreprocessor.process_data(pos, data_in_group)
            self.chunk_for_noise.append(preprocessed_data)
            self.nb_sample_for_noise += preprocessed_data.shape[0]
            
            
            
            if self.nb_sample_for_noise > int(self.sample_rate * 5):
                # 5s
                noise = np.concatenate(self.chunk_for_noise)
                signals_medians, signals_mads = median_mad(noise, axis=0)

                p = dict(self.signalpreprocessor_params)
                engine = p.pop('engine', 'numpy')
                SignalPreprocessor_class = signalpreprocessor_engines[engine]
                self.sample_rate = self.input.params['sample_rate']
                source_dtype = 'float32'
                chunksize = 1024
                self.signalpreprocessor = SignalPreprocessor_class(self.sample_rate, self.nb_channel, chunksize, source_dtype)
                p['normalize'] = True
                p['signals_medians'] = signals_medians
                p['signals_mads'] = signals_mads
                self.signalpreprocessor.change_params(**p)
                print('estimate noise done')
                
                self.noise_estimated = True
                
                self.timer_autoscale = QtCore.QTimer(singleShot=True, interval=2000)
                self.timer_autoscale.timeout.connect(self.on_roi_change)
                self.timer_autoscale.start()
                
                
        
        if  not self.noise_estimated:
            return
        
        # send to oscope
        self.internal_stream.send(preprocessed_data, index=out_pos)
        
        # peak detector
        time_ind_peaks, chan_ind_peaks, peak_val_peaks = self.peakdetector.process_data(out_pos, preprocessed_data)
        
        
        if time_ind_peaks is None:
            return
        
        # update count
        with self.mutex:
            self.count_per_channel[chan_ind_peaks, self.bin_index] += 1


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


    def on_roi_change(self):
        #~ if  not self.noise_estimated:
            #~ return
        
        r = self.roi.state['size'][0] / 2
        x = self.roi.state['pos'].x() + r
        y = self.roi.state['pos'].y() + r
        
        mask_visible = np.sqrt(np.sum((self.geometry - np.array([[x, y]]))**2, axis=1)) < r
        
        #~ selected_chans,  = np.nonzero(mask)
        #~ print(selected_chans)
        
        for c in range(self.nb_channel):
            self.oscope.by_channel_params[f'ch{c}', 'visible'] = mask_visible[c]
        
        self.oscope.auto_scale()
        
        



