import numpy as np
#~ from pyqtgraph.Qt import QtCore, QtGui
from ..gui import QT
import pyqtgraph as pg
from pyqtgraph.util.mutex import Mutex

import pyacq
from pyacq import WidgetNode,ThreadPollInput, StreamConverter, InputStream
from pyacq.viewers import QOscilloscope

#~ _dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]
from ..peeler_tools import _dtype_spike
from ..tools import make_color_dict
from ..labelcodes import LABEL_UNCLASSIFIED




    
class OnlineTraceViewer(QOscilloscope):
    
    _input_specs = {'signals': dict(streamtype='signals'),
                                'spikes': dict(streamtype='events', shape = (-1, ),  dtype=_dtype_spike),
                                    }
    
    _default_params = QOscilloscope._default_params

    
    def __init__(self, **kargs):
        QOscilloscope.__init__(self, **kargs)
        self.mutex = Mutex()

    def _configure(self, peak_buffer_size = 100000, catalogue=None, **kargs):
        QOscilloscope._configure(self, **kargs)
        self.peak_buffer_size = peak_buffer_size
        self.catalogue = catalogue
        assert catalogue is not None
    
    def _initialize(self, **kargs):
        QOscilloscope._initialize(self, **kargs)
        
        self.inputs['spikes'].set_buffer(size=self.peak_buffer_size, double=False)
        
        # poller onpeak
        self._last_peak = 0
        self.poller_peak = ThreadPollInput(input_stream=self.inputs['spikes'], return_data=True)
        self.poller_peak.new_data.connect(self._on_new_peak)
        
        self.spikes_array = self.inputs['spikes'].buffer.buffer
        
        self.scatters = {}
        self.change_catalogue(self.catalogue)

        self.params['xsize'] = 1.
        self.params['decimation_method'] = 'min_max'
        self.params['mode'] = 'scan'
        self.params['scale_mode'] = 'same_for_all'
        self.params['display_labels'] = True
        
        self.timer_scale = QT.QTimer(singleShot=True, interval=500)
        self.timer_scale.timeout.connect(self.auto_scale)
        self.timer_scale.start()

    def _start(self, **kargs):
        QOscilloscope._start(self, **kargs)
        self._last_peak = 0
        self.poller_peak.start()

    def _stop(self, **kargs):
        QOscilloscope._stop(self, **kargs)
        self.poller_peak.stop()
        self.poller_peak.wait()

    def _close(self, **kargs):
        QOscilloscope._close(self, **kargs)
    
    def reset_curves_data(self):
        QOscilloscope.reset_curves_data(self)
        self.t_vect_full = np.arange(0,self.full_size, dtype=float)/self.sample_rate
        self.t_vect_full -= self.t_vect_full[-1]
    
    def _on_new_peak(self, pos, data):
        self._last_peak = pos
    
    def autoestimate_scales(self):
        # in our case preprocesssed signal is supposed to be normalized
        self.all_mean = np.zeros(self.nb_channel,)
        self.all_sd = np.ones(self.nb_channel,)
        return self.all_mean, self.all_sd
    
    def change_catalogue(self, catalogue):
        with self.mutex:
            
            for k, v in self.scatters.items():
                self.plot.removeItem(v)
            self.scatters = {}
            
            self.catalogue = catalogue

            colors = make_color_dict(self.catalogue['clusters'])
            
            self.qcolors = {}
            for k, color in colors.items():
                r, g, b = color
                self.qcolors[k] = QT.QColor(int(r*255), int(g*255), int(b*255))
            
            self.all_plotted_labels = self.catalogue['cluster_labels'].tolist() + [LABEL_UNCLASSIFIED]
            
            for k in self.all_plotted_labels:
                qcolor = self.qcolors[k]
                qcolor.setAlpha(150)
                scatter = pg.ScatterPlotItem(x=[ ], y= [ ], pen=None, brush=qcolor, size=10, pxMode = True)
                self.scatters[k] = scatter
                self.plot.addItem(scatter)

            
            
        
    
    def _refresh(self, **kargs):
        if self.visibleRegion().isEmpty():
            # when several tabs not need to refresh
            return
        
        with self.mutex:
            QOscilloscope._refresh(self, **kargs)
            
            mode = self.params['mode']
            gains = np.array([p['gain'] for p in self.by_channel_params.children()])
            offsets = np.array([p['offset'] for p in self.by_channel_params.children()])
            visibles = np.array([p['visible'] for p in self.by_channel_params.children()], dtype=bool)
            
            head = self._head
            full_arr = self.inputs['signals'].get_data(head-self.full_size, head)
            if self._last_peak==0:
                return

            keep = (self.spikes_array['index']>head - self.full_size) & (self.spikes_array['index']<head)
            spikes = self.spikes_array[keep]
            
            spikes_ind = spikes['index'] - (head - self.full_size)
            spikes_ind = spikes_ind[spikes_ind<full_arr.shape[0]] # to avoid bug if last peak is great than head
            real_spikes_amplitude = full_arr[spikes_ind, :]
            spikes_amplitude = real_spikes_amplitude.copy()
            spikes_amplitude[:, visibles] *= gains[visibles]
            spikes_amplitude[:, visibles] += offsets[visibles]
            
            if mode=='scroll':
                peak_times = self.t_vect_full[spikes_ind]
            elif mode =='scan':
                #some trick to play with fake time
                front = head % self.full_size
                ind1 = (spikes['index']%self.full_size)<front
                ind2 = (spikes['index']%self.full_size)>front
                peak_times = self.t_vect_full[spikes_ind]
                peak_times[ind1] += (self.t_vect_full[front] - self.t_vect_full[-1])
                peak_times[ind2] += (self.t_vect_full[front] - self.t_vect_full[0])
            
            for i, k in enumerate(self.all_plotted_labels):
                keep = k==spikes['cluster_label']
                if np.sum(keep)>0:
                    if k>=0:
                        chan = self.catalogue['clusters']['extremum_channel'][i]
                        if visibles[chan]:
                            times, amps = peak_times[keep], spikes_amplitude[keep, :][:, chan]
                        else:
                            times, amps = [], []
                            
                    else:
                        chan_max = np.argmax(np.abs(real_spikes_amplitude[keep, :]), axis=1)
                        keep2 = visibles[chan_max]
                        chan_max = chan_max[keep2]
                        keep[keep] &= keep2
                        times, amps = peak_times[keep], spikes_amplitude[keep, chan_max]
                    
                    self.scatters[k].setData(times, amps)
                    
                else:
                    self.scatters[k].setData([], [])
        
    def auto_scale(self, spacing_factor=25.):
        self.params_controller.compute_rescale(spacing_factor=spacing_factor)
        self.refresh()