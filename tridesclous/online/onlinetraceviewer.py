import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


import pyacq
from pyacq import WidgetNode,ThreadPollInput, StreamConverter, InputStream
from pyacq.viewers import QOscilloscope

#~ _dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]
from ..peeler import _dtype_spike


class OnlineTraceViewer(QOscilloscope):
    
    _input_specs = {'signals': dict(streamtype='signals'),
                                'spikes': dict(streamtype='events', shape = (-1, ),  dtype=_dtype_spike),
                                    }
    
    _default_params = QOscilloscope._default_params

    
    def __init__(self, **kargs):
        QOscilloscope.__init__(self, **kargs)

    def _configure(self, peak_buffer_size = 1000, **kargs):
        QOscilloscope._configure(self, **kargs)
        self.peak_buffer_size = peak_buffer_size
    
    def _initialize(self, **kargs):
        QOscilloscope._initialize(self, **kargs)
        
        self.inputs['spikes'].set_buffer(size=self.peak_buffer_size, double=False)#TODO check if double is necessary
        
        # poller onpeak
        self._last_peak = 0
        self.poller_peak = ThreadPollInput(input_stream=self.inputs['spikes'], return_data=True)
        self.poller_peak.new_data.connect(self._on_new_peak)
        
        self.peaks_array = self.inputs['spikes'].buffer.buffer
        
        self.scatters = []
        for i in range(self.nb_channel):
            color = QtGui.QColor('#FF00FF')#TODO
            color.setAlpha(150)
            scatter = pg.ScatterPlotItem(x=[ ], y= [ ], pen=None, brush=color, size=10, pxMode = True)
            self.scatters.append(scatter)
            self.plot.addItem(scatter)
        

    def _start(self, **kargs):
        QOscilloscope._start(self, **kargs)
        self._last_peak = 0
        #~ if self.conv_peak is not None:
            #~ self.conv_peak.start()
        self.poller_peak.start()

    def _stop(self, **kargs):
        QOscilloscope._stop(self, **kargs)
        #~ if self.conv_peak is not None:
            #~ self.conv_peak.stop()
        self.poller_peak.stop()
        self.poller_peak.wait()

    def _close(self, **kargs):
        QOscilloscope._close(self, **kargs)
    
    def reset_curves_data(self):
        QOscilloscope.reset_curves_data(self)
        #~ sr = self.inputs['signals'].params['sample_rate']
        self.t_vect_full = np.arange(0,self.full_size, dtype=float)/self.sample_rate
        self.t_vect_full -= self.t_vect_full[-1]
    
    def _on_new_peak(self, pos, data):
        self._last_peak = pos
    
    def autoestimate_scales(self):
        #~ print(self.inputs['signals'].params)
        #~ n = self.inputs['signals'].params['nb_channel']
        self.all_mean = np.zeros(self.nb_channel,)
        self.all_sd = np.ones(self.nb_channel,)
        return self.all_mean, self.all_sd

    
    def _refresh(self, **kargs):
        QOscilloscope._refresh(self, **kargs)
        
        mode = self.params['mode']
        gains = np.array([p['gain'] for p in self.by_channel_params.children()])
        offsets = np.array([p['offset'] for p in self.by_channel_params.children()])
        visibles = np.array([p['visible'] for p in self.by_channel_params.children()], dtype=bool)
        
        head = self._head
        full_arr = self.inputs['signals'].get_data(head-self.full_size, head)
        if self._last_peak==0:
            return

        keep = (self.peaks_array['index']>head - self.full_size) & (self.peaks_array['index']<head)
        peaks = self.peaks_array['index'][keep]
        peaks_ind = peaks - (head - self.full_size)
        peaks_amplitude = full_arr[peaks_ind, :]
        peaks_amplitude[:, visibles] *= gains[visibles]
        peaks_amplitude[:, visibles] += offsets[visibles]
        
        if mode=='scroll':
            peak_times = self.t_vect_full[peaks_ind]
        elif mode =='scan':
            #some trick to play with fake time
            front = head % self.full_size
            ind1 = (peaks%self.full_size)<front
            ind2 = (peaks%self.full_size)>front
            peak_times = self.t_vect_full[peaks_ind]
            peak_times[ind1] += (self.t_vect_full[front] - self.t_vect_full[-1])
            peak_times[ind2] += (self.t_vect_full[front] - self.t_vect_full[0])
        
        for c, visible in enumerate(visibles):
            if visible:
                self.scatters[c].setData(peak_times, peaks_amplitude[:, c])

