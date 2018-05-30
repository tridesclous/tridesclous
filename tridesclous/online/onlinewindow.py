import numpy as np
from ..gui import QT
import pyqtgraph as pg

from pyacq.core import WidgetNode


from .onlinepeeler import OnlinePeeler
from .onlinetraceviewer import OnlineTraceViewer
from .onlinetools import make_empty_catalogue

"""
TODO:
  * nodegroup friend for peeler
  * compute_median_mad
  * compute catalogueconstructor
  * catalogue persistent in workdir
  * params GUI for signal processor peak detection


"""

class OnlineWindow(WidgetNode):
    
    _input_specs = {'signals': dict(streamtype='signals')}
    
    def __init__(self, parent=None):
        WidgetNode.__init__(self, parent=parent)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        h = QT.QHBoxLayout()
        
        
        self.traceviewer = OnlineTraceViewer()
        self.layout.addWidget(self.traceviewer)
        self.traceviewer.show()
        
        self.peeler = OnlinePeeler()
    
    def _configure(self, chan_grp=0, channel_indexes=[], chunksize=1024, workdir=''):
        self.chan_grp = chan_grp
        self.channel_indexes = np.array(channel_indexes, dtype='int64')
        self.chunksize = chunksize
        self.workdir = workdir
    
    def after_input_connect(self, inputname):
        if inputname !='signals':
            return
        
        self.total_channel = self.input.params['shape'][1]
        assert np.all(self.channel_indexes<=self.total_channel), 'channel_indexes not compatible with total_channel'
        self.nb_channel = len(self.channel_indexes)
        
    
    def _initialize(self, **kargs):
        #TODO restore a persitent catalogue
        self.catalogue = make_empty_catalogue(
                    channel_indexes=self.channel_indexes,
                    n_left=-20, n_right=40, internal_dtype='float32',
                    peak_detector_params={'relative_threshold': np.inf},
                    signals_medians = None,
                    signals_mads = None,
            )



        self.peeler.configure(catalogue=self.catalogue, in_group_channels=self.channel_indexes, chunksize=self.chunksize)
        self.peeler.input.connect(self.input.params)
        print(self.input.params)
        
        #TODO choose better stream params with sharedmem
        stream_params = dict(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
        self.peeler.outputs['signals'].configure(**stream_params)
        self.peeler.outputs['spikes'].configure(**stream_params)
        self.peeler.initialize()
        
        
        self.traceviewer.configure(peak_buffer_size=1000, catalogue=self.catalogue)
        self.traceviewer.inputs['signals'].connect(self.peeler.outputs['signals'])
        self.traceviewer.inputs['spikes'].connect(self.peeler.outputs['spikes'])
        self.traceviewer.initialize()
        
        self.traceviewer.params['xsize'] = 3.
        self.traceviewer.params['decimation_method'] = 'min_max'
        self.traceviewer.params['mode'] = 'scan'
        self.traceviewer.params['scale_mode'] = 'same_for_all'
        
        
    
    def _start(self):
        self.peeler.start()
        self.traceviewer.start()
        
        self.timer = QT.QTimer(singleShot=True, interval=1000)
        self.timer.timeout.connect(self.auto_scale_trace)
        self.timer.start()    
        
        
    def _stop(self):
        self.peeler.stop()
        self.traceviewer.stop()
        
        
    def _close(self):
        pass
    
    def auto_scale_trace(self):
        # add factor in pyacq.oscilloscope autoscale (def compute_rescale)
        self.traceviewer.auto_scale()
    
    def compute_median_mad(self):
        pass
    #~ preprocessor_params = {}
    #~ signals_medians, signals_mads = estimate_medians_mads_after_preprocesing(sigs[:, channel_indexes], sample_rate,
                        #~ preprocessor_params=preprocessor_params)
    #~ empty_catalogue = make_empty_catalogue(
                #~ channel_indexes=channel_indexes,
                #~ n_left=-20, n_right=40, internal_dtype='float32',                
                #~ preprocessor_params=preprocessor_params,
                #~ peak_detector_params={'relative_threshold': 10},
                #~ clean_waveforms_params={},
                
                #~ signals_medians = signals_medians,
                #~ signals_mads = signals_mads,
        
        #~ )
        
        

    
