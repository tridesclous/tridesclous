import os
import time

import numpy as np
from ..gui import QT
import pyqtgraph as pg

from pyqtgraph.util.mutex import Mutex

from pyacq.core import WidgetNode, InputStream, ThreadPollInput
from pyacq.rec import RawRecorder

from ..dataio import DataIO
from ..catalogueconstructor import CatalogueConstructor
from .onlinepeeler import OnlinePeeler
from .onlinetraceviewer import OnlineTraceViewer
from .onlinetools import make_empty_catalogue

from ..signalpreprocessor import estimate_medians_mads_after_preprocesing


"""
TODO:
  * nodegroup friend for peeler
  * compute_median_mad
  * compute catalogueconstructor
  * catalogue persistent in workdir
  * params GUI for signal processor peak detection
  * share data for catalogue workdir with other isntance if on same machine


"""

class OnlineWindow(WidgetNode):
    """
    Online spike sorting widget for ONE channel group:
        1. It start with an empty catalogue with no nosie estimation (medians/mads)
        2. Do an auto scale with timer
        3. Estimate medians/mads with use control and start spike with no label (-10=unlabbeled)
        4. Start a catalogue constructor on user demand
        5. Change the catalogue of the peeler with new cluser.
    
    
    """
    _input_specs = {'signals': dict(streamtype='signals')}
    
    request_compute = QT.pyqtSignal()
    
    def __init__(self, parent=None):
        WidgetNode.__init__(self, parent=parent)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        h = QT.QHBoxLayout()
        self.layout.addLayout(h)
        
        but = QT.QPushButton('auto scale')
        h.addWidget(but)
        but.clicked.connect(self.auto_scale_trace)

        but = QT.QPushButton('Estimate noise')
        h.addWidget(but)
        but.clicked.connect(self.compute_median_mad)

        but = QT.QPushButton('Start record for catalogue')
        h.addWidget(but)
        but.clicked.connect(self.start_rec_for_catalogue)
        
        self.traceviewer = OnlineTraceViewer()
        self.layout.addWidget(self.traceviewer)
        self.traceviewer.show()
        
        self.rtpeeler = OnlinePeeler()
        
        self.mutex = Mutex()
    
    def _configure(self, chan_grp=0, channel_indexes=[], chunksize=1024, workdir=''):
        self.chan_grp = chan_grp
        self.channel_indexes = np.array(channel_indexes, dtype='int64')
        self.chunksize = chunksize
        self.workdir = workdir
        
        #~ self.median_estimation_duration = 1
        self.median_estimation_duration = 3.
        self.catalogue_constructor_duration = 10.
        
        
    
    def after_input_connect(self, inputname):
        if inputname !='signals':
            return
        
        self.total_channel = self.input.params['shape'][1]
        assert np.all(self.channel_indexes<=self.total_channel), 'channel_indexes not compatible with total_channel'
        self.nb_channel = len(self.channel_indexes)
        self.sample_rate = self.input.params['sample_rate']
    
    def get_catalogue_params(self):
        # TODO do it with gui property and defutl
        params = dict(
            n_left=-20, n_right=40, internal_dtype='float32',
            
            #TODO
            preprocessor_params={},
            peak_detector_params={'relative_threshold' : 8.},
            clean_waveforms_params={},
            
            signals_medians=self.signals_medians,
            signals_mads=self.signals_mads,
            
        )
        if params['signals_medians'] is not None:
            params['signals_medians']  = params['signals_medians'] .copy()
            params['signals_mads']  = params['signals_mads'] .copy()
        
        
        return params
        
    
    def _initialize(self, **kargs):
        self.signals_medians = None
        self.signals_mads = None
        
        #TODO restore a persitent catalogue
        params = self.get_catalogue_params()
        params['peak_detector_params']['relative_threshold'] = np.inf
        self.catalogue = make_empty_catalogue(
                    channel_indexes=self.channel_indexes,
                    **params)
        
        # set a buffer on raw input for median/mad estimation
        buffer_size_margin = 3.
        self.input.set_buffer(size=int((self.median_estimation_duration+buffer_size_margin)*self.sample_rate),
                            double=True, axisorder=None, shmem=None, fill=None)
        self.thread_poll_input = ThreadPollInput(self.input)
        

        self.rtpeeler.configure(catalogue=self.catalogue, in_group_channels=self.channel_indexes, chunksize=self.chunksize)
        self.rtpeeler.input.connect(self.input.params)
        print(self.input.params)
        
        #TODO choose better stream params with sharedmem
        stream_params = dict(protocol='tcp', interface='127.0.0.1', transfermode='plaindata')
        self.rtpeeler.outputs['signals'].configure(**stream_params)
        self.rtpeeler.outputs['spikes'].configure(**stream_params)
        self.rtpeeler.initialize()
        
        
        self.traceviewer.configure(peak_buffer_size=1000, catalogue=self.catalogue)
        self.traceviewer.inputs['signals'].connect(self.rtpeeler.outputs['signals'])
        self.traceviewer.inputs['spikes'].connect(self.rtpeeler.outputs['spikes'])
        self.traceviewer.initialize()
        
        self.traceviewer.params['xsize'] = 1.
        self.traceviewer.params['decimation_method'] = 'min_max'
        self.traceviewer.params['mode'] = 'scan'
        self.traceviewer.params['scale_mode'] = 'same_for_all'


        # timer for autoscale
        self.timer_scale = QT.QTimer(singleShot=True, interval=500)
        self.timer_scale.timeout.connect(self.auto_scale_trace)
        # timer for median/mad
        self.timer_med = QT.QTimer(singleShot=True, interval=int(self.median_estimation_duration*1000)+1000)
        self.timer_med.timeout.connect(self.on_done_median_estimation_duration)
        # timer for catalogue
        self.timer_catalogue = QT.QTimer(singleShot=True, interval=int(self.catalogue_constructor_duration*1000)+1000)
        self.timer_catalogue.timeout.connect(self.on_done_catalogue_constructor)
        
        # stuf for recording a chunk for catalogue constructor
        if not os.path.exists(self.workdir):
             os.makedirs(self.workdir)
        self.rec = None
        self.dataio = None
        self.catalogueconstructor = None
        self.worker_thread = QT.QThread(parent=self)
        self.worker = None
    
    def _start(self):
        self.rtpeeler.start()
        self.traceviewer.start()
        
        self.thread_poll_input.start()
        self.worker_thread.start()
        self.timer_scale.start()
        

    def _stop(self):
        self.rtpeeler.stop()
        self.traceviewer.stop()
        
        self.thread_poll_input.stop()
        self.thread_poll_input.wait()
        
        self.worker_thread.quit()
        self.worker_thread.wait()
        
    def _close(self):
        pass
    
    def auto_scale_trace(self):
        # add factor in pyacq.oscilloscope autoscale (def compute_rescale)
        self.traceviewer.auto_scale(spacing_factor=25.)
    
    def compute_median_mad(self):
        """
        Wait for a while until input buffer is long anought to estimate the medians/mads
        """
        if self.timer_med.isActive():
            return
        
        self.timer_med.start()
        
        #~ self.tail = self.thread_poll_input.pos()
        #~ print('self.tail', self.tail)
    
    def on_done_median_estimation_duration(self):
        print('on_done_median_estimation_duration')
        head = self.thread_poll_input.pos()
        #~ print('self.tail', self.tail)
        #~ print('head', head)
        length = int((self.median_estimation_duration)*self.sample_rate)
        sigs = self.input.get_data(head-length, head, copy=False, join=True)
        #~ print(sigs.shape)
        
        
        self.signals_medians, self.signals_mads = estimate_medians_mads_after_preprocesing(
                        sigs[:, self.channel_indexes], self.sample_rate,
                        preprocessor_params=self.get_catalogue_params()['preprocessor_params'])
        print(self.signals_medians, self.signals_mads)
        
        params = self.get_catalogue_params() 
        catalogue = make_empty_catalogue(
                    channel_indexes=self.channel_indexes,
                    **params)
        
        self.change_catalogue(catalogue)
    
    def change_catalogue(self, catalogue):
        self.catalogue = catalogue
        self.rtpeeler.change_catalogue(self.catalogue)
        self.traceviewer.change_catalogue(self.catalogue)
        
        xsize = self.traceviewer.params['xsize']
        self.timer_scale.setInterval(int(xsize*1000.))
        self.timer_scale.start()
    
    def start_rec_for_catalogue(self):
        if self.timer_catalogue.isActive():
            return
        if self.rec is not None:
            return
        dirname = os.path.join(self.workdir, 'chan_grp{}'.format(self.chan_grp))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        dirname = os.path.join(dirname, 'raw_sigs')
        if os.path.exists(dirname):
            print('already exists')
            return

        
        
        self.rec = RawRecorder()
        
        print(dirname)
        self.rec.configure(streams=[self.input.params], autoconnect=True, dirname=dirname)
        self.rec.initialize()
        self.rec.start()
        
        self.timer_catalogue.start()

        
    def on_done_catalogue_constructor(self):
        print('on_done_median_estimation_duration')
        head = self.thread_poll_input.pos()
        #~ print('self.tail', self.tail)
        print('head', head)
        #~ length = int((self.median_estimation_duration)*self.sample_rate)
        #~ sigs = self.input.get_data(head-length, head, copy=False, join=True)
        
        self.rec.stop()
        self.rec.close()
        self.rec = None
        
        self.dataio = DataIO(dirname=os.path.join(self.workdir, 'chan_grp{}'.format(self.chan_grp), 'tdc_online'))
        
        

        #~ localdir, filenames, params = download_dataset(name='olfactory_bulb')
        #~ filenames = filenames[:1] #only first file
        filenames = os.path.join(self.workdir, 'chan_grp{}'.format(self.chan_grp), 'raw_sigs', 'input0.raw')
        self.dataio.set_data_source(type='RawData', filenames=filenames, sample_rate=self.sample_rate, 
                    dtype=self.input.params['dtype'], total_channel=self.total_channel)
        channel_group = {self.chan_grp:{'channels':self.channel_indexes.tolist()}}
        self.dataio.set_channel_groups(channel_group)
        
        print(self.dataio)

        self.catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=self.chan_grp)
        print(self.catalogueconstructor)
        
        params = self.get_catalogue_params() 
        
        self.worker = Worker(self.catalogueconstructor,params, self.chunksize)
        self.worker.moveToThread(self.worker_thread)
        self.request_compute.connect(self.worker.compute)
        self.worker.done.connect(self.on_compute_done)
        self.request_compute.emit()
    
    
    def on_compute_done(self):
        print('on_compute_done')
        catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grp)
        self.change_catalogue(catalogue)

class Worker(QT.QObject):
    #~ data_ready = QT.pyqtSignal(float, float, float, object, object, object, object, object)
    done = QT.pyqtSignal()
    def __init__(self, catalogueconstructor, params, chunksize, parent=None):
        QT.QObject.__init__(self, parent=parent)
        self.catalogueconstructor = catalogueconstructor
        self.params = params
        self.chunksize = chunksize
    
    def compute(self):
        print('compute')
        
        catalogueconstructor = self.catalogueconstructor
        
        d = {}
        d.update(self.params['preprocessor_params'])
        d.update(self.params['peak_detector_params'])
        
        catalogueconstructor.set_preprocessor_params(chunksize=self.chunksize, **d)
        print('youpi')


        t1 = time.perf_counter()
        catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.) #TODO
        t2 = time.perf_counter()
        print('estimate_signals_noise', t2-t1)
        
        t1 = time.perf_counter()
        catalogueconstructor.run_signalprocessor(duration=10., detect_peak=True)
        t2 = time.perf_counter()
        print('run_signalprocessor_loop', t2-t1)

        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=-25, n_right=40, mode='rand', nb_max=5000)
        t2 = time.perf_counter()
        print('extract_some_waveforms rand', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)

        t1 = time.perf_counter()
        catalogueconstructor.extract_some_waveforms(n_left=None, n_right=None, mode='rand', nb_max=20000)
        t2 = time.perf_counter()
        print('extract_some_waveforms rand', t2-t1)
        print(catalogueconstructor.some_waveforms.shape)

        t1 = time.perf_counter()
        catalogueconstructor.clean_waveforms(alien_value_threshold=60.)
        t2 = time.perf_counter()
        print('clean_waveforms', t2-t1)
        
        print(catalogueconstructor)
        

        #extract_some_noise
        t1 = time.perf_counter()
        catalogueconstructor.extract_some_noise(nb_snippet=400)
        t2 = time.perf_counter()
        print('extract_some_noise', t2-t1)
        
        #~ # PCA
        t1 = time.perf_counter()
        catalogueconstructor.project(method='global_pca', n_components=7, batch_size=16384)
        t2 = time.perf_counter()
        print('project pca', t2-t1)

        # peak_max
        #~ t1 = time.perf_counter()
        #~ catalogueconstructor.project(method='peak_max')
        #~ t2 = time.perf_counter()
        #~ print('project peak_max', t2-t1)
        #~ print(catalogueconstructor.some_features.shape)

        #~ t1 = time.perf_counter()
        #~ catalogueconstructor.extract_some_waveforms(index=np.arange(1000))
        #~ t2 = time.perf_counter()
        #~ print('extract_some_waveforms others', t2-t1)
        #~ print(catalogueconstructor.some_waveforms.shape)

        
        # cluster
        t1 = time.perf_counter()
        catalogueconstructor.find_clusters(method='sawchaincut')
        t2 = time.perf_counter()
        print('find_clusters', t2-t1)
        
        print(catalogueconstructor)
        
        self.catalogueconstructor.make_catalogue_for_peeler()
        


        
        time.sleep(1.)
        self.done.emit()
    
