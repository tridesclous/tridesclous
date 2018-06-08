import os
import time
import shutil

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
from ..gui import CatalogueWindow
from ..gui.mainwindow import error_box_msg, apply_all_catalogue_steps
from ..gui.gui_params import fullchain_params, features_params_by_methods, cluster_params_by_methods, peak_detector_params
from ..gui.tools import ParamDialog, MethodDialog, get_dict_from_group_param
from ..signalpreprocessor import estimate_medians_mads_after_preprocesing


"""
TODO:
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
        
        self.traceviewer = OnlineTraceViewer()
        #~ self.layout.addWidget(self.traceviewer)
        h.addWidget(self.traceviewer)
        self.traceviewer.show()


        self.toolbar = QT.QToolBar(orientation=QT.Vertical)
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        #~ self.addToolBar(QT.LeftToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))
        h.addWidget(self.toolbar)
        
        self.create_actions_and_menu()
        
        self.dialog_fullchain_params = ParamDialog(fullchain_params, parent=self)
        self.dialog_fullchain_params.params['duration'] = 10. # for debug
        self.dialog_fullchain_params.resize(450, 600)
        
        
        self.dialog_method_features = MethodDialog(features_params_by_methods, parent=self,
                        title='Which feature method ?', selected_method='peak_max')
        self.dialog_method_cluster = MethodDialog(cluster_params_by_methods, parent=self,
                        title='Which cluster method ?', selected_method = 'sawchaincut')
        
        
        
        self.mutex = Mutex()

    def create_actions_and_menu(self):
        #~ return
        
        #~ self.main_menu = self.menuBar().addMenu(self.tr("Main"))
        #~ self.main_menu = QT.QMenuBar()
        #~ self.toolbar.addWidget(self.main_menu)

        do_autoscale = QT.QAction('Auto scale', self, shortcut = "a" ) #, icon=QT.QIcon(":document-open.svg"))
        do_autoscale.triggered.connect(self.auto_scale_trace)
        #~ self.main_menu.addAction(do_autoscale)
        self.toolbar.addAction(do_autoscale)

        self.do_compute_median_mad = QT.QAction('Detec peak only', self) #, icon=QT.QIcon(":document-open.svg"))
        self.do_compute_median_mad.triggered.connect(self.compute_median_mad)
        #~ self.main_menu.addAction(do_autoscale)
        self.toolbar.addAction(self.do_compute_median_mad)

        self.do_start_rec = QT.QAction('Start new catalogue', self) #, icon=QT.QIcon(":document-open.svg"))
        self.do_start_rec.triggered.connect(self.start_new_catalogue)
        #~ self.main_menu.addAction(do_autoscale)
        self.toolbar.addAction(self.do_start_rec)

        do_open_cataloguewin = QT.QAction('Edit catalogue', self) #, icon=QT.QIcon(":document-open.svg"))
        do_open_cataloguewin.triggered.connect(self.open_cataloguewindow)
        #~ self.main_menu.addAction(do_autoscale)
        self.toolbar.addAction(do_open_cataloguewin)

        do_delete_catalogue = QT.QAction('Delete catalogue', self) #, icon=QT.QIcon(":document-open.svg"))
        do_delete_catalogue.triggered.connect(self.delete_catalogue)
        #~ self.main_menu.addAction(do_autoscale)
        self.toolbar.addAction(do_delete_catalogue)


    def warn(self, text, title='Error in tridesclous'):
        mb = QT.QMessageBox.warning(self, title,text, 
                QT.QMessageBox.Ok ,
                QT.QMessageBox.NoButton)
    
    def errorToMessageBox(self, e):
        self.warn(error_box_msg.format(e))
    
    def _configure(self, chan_grp=0, channel_indexes=[], chunksize=1024, workdir='',
                            nodegroup_friend=None):
        self.chan_grp = chan_grp
        self.channel_indexes = np.array(channel_indexes, dtype='int64')
        self.chunksize = chunksize
        self.workdir = workdir
        
        self.nodegroup_friend = nodegroup_friend

        if self.nodegroup_friend is None:
            self.rtpeeler = OnlinePeeler()
        else:
            self.nodegroup_friend.register_node_type_from_module('tridesclous.online', 'OnlinePeeler')
            self.rtpeeler = self.nodegroup_friend.create_node('OnlinePeeler')
        
        #~ self.median_estimation_duration = 1
        self.median_estimation_duration = 3.

        # prepare workdir
        #~ if not os.path.exists(self.workdir):
             #~ os.makedirs(self.workdir)
        dirname = os.path.join(self.workdir, 'chan_grp{}'.format(self.chan_grp))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.raw_sigs_dir = os.path.join(self.workdir, 'chan_grp{}'.format(self.chan_grp), 'raw_sigs')
        self.dataio_dir = os.path.join(self.workdir, 'chan_grp{}'.format(self.chan_grp), 'tdc_online')
        
        
    
    def after_input_connect(self, inputname):
        if inputname !='signals':
            return
        
        self.total_channel = self.input.params['shape'][1]
        assert np.all(self.channel_indexes<=self.total_channel), 'channel_indexes not compatible with total_channel'
        self.nb_channel = len(self.channel_indexes)
        self.sample_rate = self.input.params['sample_rate']
        
        channel_info = self.input.params.get('channel_info', None)
        if channel_info is None:
            self.channel_names = ['ch{}'.format(c) for c in range(self.nb_channel)]
        else:
            self.channel_names = [ch_info['name'] for ch_info in channel_info]

        #TODO
        if 1<= self.nb_channel <9:
            feat_method = 'global_pca'
        else: 
            feat_method = 'peak_max'
        
        self.dialog_method_features.param_method['method'] = feat_method


    def _initialize(self, **kargs):
        self.signals_medians = None
        self.signals_mads = None
        
        try:
            #try to load persitent catalogue
            self.dataio = DataIO(dirname=self.dataio_dir)
            self.catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=self.chan_grp)
            print(self.catalogueconstructor)
            self.catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grp)
        except:
            # work with empty catalogue
            self.dataio = None
            self.catalogueconstructor = None
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
        

        self.rtpeeler.configure(catalogue=self.catalogue, 
                    in_group_channels=self.channel_indexes, chunksize=self.chunksize)
        self.rtpeeler.input.connect(self.input.params)
        #~ print(self.input.params)
        
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
        self.timer_catalogue = QT.QTimer(singleShot=True)
        self.timer_catalogue.timeout.connect(self.on_raw_signals_recorded)
        
        
        # stuf for recording a chunk for catalogue constructor
        self.rec = None
        self.dataio = None
        self.catalogueconstructor = None
        self.cataloguewindow = None
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

    def get_catalogue_params(self):
        # TODO do it with gui property and defutl
        
        p = self.dialog_fullchain_params.get()
        p['preprocessor'].pop('chunksize')
        
        params = dict(
            #~ n_left=-20,
            n_left=p['extract_waveforms']['n_left'],
            #~ n_right=40,
            n_right=p['extract_waveforms']['n_right'],
            
            internal_dtype='float32',
            
            #TODO
            preprocessor_params=p['preprocessor'],
            peak_detector_params=p['peak_detector'], #{'relative_threshold' : 8.},
            clean_waveforms_params=p['clean_waveforms'],
            
            signals_medians=self.signals_medians,
            signals_mads=self.signals_mads,
            
        )
        if params['signals_medians'] is not None:
            params['signals_medians']  = params['signals_medians'] .copy()
            params['signals_mads']  = params['signals_mads'] .copy()
        
        print(params)
        return params    
    
    def auto_scale_trace(self):
        # add factor in pyacq.oscilloscope autoscale (def compute_rescale)
        self.traceviewer.auto_scale(spacing_factor=25.)
    
    def compute_median_mad(self):
        """
        Wait for a while until input buffer is long anought to estimate the medians/mads
        """
        if self.timer_med.isActive():
            return
        
        if not self.dialog_fullchain_params.exec_():
            return

        self.timer_med.start()
        
        but = self.toolbar.widgetForAction(self.do_compute_median_mad)
        but.setStyleSheet("QToolButton:!hover { background-color: red }")

        
        #~ self.tail = self.thread_poll_input.pos()
        #~ print('self.tail', self.tail)
    
    def on_done_median_estimation_duration(self):
        
        but = self.toolbar.widgetForAction(self.do_compute_median_mad)
        but.setStyleSheet("")
        
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

    def on_new_catalogue(self):
        print('on_new_catalogue')
        catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grp)
        self.change_catalogue(catalogue)

    
    def start_new_catalogue(self):
        if self.timer_catalogue.isActive():
            return
        if self.rec is not None:
            return
        
        if os.path.exists(self.raw_sigs_dir):
            self.warn('A catalogue already exists.\nDelete it and start rec again.')
            return

        
        if not self.dialog_fullchain_params.exec_():
            return
        
        if not self.dialog_method_features.exec_():
            return

        if not self.dialog_method_cluster.exec_():
            return
        
        # get duration for raw sigs record
        fullchain_kargs = self.dialog_fullchain_params.get()
        self.timer_catalogue.setInterval(int((fullchain_kargs['duration']+1)*1000.))
        

        if self.cataloguewindow is not None:
            self.cataloguewindow.close()
            self.cataloguewindow = None
        if self.catalogueconstructor is not None:
            self.catalogueconstructor = None
        self.dataio = None
        
        self.rec = RawRecorder()
        self.rec.configure(streams=[self.input.params], autoconnect=True, dirname=self.raw_sigs_dir)
        self.rec.initialize()
        self.rec.start()
        
        self.timer_catalogue.start()
        but = self.toolbar.widgetForAction(self.do_start_rec)
        but.setStyleSheet("QToolButton:!hover { background-color: red }")


        
    def on_raw_signals_recorded(self):
        print('on_raw_signals_recorded')
        self.rec.stop()
        self.rec.close()
        
        but = self.toolbar.widgetForAction(self.do_start_rec)
        but.setStyleSheet("")
        
        self.rec = None
        
        self.dataio = DataIO(dirname=self.dataio_dir)
        filenames = [os.path.join(self.raw_sigs_dir, 'input0.raw')]
        self.dataio.set_data_source(type='RawData', filenames=filenames, sample_rate=self.sample_rate, 
                    dtype=self.input.params['dtype'], total_channel=self.total_channel)
        channel_group = {self.chan_grp:{'channels':self.channel_indexes.tolist()}}
        self.dataio.set_channel_groups(channel_group)
        
        print(self.dataio)

        self.catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=self.chan_grp)
        print(self.catalogueconstructor)
        
        #~ params = self.get_catalogue_params()
        fullchain_kargs = self.dialog_fullchain_params.get()
        fullchain_kargs['preprocessor']['chunksize'] = self.chunksize
        
        feat_method = self.dialog_method_features.param_method['method']
        feat_kargs = get_dict_from_group_param(self.dialog_method_features.all_params[feat_method], cascade=True)

        clust_method = self.dialog_method_cluster.param_method['method']
        clust_kargs = get_dict_from_group_param(self.dialog_method_cluster.all_params[clust_method], cascade=True)

        print('feat_method', feat_method, 'clust_method', clust_method)
        self.worker = Worker(self.catalogueconstructor, fullchain_kargs,
                feat_method, feat_kargs,
                clust_method, clust_kargs)
        
        
        
        self.worker.moveToThread(self.worker_thread)
        self.request_compute.connect(self.worker.compute)
        self.worker.done.connect(self.on_new_catalogue)
        self.worker.compute_catalogue_error.connect(self.on_compute_catalogue_error)
        self.request_compute.emit()
    
    def on_compute_catalogue_error(self, e):
        self.errorToMessageBox(e)
    
    def open_cataloguewindow(self):
        if self.catalogueconstructor is None:
            return
        
        if self.cataloguewindow is  None:
            self.cataloguewindow = CatalogueWindow(self.catalogueconstructor)
            self.cataloguewindow.new_catalogue.connect(self.on_new_catalogue)
        
        self.cataloguewindow.show()
    
    def delete_catalogue(self):
        # this delete catalogue and raw sigs
        if self.cataloguewindow is not None:
            self.cataloguewindow.close()
            self.cataloguewindow = None
        if self.catalogueconstructor is not None:
            self.catalogueconstructor = None
        self.dataio = None
        
        if os.path.exists(self.dataio_dir):
            shutil.rmtree(self.dataio_dir)
        if os.path.exists(self.raw_sigs_dir):
            shutil.rmtree(self.raw_sigs_dir)
        
        # make empty catalogue
        params = self.get_catalogue_params() 
        catalogue = make_empty_catalogue(
                    channel_indexes=self.channel_indexes,
                    **params)
        self.change_catalogue(catalogue)



class Worker(QT.QObject):
    #~ data_ready = QT.pyqtSignal(float, float, float, object, object, object, object, object)
    done = QT.pyqtSignal()
    compute_catalogue_error = QT.pyqtSignal(object)
    def __init__(self, catalogueconstructor, fullchain_kargs, 
                feat_method, feat_kargs,
                clust_method, clust_kargs, parent=None):
        QT.QObject.__init__(self, parent=parent)
        
        self.catalogueconstructor = catalogueconstructor
        self.fullchain_kargs = fullchain_kargs
        self.feat_method = feat_method
        self.feat_kargs = feat_kargs
        self.clust_method = clust_method
        self.clust_kargs = clust_kargs
    
    def compute(self):
        print('compute')
        
        catalogueconstructor = self.catalogueconstructor
        print(self.catalogueconstructor.dataio)
        print('self.fullchain_kargs duration', self.fullchain_kargs['duration'])
        
        try:
            apply_all_catalogue_steps(self.catalogueconstructor, self.fullchain_kargs, 
                    self.feat_method, self.feat_kargs, self.clust_method, self.clust_kargs,
                    #~ verbose=False,
                    verbose=True,
                    )
            
            self.catalogueconstructor.make_catalogue_for_peeler()
        except Exception as e:
            self.compute_catalogue_error.emit(e)
            return
            
        
        #~ time.sleep(1.)
        self.done.emit()
