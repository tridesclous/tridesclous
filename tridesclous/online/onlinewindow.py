import os
import time
import shutil

import numpy as np
from ..gui import QT
import pyqtgraph as pg

from pyqtgraph.util.mutex import Mutex

import pyacq
from pyacq.core import WidgetNode, InputStream, ThreadPollInput
from pyacq.rec import RawRecorder

from ..dataio import DataIO
from ..catalogueconstructor import CatalogueConstructor
from .onlinepeeler import OnlinePeeler
from .onlinetraceviewer import OnlineTraceViewer
from .onlinetools import make_empty_catalogue, lighter_catalogue
from ..gui import CatalogueWindow
from ..gui.mainwindow import error_box_msg, apply_all_catalogue_steps
from ..gui.gui_params import fullchain_params, features_params_by_methods, cluster_params_by_methods, peak_detector_params
from ..gui.tools import ParamDialog, MethodDialog, get_dict_from_group_param
from ..signalpreprocessor import estimate_medians_mads_after_preprocesing


"""
TODO:
  * don't send full catalogue with serializer but with path to catalogue
  * label counter recording
  * widget overview

  
"""


class MainWindowNode(QT.QMainWindow, pyacq.Node):
    """Base class for Nodes that implement a QWidget user interface.
    """
    def __init__(self, **kargs):
        QT.QMainWindow.__init__(self)
        pyacq.Node.__init__(self, **kargs)
    
    def close(self):
        Node.close(self)
        QT.QMainWindow.close(self)

    def closeEvent(self,event):
        if self.running():
            self.stop()
        if not self.closed():
            pyacq.Node.close(self)
        event.accept()



class TdcOnlineWindow(MainWindowNode):
    """
    Online spike sorting widget for several channel group:
        1. It start with an empty catalogue with no nosie estimation (medians/mads)
        2. Do an auto scale with timer
        3. Estimate medians/mads with use control and start spike with no label (-10=unlabbeled)
        4. Start a catalogue constructor on user demand
        5. Change the catalogue of each peeler with new clusters.
        6. Catalogue can be refined with the CatalogueWindow and on make_catalogue the new catalogue is applyed 
            on Peeler.
    
    """
    _input_specs = {'signals': dict(streamtype='signals')}
    
    request_compute = QT.pyqtSignal()
    
    def __init__(self):
        MainWindowNode.__init__(self)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        
        self.docks = {}
        self.docks['main'] = QT.QDockWidget('overview')
        self.main_w = QT.QWidget()
        self.docks['main'].setWidget(self.main_w)
        self.addDockWidget(QT.TopDockWidgetArea, self.docks['main'])
        
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
        
        self.toolbar = QT.QToolBar(orientation=QT.Vertical)
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.RightToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))


        do_autoscale = QT.QAction('Auto scale', self, shortcut = "a" ) #, icon=QT.QIcon(":document-open.svg"))
        do_autoscale.triggered.connect(self.auto_scale_trace)
        #~ self.main_menu.addAction(do_autoscale)
        self.toolbar.addAction(do_autoscale)

        self.do_compute_median_mad = QT.QAction('Detec peak only', self) #, icon=QT.QIcon(":document-open.svg"))
        self.do_compute_median_mad.triggered.connect(self.compute_median_mad)
        #~ self.main_menu.addAction(do_autoscale)
        self.toolbar.addAction(self.do_compute_median_mad)

        self.do_start_rec = QT.QAction('Start new catalogue', self) #, icon=QT.QIcon(":document-open.svg"))
        self.do_start_rec.triggered.connect(self.start_new_recording)
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
    
    def _configure(self, channel_groups=[], chunksize=1024, workdir='',
                            outputstream_params={'protocol': 'tcp', 'interface':'127.0.0.1', 'transfermode':'plaindata'},
                            nodegroup_friends=None, 
                            ):
        
        self.channel_groups = channel_groups
        
        self.chunksize = chunksize
        self.workdir = workdir
        self.outputstream_params = outputstream_params
        self.nodegroup_friends = nodegroup_friends

        
        #~ self.median_estimation_duration = 1
        self.median_estimation_duration = 3.

        # prepare workdir
        if not os.path.exists(self.workdir):
             os.makedirs(self.workdir)
        
        self.raw_sigs_dir = os.path.join(self.workdir, 'raw_sigs')
        self.dataio_dir = os.path.join(self.workdir, 'tdc_online')
        
        self.dataio = DataIO(dirname=self.dataio_dir)
        print(self.dataio)
        print(self.dataio.datasource)
        
        


        self.signals_medians = {chan_grp:None for chan_grp in self.channel_groups}
        self.signals_mads = {chan_grp:None for chan_grp in self.channel_groups}
        
        
        print('self.dataio.datasource', self.dataio.datasource)
        # load exists catalogueconstructor
        self.catalogueconstructors = {}
        if self.dataio.datasource is None:
            # not raw sigs recorded
            pass
        else:
            for chan_grp in self.channel_groups:
                #~ print(chan_grp)
                cc = CatalogueConstructor(dataio=self.dataio, chan_grp=chan_grp)
                self.catalogueconstructors[chan_grp] = cc
                if cc.signals_medians is not None:
                    self.signals_medians[chan_grp] = cc.signals_medians
                    self.signals_mads[chan_grp] = cc.signals_mads
                #~ print(cc)
        
        
        # loads exists catalogue or make empty ones
        self.catalogues = {}
        for chan_grp, channel_group in self.channel_groups.items():
            catalogue = self.dataio.load_catalogue(chan_grp=chan_grp)
            if catalogue is None:
                # make an empty/fake catalogue because
                # the peeler need it anyway
                params = self.get_catalogue_params()
                params['peak_detector_params']['relative_threshold'] = np.inf
                catalogue = make_empty_catalogue(chan_grp=chan_grp,channel_indexes=channel_group['channels'],**params)
                print('empty catalogue for', chan_grp)
            self.catalogues[chan_grp] = catalogue

        # make trace viewer in tabs
        self.traceviewers ={}
        for chan_grp in self.channel_groups:
            traceviewer = OnlineTraceViewer()
            self.traceviewers[chan_grp] = traceviewer
            name = 'chan_grp{}'.format(chan_grp) # TODO better name when few channels
            self.docks[chan_grp] = QT.QDockWidget(name,self)
            self.docks[chan_grp].setWidget(traceviewer)
            self.tabifyDockWidget(self.docks['main'], self.docks[chan_grp])
        
        self.cataloguewindows = {}

        
        
    
    def after_input_connect(self, inputname):
        if inputname !='signals':
            return
        
        self.total_channel = self.input.params['shape'][1]
        self.sample_rate = self.input.params['sample_rate']
        
        channel_info = self.input.params.get('channel_info', None)
        if channel_info is None:
            all_channel_names = ['ch{}'.format(c) for c in range(self.total_channel)]
        else:
            all_channel_names = [ch_info['name'] for ch_info in channel_info]

        self.nb_channels = {}
        self.channel_names = {}
        for chan_grp, channel_group in self.channel_groups.items():
            channel_indexes = np.array(channel_group['channels'])
            assert np.all(channel_indexes<=self.total_channel), 'channel_indexes not compatible with total_channel'
            self.nb_channels[chan_grp] = len(channel_indexes)
            self.channel_names[chan_grp] = [ all_channel_names[c] for c in channel_indexes]
        
        # change default method depending the channel counts
        if 1<= max(self.nb_channels.values()) <9:
            feat_method = 'global_pca'
        else: 
            feat_method = 'peak_max'
        self.dialog_method_features.param_method['method'] = feat_method


    def _initialize(self, **kargs):
        
        
        self.rtpeelers = {}
        if self.nodegroup_friends is None:
            for chan_grp in self.channel_groups:
                self.rtpeelers[chan_grp] = OnlinePeeler()
        else:
            # len(self.nodegroup_friends) is not necessary len(channel_groups)
            # so we do a ring assignement

            for nodegroup_friend in self.nodegroup_friends:
                nodegroup_friend.register_node_type_from_module('tridesclous.online', 'OnlinePeeler')
            
            self.grp_nodegroup_friends = {}
            for i, chan_grp in enumerate(self.channel_groups):
                nodegroup_friend = self.nodegroup_friends[i%len(self.nodegroup_friends)]
                self.grp_nodegroup_friends[chan_grp] = nodegroup_friend
                self.rtpeelers[chan_grp] = nodegroup_friend.create_node('OnlinePeeler')
                
        
        
        # set a buffer on raw input for median/mad estimation
        buffer_size_margin = 3.
        self.input.set_buffer(size=int((self.median_estimation_duration+buffer_size_margin)*self.sample_rate),
                            double=True, axisorder=None, shmem=None, fill=None)
        self.thread_poll_input = ThreadPollInput(self.input)
        
        for chan_grp, channel_group in self.channel_groups.items():
            rtpeeler = self.rtpeelers[chan_grp]
            rtpeeler.configure(catalogue=self.catalogues[chan_grp], 
                    in_group_channels=channel_group['channels'], chunksize=self.chunksize)
            rtpeeler.input.connect(self.input.params)
            rtpeeler.outputs['signals'].configure(**self.outputstream_params)
            rtpeeler.outputs['spikes'].configure(**self.outputstream_params)
            rtpeeler.initialize()
        
            traceviewer = self.traceviewers[chan_grp]
            traceviewer.configure(peak_buffer_size=1000, catalogue=self.catalogues[chan_grp])
            traceviewer.inputs['signals'].connect(rtpeeler.outputs['signals'])
            traceviewer.inputs['spikes'].connect(rtpeeler.outputs['spikes'])
            traceviewer.initialize()
        

        # timer for autoscale (after new catalogue)
        self.timer_scale = QT.QTimer(singleShot=True, interval=500)
        self.timer_scale.timeout.connect(self.auto_scale_trace)
        # timer for median/mad
        self.timer_med = QT.QTimer(singleShot=True, interval=int(self.median_estimation_duration*1000)+1000)
        self.timer_med.timeout.connect(self.on_done_median_estimation_duration)
        # timer for catalogue
        self.timer_recording = QT.QTimer(singleShot=True)
        self.timer_recording.timeout.connect(self.on_recording_done)
        
        
        # stuf for recording a chunk for catalogue constructor
        self.rec = None
        self.worker_thread = QT.QThread(parent=self)
        self.worker = None
    
    def _start(self):
        for chan_grp in self.channel_groups:
            self.rtpeelers[chan_grp].start()
            self.traceviewers[chan_grp].start()
        
        self.thread_poll_input.start()
        self.worker_thread.start()

    def _stop(self):
        for chan_grp in self.channel_groups:
            self.rtpeelers[chan_grp].stop()
            self.traceviewers[chan_grp].stop()
        
        self.thread_poll_input.stop()
        self.thread_poll_input.wait()
        
        self.worker_thread.quit()
        self.worker_thread.wait()
        
    def _close(self):
        pass

    def get_catalogue_params(self):
        p = self.dialog_fullchain_params.get()
        p['preprocessor'].pop('chunksize')
        
        params = dict(
            n_left=p['extract_waveforms']['n_left'],
            n_right=p['extract_waveforms']['n_right'],
            
            internal_dtype='float32',
            
            preprocessor_params=p['preprocessor'],
            peak_detector_params=p['peak_detector'], #{'relative_threshold' : 8.},
            clean_waveforms_params=p['clean_waveforms'],
            
            signals_medians=None,
            signals_mads=None,
            
        )
        #~ if params['signals_medians'] is not None:
            #~ params['signals_medians']  = params['signals_medians'] .copy()
            #~ params['signals_mads']  = params['signals_mads'] .copy()
        
        #~ print(params)
        return params    
    
    def auto_scale_trace(self):
        # add factor in pyacq.oscilloscope autoscale (def compute_rescale)
        for chan_grp in self.channel_groups:
            self.traceviewers[chan_grp].auto_scale(spacing_factor=25.)
    
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
        
        for chan_grp, channel_group in self.channel_groups.items():
        
            self.signals_medians[chan_grp], self.signals_mads[chan_grp] = estimate_medians_mads_after_preprocesing(
                            sigs[:, channel_group['channels']], self.sample_rate,
                            preprocessor_params=self.get_catalogue_params()['preprocessor_params'])
            print(chan_grp, self.signals_medians[chan_grp], self.signals_mads[chan_grp])
        
            channel_indexes = np.array(channel_group['channels'])
            params = self.get_catalogue_params()
            params['signals_medians'] = self.signals_medians[chan_grp]
            params['signals_mads'] = self.signals_mads[chan_grp]
            catalogue = make_empty_catalogue(chan_grp=chan_grp,channel_indexes=channel_indexes,**params)
            #~ self.catalogues[chan_grp] = catalogue
            
            self.change_one_catalogue(catalogue)
    
    def change_one_catalogue(self, catalogue):
        chan_grp = catalogue['chan_grp']
        self.catalogues[chan_grp] = catalogue
        self.rtpeelers[chan_grp].change_catalogue(catalogue)
        self.traceviewers[chan_grp].change_catalogue(catalogue)
        #TODO function lighter_catalogue for traceviewers
        # TODO path for rtpeeler to avoid serilisation
        
        #~ self.traceviewer.change_catalogue(lighter_catalogue(self.catalogue))
        
        xsize = self.traceviewers[chan_grp].params['xsize']
        self.timer_scale.setInterval(int(xsize*1000.))
        self.timer_scale.start()

    def on_new_catalogue(self, chan_grp=None):
        print('on_new_catalogue', chan_grp)
        if chan_grp is None:
            return
        catalogue = self.dataio.load_catalogue(chan_grp=chan_grp)
        self.change_one_catalogue(catalogue)

    def start_new_recording(self):
        if self.timer_recording.isActive():
            return
        
        if self.rec is not None:
            return
        
        if os.path.exists(self.raw_sigs_dir):
            self.warn('Sigs and catalogue already exist.\nDelete it and start rec again.')
            return

        
        if not self.dialog_fullchain_params.exec_():
            return
        
        if not self.dialog_method_features.exec_():
            return

        if not self.dialog_method_cluster.exec_():
            return
        
        # get duration for raw sigs record
        fullchain_kargs = self.dialog_fullchain_params.get()
        self.timer_recording.setInterval(int((fullchain_kargs['duration']+1)*1000.))
        
        for chan_grp, w in self.cataloguewindows.items():
            w.close()
        self.cataloguewindows = {}
        self.catalogueconstructors = {}
        
        self.rec = RawRecorder()
        self.rec.configure(streams=[self.input.params], autoconnect=True, dirname=self.raw_sigs_dir)
        self.rec.initialize()
        self.rec.start()
        
        self.timer_recording.start()
        but = self.toolbar.widgetForAction(self.do_start_rec)
        but.setStyleSheet("QToolButton:!hover { background-color: red }")


        
    def on_recording_done(self):
        print('on_recording_done')
        self.rec.stop()
        self.rec.close()
        self.rec = None
        
        but = self.toolbar.widgetForAction(self.do_start_rec)
        but.setStyleSheet("")
        
        # create new dataio
        self.dataio = DataIO(dirname=self.dataio_dir)
        filenames = [os.path.join(self.raw_sigs_dir, 'input0.raw')]
        self.dataio.set_data_source(type='RawData', filenames=filenames, sample_rate=self.sample_rate, 
                    dtype=self.input.params['dtype'], total_channel=self.total_channel)
        self.dataio.set_channel_groups(self.channel_groups)
        print(self.dataio)

        # create new self.catalogueconstructors
        for chan_grp in self.channel_groups:
            cc = CatalogueConstructor(dataio=self.dataio, chan_grp=chan_grp)
            self.catalogueconstructors[chan_grp] = cc
            #~ if cc.signals_medians is not None:
                #~ self.signals_medians[chan_grp] = cc.signals_medians
                #~ self.signals_mads[chan_grp] = cc.signals_mads
            print(cc)

        #~ params = self.get_catalogue_params()
        fullchain_kargs = self.dialog_fullchain_params.get()
        fullchain_kargs['preprocessor']['chunksize'] = self.chunksize
        
        feat_method = self.dialog_method_features.param_method['method']
        feat_kargs = get_dict_from_group_param(self.dialog_method_features.all_params[feat_method], cascade=True)

        clust_method = self.dialog_method_cluster.param_method['method']
        clust_kargs = get_dict_from_group_param(self.dialog_method_cluster.all_params[clust_method], cascade=True)

        print('feat_method', feat_method, 'clust_method', clust_method)
        self.worker = Worker(self.catalogueconstructors, fullchain_kargs,
                feat_method, feat_kargs,
                clust_method, clust_kargs)
        
        self.worker.moveToThread(self.worker_thread)
        self.request_compute.connect(self.worker.compute)
        self.worker.done.connect(self.on_new_catalogue)
        self.worker.compute_catalogue_error.connect(self.on_compute_catalogue_error)
        self.request_compute.emit()
    
    def on_compute_catalogue_error(self, e):
        self.errorToMessageBox(e)
    
    def get_visible_tab(self):
        for chan_grp, traceviewer in self.traceviewers.items():
            if not traceviewer.visibleRegion().isEmpty():
                return chan_grp
    
    def open_cataloguewindow(self):
        chan_grp = self.get_visible_tab()
        print('open_cataloguewindow', chan_grp)

        if chan_grp is None:
            return
        
        if chan_grp not in self.catalogueconstructors:
            return
        
        if chan_grp not in self.cataloguewindows:
            self.cataloguewindows[chan_grp] = CatalogueWindow(self.catalogueconstructors[chan_grp])
            self.cataloguewindows[chan_grp].new_catalogue.connect(self.on_new_catalogue)
            
            name = self.docks[chan_grp].windowTitle()
            self.cataloguewindows[chan_grp].setWindowTitle(name)
            
        
        self.cataloguewindows[chan_grp].show()
    
    def delete_catalogue(self):
        # this delete catalogue and raw sigs
        for chan_grp, w in self.cataloguewindows.items():
            w.close()
        self.cataloguewindows = {}
        
        self.catalogueconstructors = {}
        
        if os.path.exists(self.dataio_dir):
            shutil.rmtree(self.dataio_dir)
        if os.path.exists(self.raw_sigs_dir):
            shutil.rmtree(self.raw_sigs_dir)
        
        self.dataio = DataIO(dirname=self.dataio_dir)
        
        # make empty catalogues
        for chan_grp, channel_group in self.channel_groups.items():
            channel_indexes = np.array(channel_group['channels'])
            params = self.get_catalogue_params()
            params['peak_detector_params']['relative_threshold'] = np.inf
            catalogue = make_empty_catalogue(chan_grp=chan_grp,channel_indexes=channel_indexes,**params)
            self.change_one_catalogue(catalogue)


class Worker(QT.QObject):
    done = QT.pyqtSignal(int)
    compute_catalogue_error = QT.pyqtSignal(object)
    def __init__(self, catalogueconstructors, fullchain_kargs, 
                feat_method, feat_kargs,
                clust_method, clust_kargs, parent=None):
        QT.QObject.__init__(self, parent=parent)
        
        self.catalogueconstructors = catalogueconstructors
        self.fullchain_kargs = fullchain_kargs
        self.feat_method = feat_method
        self.feat_kargs = feat_kargs
        self.clust_method = clust_method
        self.clust_kargs = clust_kargs
    
    def compute(self):
        print('compute')
        
        for chan_grp, catalogueconstructor in self.catalogueconstructors.items():
        
            print(catalogueconstructor)
            print('self.fullchain_kargs duration', self.fullchain_kargs['duration'])
            
            try:
            #~ if 1:
                apply_all_catalogue_steps(catalogueconstructor, self.fullchain_kargs, 
                        self.feat_method, self.feat_kargs, self.clust_method, self.clust_kargs,
                        verbose=False,
                        #~ verbose=True,
                        )
                
                catalogueconstructor.make_catalogue_for_peeler()
                print(catalogueconstructor)
                self.done.emit(chan_grp)
                
            except Exception as e:
                self.compute_catalogue_error.emit(e)
                return
            
