import os, sys
import time
import shutil
import datetime
from pprint import pprint

import numpy as np
from ..gui import QT
import pyqtgraph as pg

from pyqtgraph.util.mutex import Mutex

import pyacq
from pyacq.core import WidgetNode, InputStream, ThreadPollInput
#~ from pyacq.core.stream.arraytools import make_dtype
from pyacq.rec import RawRecorder

# internals import
from ..dataio import DataIO
from ..catalogueconstructor import CatalogueConstructor
from .. cataloguetools import apply_all_catalogue_steps
from ..signalpreprocessor import estimate_medians_mads_after_preprocesing

from ..gui import CatalogueWindow
from ..gui.mainwindow import error_box_msg
from ..gui.gui_params import fullchain_params, features_params_by_methods, cluster_params_by_methods, peak_detector_params
from ..gui.tools import ParamDialog, MethodDialog, get_dict_from_group_param
from ..autoparams import get_auto_params_for_catalogue

from .onlinepeeler import OnlinePeeler
from .onlinetraceviewer import OnlineTraceViewer
from .onlinetools import make_empty_catalogue, lighter_catalogue
from .onlinewaveformhistviewer import OnlineWaveformHistViewer





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
    
    This windows glue for several channel group:
       * peeler
       * tarveviewer
       * waveformhistorgram viewer
    
    
    Each peeler can be run in a separate NodeGroup (and so machine) if nodegroup_friends
    is provide. This is usefull to distribute the computation.
    
    Each catalogue can reset throuth this UI by clicking on "Rec for catalogue".
    Then a background recording is launch and at the end of duration chossen by user
    a new catalogue is recompute for each channel group. Each catalogue can cleaned with
    the CatalogueWindow.
    
    
    """
    _input_specs = {'signals': dict(streamtype='signals')}
    
    request_compute = QT.pyqtSignal()
    
    def __init__(self):
        MainWindowNode.__init__(self)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        
        self.docks = {}
        self.docks['overview'] = QT.QDockWidget('overview')
        self.overview = WidgetOverview(self)
        self.docks['overview'].setWidget(self.overview)
        self.addDockWidget(QT.TopDockWidgetArea, self.docks['overview'])
        
        self.create_actions_and_menu()
        
        self.dialog_fullchain_params = ParamDialog(fullchain_params, parent=self)
        #~ self.dialog_fullchain_params.params['duration'] = 10. # for debug
        #~ self.dialog_fullchain_params.params['peak_detector', 'relative_threshold'] = 8
        self.dialog_fullchain_params.resize(450, 600)
        
        
        self.dialog_method_features = MethodDialog(features_params_by_methods, parent=self,
                        title='Which feature method ?', selected_method='global_pca')
        self.dialog_method_cluster = MethodDialog(cluster_params_by_methods, parent=self,
                        title='Which cluster method ?', selected_method = 'pruningshears')
        
        
        
        self.mutex = Mutex()

    def create_actions_and_menu(self):
        #~ return
        
        self.toolbar = QT.QToolBar(orientation=QT.Vertical)
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.RightToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))


        do_autoscale = QT.QAction('Auto scale', self, shortcut = "a" ) #, icon=QT.QIcon(":document-open.svg"))
        do_autoscale.triggered.connect(self.auto_scale_trace)
        self.toolbar.addAction(do_autoscale)
        
        # NOT USEFULL
        #~ self.do_compute_median_mad = QT.QAction('Detec peak only', self) #, icon=QT.QIcon(":document-open.svg"))
        #~ self.do_compute_median_mad.triggered.connect(self.compute_median_mad)
        #~ self.toolbar.addAction(self.do_compute_median_mad)

        self.do_start_rec = QT.QAction('Rec for catalogue', self) #, icon=QT.QIcon(":document-open.svg"))
        self.do_start_rec.triggered.connect(self.start_new_recording)
        self.toolbar.addAction(self.do_start_rec)

        do_open_cataloguewin = QT.QAction('Edit catalogue', self) #, icon=QT.QIcon(":document-open.svg"))
        do_open_cataloguewin.triggered.connect(self.open_cataloguewindow)
        self.toolbar.addAction(do_open_cataloguewin)

        do_delete_catalogue = QT.QAction('Delete catalogues', self) #, icon=QT.QIcon(":document-open.svg"))
        do_delete_catalogue.triggered.connect(self.delete_catalogues)
        self.toolbar.addAction(do_delete_catalogue)
        
        do_show_waveform = QT.QAction('Show waveforms', self)
        do_show_waveform.triggered.connect(self.show_waveforms)
        self.toolbar.addAction(do_show_waveform)
        do_show_waveform.setCheckable(True)
        do_show_waveform.setChecked(False)


    def warn(self, text, title='Error in tridesclous'):
        mb = QT.QMessageBox.warning(self, title,text, 
                QT.QMessageBox.Ok ,
                QT.QMessageBox.NoButton)
    
    def errorToMessageBox(self, e):
        self.warn(error_box_msg.format(e))
    
    def _configure(self, channel_groups=[], chunksize=1024, workdir='',
                            outputstream_params={'protocol': 'tcp', 'interface':'127.0.0.1', 'transfermode':'plaindata'},
                            nodegroup_friends=None, 
                            peeler_params={},
                            initial_catalogue_params={},
                            ):
        
        self.sample_rate = None
        
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
        
        #if is this fail maybe the dir is old dir moved
        try:
            self.dataio = DataIO(dirname=self.dataio_dir)
        except:
            # the dataio_dir is corrupted
            dtime = '{:%Y-%m-%d %Hh%Mm%S}'.format(datetime.datetime.now())
            shutil.move(self.dataio_dir, self.dataio_dir+'_corrupted_'+dtime)
            self.dataio = DataIO(dirname=self.dataio_dir)
        
        self.signals_medians = {chan_grp:None for chan_grp in self.channel_groups}
        self.signals_mads = {chan_grp:None for chan_grp in self.channel_groups}
        
        # create geometry outside the dataio because
        # the datasource do not exists yet
        self.all_geometry = {}
        for chan_grp in self.channel_groups:
            channel_group = self.channel_groups[chan_grp]
            assert 'geometry' in channel_group
            geometry = [ channel_group['geometry'][chan] for chan in channel_group['channels'] ]
            geometry = np.array(geometry, dtype='float64')
            self.all_geometry[chan_grp] = geometry
        
        
        #~ print('self.dataio.datasource', self.dataio.datasource)
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
                    self.signals_medians[chan_grp] = np.array(cc.signals_medians, copy=True)
                    self.signals_mads[chan_grp] = np.array(cc.signals_mads, copy=True)
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
        self.waveformviewers = {}
        for chan_grp in self.channel_groups:
            traceviewer = OnlineTraceViewer()
            self.traceviewers[chan_grp] = traceviewer
            waveformviewer = OnlineWaveformHistViewer()
            self.waveformviewers[chan_grp] = waveformviewer
            widget = QT.QWidget()
            v = QT.QVBoxLayout()
            widget.setLayout(v)
            v.addWidget(traceviewer)
            v.addWidget(waveformviewer)
            waveformviewer.hide()
            
            name = 'chan_grp{}'.format(chan_grp) # TODO better name when few channels
            self.docks[chan_grp] = QT.QDockWidget(name,self)
            #~ self.docks[chan_grp].setWidget(traceviewer)
            self.docks[chan_grp].setWidget(widget)
            self.tabifyDockWidget(self.docks['overview'], self.docks[chan_grp])
        
        self.cataloguewindows = {}
        
        # TODO use autoparams
        
        # peeler params = 
        self.peeler_params = {
            'engine': 'geometrical',
        }
        self.peeler_params.update(peeler_params)
        
        if 'chunksize' in self.peeler_params:
            self.peeler_params.pop('chunksize')
        
        self.initial_catalogue_params = initial_catalogue_params
        

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
            


        # make auto params
        fisrt_chan_grp = list(self.channel_groups.keys())[0]
        n = len(self.channel_groups[fisrt_chan_grp]['channels'])
        params = get_auto_params_for_catalogue(dataio=None, chan_grp=fisrt_chan_grp,
                                        nb_chan=n, sample_rate=self.sample_rate, context='online')
        print('get_auto_params_for_catalogue in after_input_connect ')
        params = dict(params)
        
        for k,v  in self.initial_catalogue_params.items():
            # update default with initial_catalogue_params
            # nested at one level
            assert k in params , f'params "{k}" in initial_catalogue_params not handled'

            if isinstance(v, dict):
                params[k].update(v)
            else:
                params[k] = v
        
        d = dict(params)
        for k in ('feature_method', 'feature_kargs', 'cluster_method', 'cluster_kargs'):
            d.pop(k)
        
        self.dialog_fullchain_params.set(d)
        self.dialog_method_features.set_method(params['feature_method'], params['feature_kargs'])
        self.dialog_method_cluster.set_method(params['cluster_method'], params['cluster_kargs'])

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
        
        self.sigs_shmem_converters = {}
        self.spikes_shmem_converters = {}
        
        for chan_grp, channel_group in self.channel_groups.items():
            
            
            rtpeeler = self.rtpeelers[chan_grp]
            rtpeeler.configure(catalogue=self.catalogues[chan_grp], 
                    in_group_channels=channel_group['channels'], chunksize=self.chunksize,
                    geometry=self.all_geometry[chan_grp], **self.peeler_params )
            rtpeeler.input.connect(self.input.params)
            rtpeeler.outputs['signals'].configure(**self.outputstream_params)
            rtpeeler.outputs['spikes'].configure(**self.outputstream_params)
            rtpeeler.initialize()
            
            # TODO cleaner !!!!!
            peak_buffer_size = 100000
            sig_buffer_size = int(15.*self.sample_rate)
            
            if self.outputstream_params['transfermode'] != 'sharedmem':
                print('create stream converter !!!!! no sharedmem')

                #if outputstream_params is plaindata (because distant)
                # create 2 StreamConverter to convert to sharedmeme so that
                #  input buffer will shared between traceviewer and waveformviewer
                sigs_shmem_converter = pyacq.StreamConverter()
                sigs_shmem_converter.configure()
                sigs_shmem_converter.input.connect(rtpeeler.outputs['signals'])
                stream_params = dict(transfermode='sharedmem', buffer_size=sig_buffer_size, double=True)
                for k in ('dtype', 'shape', 'sample_rate', 'channel_info', ):
                    param = rtpeeler.outputs['signals'].params.get(k, None)
                    if param is not None:
                        stream_params[k] = param
                sigs_shmem_converter.output.configure(**stream_params)
                sigs_shmem_converter.initialize()
                
                spikes_shmem_converter = pyacq.StreamConverter()
                spikes_shmem_converter.configure()
                spikes_shmem_converter.input.connect(rtpeeler.outputs['spikes'])
                stream_params = dict(transfermode='sharedmem', buffer_size=peak_buffer_size, double=False)
                for k in ('dtype', 'shape',):
                    param = rtpeeler.outputs['spikes'].params.get(k, None)
                    if param is not None:
                        stream_params[k] = param
                spikes_shmem_converter.output.configure(**stream_params)
                spikes_shmem_converter.initialize()
                
            else:
                sigs_shmem_converter = None
                spikes_shmem_converter = None
            
            self.sigs_shmem_converters[chan_grp] = sigs_shmem_converter
            self.spikes_shmem_converters[chan_grp] = spikes_shmem_converter
            
            traceviewer = self.traceviewers[chan_grp]
            traceviewer.configure(peak_buffer_size=peak_buffer_size, catalogue=self.catalogues[chan_grp])
            if sigs_shmem_converter is None:
                traceviewer.inputs['signals'].connect(rtpeeler.outputs['signals'])
                traceviewer.inputs['spikes'].connect(rtpeeler.outputs['spikes'])
            else:
                traceviewer.inputs['signals'].connect(sigs_shmem_converter.output)
                traceviewer.inputs['spikes'].connect(spikes_shmem_converter.output)
            traceviewer.initialize()

            waveformviewer = self.waveformviewers[chan_grp]
            waveformviewer.configure(peak_buffer_size=peak_buffer_size, catalogue=self.catalogues[chan_grp])
            if spikes_shmem_converter is None:
                waveformviewer.inputs['signals'].connect(rtpeeler.outputs['signals'])
                waveformviewer.inputs['spikes'].connect(rtpeeler.outputs['spikes'])
            else:
                waveformviewer.inputs['signals'].connect(sigs_shmem_converter.output)
                waveformviewer.inputs['spikes'].connect(spikes_shmem_converter.output)
            waveformviewer.initialize()
        

        # timer for autoscale (after new catalogue)
        self.timer_scale = QT.QTimer(singleShot=True, interval=500)
        self.timer_scale.timeout.connect(self.auto_scale_trace)
        # timer for median/mad
        #~ self.timer_med = QT.QTimer(singleShot=True, interval=int(self.median_estimation_duration*1000)+1000)
        #~ self.timer_med.timeout.connect(self.on_done_median_estimation_duration)
        # timer for catalogue
        self.timer_recording = QT.QTimer(singleShot=True)
        self.timer_recording.timeout.connect(self.on_recording_done)
        self.timer_recording_refresh = QT.QTimer(singleShot=False, interval=1000)
        self.timer_recording_refresh.timeout.connect(self.resfresh_rec_label)
        
        
        # stuf for recording a chunk for catalogue constructor
        self.rec = None
        self.worker_thread = QT.QThread(parent=self)
        self.worker = None
        
        self.overview.refresh()
    
    def _start(self):
        for chan_grp in self.channel_groups:
            self.rtpeelers[chan_grp].start()
            if self.sigs_shmem_converters[chan_grp] is not None:
                self.sigs_shmem_converters[chan_grp].start()
            if self.spikes_shmem_converters[chan_grp] is not None:
                self.spikes_shmem_converters[chan_grp].start()
            self.traceviewers[chan_grp].start()
            #~ self.waveformviewers[chan_grp].start()
        
        self.thread_poll_input.start()
        self.worker_thread.start()

    def _stop(self):
        for chan_grp in self.channel_groups:
            self.rtpeelers[chan_grp].stop()
            self.traceviewers[chan_grp].stop()
            if self.sigs_shmem_converters[chan_grp] is not None:
                self.sigs_shmem_converters[chan_grp].stop()
            if self.spikes_shmem_converters[chan_grp] is not None:
                self.spikes_shmem_converters[chan_grp].stop()
            if self.waveformviewers[chan_grp].running():
                self.waveformviewers[chan_grp].stop()
        
        self.thread_poll_input.stop()
        self.thread_poll_input.wait()
        
        self.worker_thread.quit()
        self.worker_thread.wait()
        
    def _close(self):
        pass

    def get_catalogue_params(self):
        p = self.dialog_fullchain_params.get()
        #~ p['preprocessor'].pop('chunksize')
        
        if self.sample_rate is None:
            # before input connect need to make fake catalogue
            n_left, n_right = -20, 40
        else:
            n_left=int(p['extract_waveforms']['wf_left_ms'] / 1000. * self.sample_rate)
            n_right=int(p['extract_waveforms']['wf_right_ms'] / 1000. * self.sample_rate)
        
        params = dict(
            n_left=n_left,
            n_right=n_right,
            
            internal_dtype='float32',
            
            preprocessor_params=p['preprocessor'],
            peak_detector_params=p['peak_detector'], #{'relative_threshold' : 8.},
            clean_peaks_params=p['clean_peaks'],
            
            signals_medians=None,
            signals_mads=None,
            
        )
        
        return params    

    def channel_group_label(self, chan_grp=0):
        label = 'chan_grp {} - '.format(chan_grp)
        channel_indexes = self.channel_groups[chan_grp]['channels']
        ch_names = np.array(self.channel_names[chan_grp])[channel_indexes]
        if len(ch_names)<8:
            label += ' '.join(ch_names)
        else:
            label += ' '.join(ch_names[:3]) + ' ... ' + ' '.join(ch_names[-2:])
        return label


    def auto_scale_trace(self):
        for chan_grp in self.channel_groups:
            self.traceviewers[chan_grp].auto_scale(spacing_factor=25.)
            if self.waveformviewers[chan_grp].running():
                self.waveformviewers[chan_grp].auto_scale()
    
    #~ def compute_median_mad(self):
        #~ """
        #~ Wait for a while until input buffer is long anought to estimate the medians/mads
        #~ """
        #~ if self.timer_med.isActive():
            #~ return
        
        #~ if not self.dialog_fullchain_params.exec_():
            #~ return

        #~ self.timer_med.start()
        
        #~ but = self.toolbar.widgetForAction(self.do_compute_median_mad)
        #~ but.setStyleSheet("QToolButton:!hover { background-color: red }")


    #~ def on_done_median_estimation_duration(self):
        
        #~ but = self.toolbar.widgetForAction(self.do_compute_median_mad)
        #~ but.setStyleSheet("")
        
        #~ head = self.thread_poll_input.pos()
        #~ length = int((self.median_estimation_duration)*self.sample_rate)
        #~ sigs = self.input.get_data(head-length, head, copy=False, join=True)
        
        #~ for chan_grp, channel_group in self.channel_groups.items():
            #~ self.signals_medians[chan_grp], self.signals_mads[chan_grp] = estimate_medians_mads_after_preprocesing(
                            #~ sigs[:, channel_group['channels']], self.sample_rate,
                            #~ preprocessor_params=self.get_catalogue_params()['preprocessor_params'])
            #~ channel_indexes = np.array(channel_group['channels'])
            #~ params = self.get_catalogue_params()
            #~ params['signals_medians'] = self.signals_medians[chan_grp]
            #~ params['signals_mads'] = self.signals_mads[chan_grp]
            #~ catalogue = make_empty_catalogue(chan_grp=chan_grp,channel_indexes=channel_indexes,**params)
            #~ self.change_one_catalogue(catalogue)
    
    def change_one_catalogue(self, catalogue):
        chan_grp = catalogue['chan_grp']
        self.catalogues[chan_grp] = catalogue
        self.rtpeelers[chan_grp].change_catalogue(catalogue)
        self.traceviewers[chan_grp].change_catalogue(catalogue)
        self.waveformviewers[chan_grp].change_catalogue(catalogue)
        
        #TODO function lighter_catalogue for traceviewers
        # TODO path for rtpeeler to avoid serilisation
        
        #~ self.traceviewer.change_catalogue(lighter_catalogue(self.catalogue))
        
        xsize = self.traceviewers[chan_grp].params['xsize']
        self.timer_scale.setInterval(int(xsize*1000.))
        self.timer_scale.start()
        
        self.overview.refresh()

    def on_new_catalogue(self, chan_grp=None):
        #~ print('on_new_catalogue', chan_grp)
        if chan_grp is None:
            return
        
        catalogue = self.dataio.load_catalogue(chan_grp=chan_grp)
        self.change_one_catalogue(catalogue)

        self.overview.refresh()

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
        p = self.dialog_fullchain_params.get()
        self.timer_recording.setInterval(int((p['duration']+1)*1000.))
        
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
        
        
        self.time_start_rec = time.perf_counter()
        self.timer_recording_refresh.start()
        
        self.overview.refresh()
        
        
    def resfresh_rec_label(self):
        dur = self.dialog_fullchain_params.get()['duration']
        now = time.perf_counter()
        remain = int(dur - (now - self.time_start_rec))
        self.do_start_rec.setText('Rec for catalogue {}s'.format(remain))
        
        
    def on_recording_done(self):
        #~ print('on_recording_done')
        self.rec.stop()
        self.rec.close()
        self.rec = None
        
        self.timer_recording_refresh.stop()
        self.do_start_rec.setText('Rec for catalogue')
        but = self.toolbar.widgetForAction(self.do_start_rec)
        but.setStyleSheet("")
        
        # create new dataio
        self.dataio = DataIO(dirname=self.dataio_dir)
        filenames = [os.path.join(self.raw_sigs_dir, 'input0.raw')]
        self.dataio.set_data_source(type='RawData', filenames=filenames, sample_rate=self.sample_rate, 
                    dtype=self.input.params['dtype'], total_channel=self.total_channel)
        self.dataio.set_channel_groups(self.channel_groups)

        # create new self.catalogueconstructors
        for chan_grp in self.channel_groups:
            cc = CatalogueConstructor(dataio=self.dataio, chan_grp=chan_grp)
            self.catalogueconstructors[chan_grp] = cc
        
        fisrt_chan_grp = list(self.channel_groups.keys())[0]
        
        params = get_auto_params_for_catalogue(self.dataio, chan_grp=fisrt_chan_grp)
        params.update(self.dialog_fullchain_params.get())
        params['chunksize'] = self.chunksize
        params['memory_mode'] = 'memmap'
        
        
        params['feature_method'] = self.dialog_method_features.param_method['method']
        params['feature_kargs'] = get_dict_from_group_param(self.dialog_method_features.all_params[params['feature_method']], cascade=True)

        params['cluster_method'] = self.dialog_method_cluster.param_method['method']
        params['cluster_kargs'] = get_dict_from_group_param(self.dialog_method_cluster.all_params[params['cluster_method']], cascade=True)
        
        # TODO here params for make catalogue
        
        #~ params['clean_cluster'] = False
        #~ params['clean_cluster_kargs'] = {}
        
        

        self.worker = Worker(self.catalogueconstructors, params)
        
        self.worker.moveToThread(self.worker_thread)
        self.request_compute.connect(self.worker.compute)
        self.worker.done.connect(self.on_new_catalogue)
        self.worker.compute_catalogue_error.connect(self.on_compute_catalogue_error)
        self.request_compute.emit()
        
        self.overview.refresh()
    
    def on_compute_catalogue_error(self, e):
        self.errorToMessageBox(e)
    
    def get_visible_tab(self):
        for chan_grp, traceviewer in self.traceviewers.items():
            if not traceviewer.visibleRegion().isEmpty():
                return chan_grp
    
    def open_cataloguewindow(self, chan_grp=None):
        print('open_cataloguewindow', chan_grp)
        if chan_grp is None:
            chan_grp = self.get_visible_tab()
            
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
    
    def delete_catalogues(self):
        print('delete_catalogues')
        # this delete catalogue and raw sigs
        for chan_grp, w in self.cataloguewindows.items():
            w.close()
        self.cataloguewindows = {}
        
        if sys.platform.startswith('win'):
            # this is an unsane hack to detach all np.memmap
            # arrays attached to dataio and all catalogueconstructor
            # because under window the GC do no doit properly
            # otherwise rmtree will fail
            from ..catalogueconstructor import _persitent_arrays
            for chan_grp, cc in  self.catalogueconstructors.items():
                for name in _persitent_arrays:
                    cc.arrays.detach_array(name, mmap_close=True)
                for name in ['processed_signals', 'spikes']:
                    self.dataio.arrays[chan_grp][0].detach_array(name, mmap_close=True)
            for a in self.dataio.datasource.array_sources :
                a._mmap.close()
        

        self.catalogueconstructors = {}
        self.dataio = None
        
        if os.path.exists(self.dataio_dir):
            shutil.rmtree(self.dataio_dir)
        if os.path.exists(self.raw_sigs_dir):
            shutil.rmtree(self.raw_sigs_dir)
        
        self.dataio = DataIO(dirname=self.dataio_dir)
        
        # make empty catalogues
        for chan_grp, channel_group in self.channel_groups.items():
            channel_indexes = np.array(channel_group['channels'])
            params = self.get_catalogue_params()
            params['peak_detector_params']['relative_threshold'] = 10000000.0 # np.inf do not work with opencl
            catalogue = make_empty_catalogue(chan_grp=chan_grp,channel_indexes=channel_indexes,**params)
            self.change_one_catalogue(catalogue)
        
        self.overview.refresh()

    def show_waveforms(self, value):
        if value:
            for chan_grp in self.channel_groups:
                self.waveformviewers[chan_grp].show()
                self.waveformviewers[chan_grp].start()
        else:
            for chan_grp in self.channel_groups:
                self.waveformviewers[chan_grp].hide()
                self.waveformviewers[chan_grp].stop()



class WidgetOverview(QT.QWidget):
    def __init__(self, mainwindow, parent=None):
        QT.QWidget.__init__(self, parent)
        self.mainwindow = mainwindow

        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.main_label = QT.QLabel('')
        self.layout.addWidget(self.main_label)
        
        self.grid = QT.QGridLayout()
        self.layout.addLayout(self.grid)
        # the len of channel_groups is not knwon at __init__
        # but when tdcOnlienWIndow get configured and intialized
        # so it is postponed at first refresh
        self.grid_done = False
        
        self.layout.addStretch()
    
    @property
    def dataio(self):
        return self.mainwindow.dataio
    
    @property
    def channel_groups(self):
        return self.mainwindow.channel_groups

    @property
    def catalogues(self):
        return self.mainwindow.catalogues
    
    def make_grid(self):
        
        #~ n = len(self.channel_groups)
        self.chan_grp_labels = {}
        for r, chan_grp in enumerate(self.channel_groups):
            self.grid.addWidget(QT.QLabel('<b>chan_grp {}</b>'.format(chan_grp)), r, 0)
            self.chan_grp_labels[chan_grp] = label = QT.QLabel('')
            self.grid.addWidget(label, r, 1)
            but = QT.QPushButton('Edit')
            but.chan_grp = chan_grp
            but.clicked.connect(self.edit_catalogue)
            but.setMaximumWidth(30)
            self.grid.addWidget(but, r, 2)
            
            
        self.grid_done = True
    
    def refresh(self):
        if not self.grid_done:
            self.make_grid()
        
        txt = ''
        
        if self.dataio is None or self.dataio.datasource is None:
            txt += 'No signal recorded yet\n'
        else:
            #~ dur = self.dataio.get_segment_length(0)/self.dataio.sample_rate
            #~ txt += 'Signal duration for catalogue: {}s\n'.format(dur)
            txt += self.dataio.__repr__()
        self.main_label.setText(txt)
        
        for chan_grp in self.channel_groups:
            label = self.chan_grp_labels[chan_grp]
            txt = ''
            
            ch_names = self.mainwindow.channel_names[chan_grp]
            txt += ' '.join(ch_names) + '\n'
            
            cat = self.catalogues[chan_grp]
            if 'empty_catalogue' in cat:
                txt += 'No catalogue yet'
            else:
                txt += 'Nb cluster: {}\n'.format(cat['centers0'].shape[0])
            
            label.setText(txt)
    
    def edit_catalogue(self):
        chan_grp = self.sender().chan_grp
        self.mainwindow.open_cataloguewindow(chan_grp=chan_grp)
        



class Worker(QT.QObject):
    done = QT.pyqtSignal(int)
    compute_catalogue_error = QT.pyqtSignal(object)
    def __init__(self, catalogueconstructors, params, parent=None):
        QT.QObject.__init__(self, parent=parent)
        
        self.catalogueconstructors = catalogueconstructors
        self.params = params
    
    def compute(self):
        #~ print('compute')
        
        for chan_grp, cc in self.catalogueconstructors.items():
            print('Compute catalogue for chan_grp:', chan_grp)
            #~ print(catalogueconstructor)
            print('self.params duration', self.params['duration'])
            #~ try:
            if 1:
                cc.apply_all_steps(self.params,  verbose=True)
                #~ catalogueconstructor.make_catalogue_for_peeler()
                print(cc)
                self.done.emit(chan_grp)
                
            #~ except Exception as e:
                #~ self.compute_catalogue_error.emit(e)
        
        # release cataloguecontrustors otherwise they cannot be deleted
        self.catalogueconstructors = {}
        
            
