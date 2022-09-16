from .myqt import QT
import pyqtgraph as pg

import time
import os
from collections import OrderedDict
import pickle
import webbrowser
from pprint import pprint

from ..dataio import DataIO
from ..datasource import data_source_classes
from .tools import get_dict_from_group_param, ParamDialog, MethodDialog #, open_dialog_methods
from  ..datasets import datasets_info, download_dataset

from ..catalogueconstructor import CatalogueConstructor
from ..cataloguetools import apply_all_catalogue_steps
from ..autoparams import get_auto_params_for_catalogue, get_auto_params_for_peelers
from .cataloguewindow import CatalogueWindow
from ..peeler import Peeler
from .peelerwindow import PeelerWindow
from .initializedatasetwindow import InitializeDatasetWindow
from .probegeometryview import ProbeGeometryView
from ..export import export_list
from ..tools import open_prb
from ..report import generate_report
from ..cltools import HAVE_PYOPENCL
from .gpuselector import GpuSelector

from . import icons

from . import gui_params

try:
    import ephyviewer
    HAVE_EPHYVIEWER = True
except:
    HAVE_EPHYVIEWER = False


error_box_msg = """
This is the raw Python error.
Unfortunatly, this is sometimes not so usefull for user...
Please send it to https://github.com/tridesclous/tridesclous/issues.

{}
"""
class MainWindow(QT.QMainWindow):
    def __init__(self, parent=None):
        QT.QMainWindow.__init__(self, parent=parent)
        
        self.setWindowIcon(QT.QIcon(':/main_icon.png'))
        
        self.dataio = None
        #~ self.catalogueconstructor = None
        
        self.resize(800, 800)

        appname = 'tridesclous'
        settings_name = 'settings'
        self.settings = QT.QSettings(appname, settings_name)
        
        self.create_actions_and_menu()
        
        w = QT.QWidget()
        self.setCentralWidget(w)
        mainlayout  = QT.QVBoxLayout()
        w.setLayout(mainlayout)
        
        self.label_info = QT.QLabel('Nothing loaded')
        mainlayout.addWidget(self.label_info)
        
        mainlayout.addStretch()
        
        self.open_windows = []
        
        self.win_viewer = None
        self.probe_viewer = None
        
        self.dialog_fullchain_params = ParamDialog(gui_params.fullchain_params, parent=self)
        self.dialog_method_features = MethodDialog(gui_params.features_params_by_methods, parent=self,
                        title='Which feature method ?', selected_method='peak_max')
        self.dialog_method_cluster = MethodDialog(gui_params.cluster_params_by_methods, parent=self,
                        title='Which cluster method ?', selected_method = 'sawchaincut')
        
        self.dialog_method_peeler = MethodDialog(gui_params.peeler_params_by_methods, parent=self,
                    title='Peeler engine')
        
        if HAVE_PYOPENCL:
            self.dialog_gpuselector = GpuSelector(settings=self.settings)
    

    def create_actions_and_menu(self):
        
        self.toolbar = QT.QToolBar(orientation=QT.Vertical)
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.TopToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))

        self.toolbar2 = QT.QToolBar(orientation=QT.Vertical)
        self.toolbar2.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.LeftToolBarArea, self.toolbar2)
        self.toolbar2.setIconSize(QT.QSize(60, 40))
        
        
        self.file_menu = self.menuBar().addMenu(self.tr("File"))
        
        do_open = QT.QAction('&Open', self, shortcut = "Ctrl+O", icon=QT.QIcon(":document-open.svg"))
        do_open.triggered.connect(self.open_dialog)
        self.file_menu.addAction(do_open)
        self.toolbar.addAction(do_open)

        do_init = QT.QAction('&Initialize dataset', self, shortcut = "Ctrl+I", icon=QT.QIcon(":document-new.svg"))
        do_init.triggered.connect(self.initialize_dataset_dialog)
        self.file_menu.addAction(do_init)
        self.toolbar.addAction(do_init)
        
        self.recetly_opened_menu = self.file_menu.addMenu('Recently opened')
        self._refresh_recetly_opened()
        
        self.dw_dataset_menu = self.file_menu.addMenu('Download test datasets')
        for name in datasets_info:
            act = self.dw_dataset_menu.addAction(name)
            act.name = name
            act.triggered.connect(self.do_download_dataset)        
        
        
        self.toolbar.addSeparator()
        
        if HAVE_EPHYVIEWER:
            open_viewer = QT.QAction('&Preview raw signals', self, icon=QT.QIcon(":ephyviewer.png"))
            open_viewer.triggered.connect(self.open_ephyviewer)
            self.toolbar.addAction(open_viewer)
        
        open_probe_viewer = QT.QAction('&Probe geometry', self, icon=QT.QIcon(":probe-geometry.svg"))
        open_probe_viewer.triggered.connect(self.open_probe_viewer)
        self.toolbar.addAction(open_probe_viewer)
        
        self.toolbar.addSeparator()
        
        do_check_prb = QT.QAction('Check PRB file', self, ) # icon=QT.QIcon(":document-export.svg"))
        do_check_prb.triggered.connect(self.check_prb_file)
        self.file_menu.addAction(do_check_prb)
        
        if HAVE_PYOPENCL:
            do_open_gpuselector = QT.QAction('Default OpenCL (GPU) selector', self)
            do_open_gpuselector.triggered.connect(self.open_gpuselector)
            self.file_menu.addAction(do_open_gpuselector)
        
        self.toolbar.addSeparator()

        do_refresh = QT.QAction(u'Refresh', self,checkable = False, icon=QT.QIcon(":/view-refresh.svg"))
        do_refresh.triggered.connect(self.refresh_with_reload)
        self.toolbar.addAction(do_refresh)
        
        do_export_spikes = QT.QAction('Export spikes', self,  icon=QT.QIcon(":document-export.svg"))
        do_export_spikes.triggered.connect(self.export_spikes)
        self.toolbar.addAction(do_export_spikes)
        self.file_menu.addAction(do_export_spikes)

        do_generate_report = QT.QAction('Generate report', self,  icon=QT.QIcon(":kwrite.svg"))
        do_generate_report.triggered.connect(self.generate_report)
        self.toolbar.addAction(do_generate_report)
        self.file_menu.addAction(do_generate_report)

        self.toolbar.addSeparator()

        help_act = QT.QAction('Help', self,checkable = False, icon=QT.QIcon(":main_icon.png"))
        help_act.triggered.connect(self.open_webbrowser_help)
        self.toolbar.addAction(help_act)
        
        
        # Toolbar2
        self.toolbar2.addWidget(QT.QLabel('<b>Steps</b>'))
        
        self.toolbar2.addWidget(QT.QLabel('Select chanel group:'))
        self.combo_chan_grp = QT.QComboBox()
        self.toolbar2.addWidget(self.combo_chan_grp)
        self.combo_chan_grp.currentIndexChanged .connect(self.on_chan_grp_change)

        do_init_cataloguewin = QT.QAction('Initialize Catalogue', self, icon=QT.QIcon(":autocorrection.svg"))
        do_init_cataloguewin.triggered.connect(self.initialize_catalogue)
        self.toolbar2.addAction(do_init_cataloguewin)
        
        do_open_cataloguewin = QT.QAction('Open CatalogueWindow', self,  icon=QT.QIcon(":catalogwinodw.png"))
        do_open_cataloguewin.triggered.connect(self.open_cataloguewin)
        self.toolbar2.addAction(do_open_cataloguewin)


        do_run_peeler = QT.QAction('Run Peeler', self,  icon=QT.QIcon(":configure-shortcuts.svg"))
        do_run_peeler.triggered.connect(self.run_peeler)
        self.toolbar2.addAction(do_run_peeler)
        
        do_open_peelerwin = QT.QAction('Open PeelerWindow', self,  icon=QT.QIcon(":peelerwindow.png"))
        do_open_peelerwin.triggered.connect(self.open_peelerwin)
        self.toolbar2.addAction(do_open_peelerwin)

        
        
        
    def open_webbrowser_help(self):
        url = "http://tridesclous.readthedocs.io"
        webbrowser.open(url, new=2)
    
    def warn(self, text, title='Error in tridesclous'):
        mb = QT.QMessageBox.warning(self, title,text, 
                QT.QMessageBox.Ok ,
                QT.QMessageBox.NoButton)
    
    def errorToMessageBox(self, e):
        self.warn(error_box_msg.format(e))

    def refresh_info(self):
        txt = self.dataio.__repr__()
        txt += '\n\n'
        #~ print('refresh_info', self.chan_grps)
        for chan_grp in self.chan_grps:
            catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=chan_grp)
            txt += catalogueconstructor.__repr__()
            txt += '\n'
        
        self.label_info.setText(txt)
    
    def open_dialog(self):
        fd = QT.QFileDialog(fileMode=QT.QFileDialog.DirectoryOnly, acceptMode=QT.QFileDialog.AcceptOpen)
        #~ fd.setNameFilters(['Hearingloss setup (*.json)', 'All (*)'])
        fd.setViewMode( QT.QFileDialog.Detail )
        if fd.exec_():
            dirname = fd.selectedFiles()[0]
            print(dirname)
            
            if DataIO.check_initialized(dirname):
                self._open_dataio(dirname)
    
    def _refresh_recetly_opened(self):
        self.recetly_opened_menu.clear()
        for dirname in self.recently_opened():
            act = self.recetly_opened_menu.addAction(str(dirname))
            act.dirname = dirname
            act.triggered.connect(self.do_open_recent)
            
    def do_open_recent(self):
        self._open_dataio(self.sender().dirname)
    
    def recently_opened(self):
        value = self.settings.value('recently_opened')
        if value is None:
            recently_opened = []
        else:
            try:
                if type(value) == str:
                    #bug on some qt
                    value = value.encode('ascii')
                recently_opened = pickle.loads(value)
                if type(recently_opened) != list:
                    recently_opened = []
            except:
                recently_opened = []
        return recently_opened
    
    def do_download_dataset(self):
        name = self.sender().name
        fd = QT.QFileDialog(fileMode=QT.QFileDialog.DirectoryOnly, acceptMode=QT.QFileDialog.AcceptOpen)
        fd.setViewMode( QT.QFileDialog.Detail )
        if fd.exec_():
            localdir = os.path.join(fd.selectedFiles()[0], name)
            if os.path.exists(localdir):
                return
            os.mkdir(localdir)
            localdir, filenames, params = download_dataset(name, localdir=localdir)
            
            dirname = os.path.join(localdir, 'tdc_'+name)
            dataio = DataIO(dirname=dirname)
            dataio.set_data_source(type='RawData', filenames=filenames, **params)
            
            self._open_dataio( dirname)
    
    
    def refresh_with_reload(self):
        if self.dataio is None:
            return
        
        self.dataio = DataIO(dirname=self.dataio.dirname)
        self.refresh_info()
    
    def _open_dataio(self, dirname):
        recently_opened =self.recently_opened()
        if dirname not in recently_opened:
            recently_opened = [dirname] + recently_opened
            recently_opened = recently_opened[:5]
            self.settings.setValue('recently_opened', pickle.dumps(recently_opened))
            self._refresh_recetly_opened()
        
        for win in self.open_windows:
            win.close()
        self.open_windows = []
        
        self.dataio = DataIO(dirname=dirname)
        
        self.combo_chan_grp.blockSignals(True)
        self.combo_chan_grp.clear()
        self.combo_chan_grp.addItems([str(k) for k in self.dataio.channel_groups.keys()] + ['ALL'])
        self.combo_chan_grp.blockSignals(False)
        self.on_chan_grp_change()
        
        # set auto params catalogue
        params = get_auto_params_for_catalogue(self.dataio, chan_grp=self.chan_grps[0])
        d = dict(params)
        for k in ('feature_method', 'feature_kargs', 'cluster_method', 'cluster_kargs'):
            d.pop(k)
        self.dialog_fullchain_params.set(d)
        self.dialog_method_features.set_method(params['feature_method'], params['feature_kargs'])
        self.dialog_method_cluster.set_method(params['cluster_method'], params['cluster_kargs'])
        
        # TODO clean_cluster
        
        # set auto params peeler
        params = get_auto_params_for_peelers(self.dataio, chan_grp=self.chan_grps[0])
        d = dict(params)
        engine = d.pop('engine')
        self.dialog_method_peeler.set_method(engine, d)
        
        

    
    def release_closed_windows(self):
        for win in list(self.open_windows):
            if not win.isVisible():
                self.open_windows.remove(win)
    
    def close_window_chan_grp(self, chan_grp):
        for win in list(self.open_windows):
            if win.controller.chan_grp == chan_grp:
                win.close()
                self.open_windows.remove(win)
        

    #~ @property
    #~ def chan_grp(self):
        #~ txt = self.combo_chan_grp.currentText()
        #~ if txt == 'ALL':
            #~ # take the first
            #~ chan_grp = list(self.dataio.channel_groups.keys())[0]
        #~ else:
            #~ chan_grp = int(txt)
        #~ return chan_grp
    
    @property
    def chan_grps(self):
        txt = self.combo_chan_grp.currentText()
        if txt == 'ALL':
            # take the first
            chan_grps = list(self.dataio.channel_groups.keys())
        else:
            chan_grps = [int(txt)]
        return chan_grps
    
    def on_chan_grp_change(self, index=None):
        #~ self.catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=self.chan_grp)
        self.refresh_info()
        
        # this set a the a by default method depending the number of channels
        n = self.dataio.nb_channel(chan_grp=self.chan_grps[0])
        if 1<=n<9:
            feat_method = 'global_pca'
        else:
            feat_method = 'peak_max'
        self.dialog_method_features.param_method['method'] = feat_method
    
    def initialize_dataset_dialog(self):
        init_dia = InitializeDatasetWindow(parent=self)
        if init_dia.exec_():
            self._open_dataio(init_dia.dirname_created)


    
    def initialize_catalogue(self):
        
        if  self.dataio is None:
            return
        
        self.release_closed_windows()
        for chan_grp in self.chan_grps:
            self.close_window_chan_grp(chan_grp)
        
        self.refresh_with_reload()
        
        if not self.dialog_fullchain_params.exec_():
            return
        if not self.dialog_method_features.exec_():
            return
        if not self.dialog_method_cluster.exec_():
            return
        
        params = {}
        params.update(self.dialog_fullchain_params.get())
        params['memory_mode'] = 'memmap'
        #~ pprint(params)
        
        method, d = self.dialog_method_features.get()
        params['feature_method'] = method
        params['feature_kargs'] = d
        
        method, d = self.dialog_method_cluster.get()
        params['cluster_method'] = method
        params['cluster_kargs'] = d
        
        #~ #TODO dialog for that
        #~ params['clean_cluster'] = False
        #~ params['clean_cluster_kargs'] = {}
        

        for chan_grp in self.chan_grps:
            print('### chan_grp', chan_grp, ' ###')
        
            #~ try:
            if 1:
                catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=chan_grp)
                apply_all_catalogue_steps(catalogueconstructor, params, verbose=True)            
                
            #~ except Exception as e:
                #~ print(e)
                #~ self.errorToMessageBox(e)
                
        self.refresh_info()
    
    
    def open_cataloguewin(self):
        if self.dataio is None: return
        if len(self.chan_grps) != 1: return
            
        #~ try:
        if True:
            catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=self.chan_grps[0])
            win = CatalogueWindow(catalogueconstructor)
            win.setWindowTitle(self.dataio.channel_group_label(chan_grp=self.chan_grps[0]))
            win.show()
            self.open_windows.append(win)
        #~ except Exception as e:
            #~ print(e)
            #~ self.errorToMessageBox(e)
    
    def run_peeler(self):
        if self.dataio is None: return
        
        #TODO find something better when several segment
        #~ lengths = [ self.dataio.datasource.get_segment_shape(i)[0] for i in range(self.dataio.nb_segment)]
        #~ max_duration = max(lengths)/self.dataio.sample_rate
        
        total_duration = sum(self.dataio.get_segment_length(seg_num=seg_num) / self.dataio.sample_rate for seg_num in range(self.dataio.nb_segment))
        
        for m in self.dialog_method_peeler.methods:
            self.dialog_method_peeler.all_params[m]['duration'] = total_duration
        
        
        #~ dia = ParamDialog(gui_params.peeler_params)
        #~ dia.resize(450, 500)
        
        #~ if not dia.exec_():
            #~ return
        #~ d = dia.get()
            
        if not self.dialog_method_peeler.exec_():
            return
        engine, d = self.dialog_method_peeler.get()
        #~ print('run_peeler')
        #~ print(engine)
        #~ print(d)
        d['engine'] = engine

        #~ duration = d['duration'] if d['limit_duration'] else None
        #~ d.pop('limit_duration')
        duration = d.pop('duration')
        print('duration', duration)
        
        errors = []
        for chan_grp in self.chan_grps:
            try:
            #~ if True:
                initial_catalogue = self.dataio.load_catalogue(chan_grp=chan_grp)
                if initial_catalogue is None:
                    txt =  """chan_grp{}
Catalogue do not exists, please do:
    1. Initialize Catalogue (if not done)
    2. Open CatalogueWindow
    3. Make catalogue for peeler
                    """.format(chan_grp)
                    errors.append(txt)
                    continue
                
                pprint(d)
                peeler = Peeler(self.dataio)
                peeler.change_params(catalogue=initial_catalogue, **d)
                
                t1 = time.perf_counter()
                peeler.run(duration=duration)
                t2 = time.perf_counter()
                print('peeler.run_loop', t2-t1)
                
            except Exception as e:
                print(e)
                error = """chan_grp{}\n{}""".format(chan_grp, e)
                errors.append(error)
        
        for error in errors:
            self.errorToMessageBox(error)
    
    def open_peelerwin(self):
        if self.dataio is None: return
        
        if len(self.chan_grps) != 1:
            return
        
        try:
            initial_catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grps[0])
            win = PeelerWindow(dataio=self.dataio, catalogue=initial_catalogue)
            win.show()
            self.open_windows.append(win)
        except Exception as e:
            print(e)
            self.errorToMessageBox(e)

    def open_ephyviewer(self):
        if self.win_viewer is not None:
            self.win_viewer.close()
            self.win_viewer = None
        
        if self.dataio  is None:
            return
        if not hasattr(self.dataio.datasource, 'rawios'):
            return
        
        sources = ephyviewer.get_sources_from_neo_rawio(self.dataio.datasource.rawios[0])
        
        self.win_viewer = ephyviewer.MainViewer()
        
        for i, sig_source in enumerate(sources['signal']):
            view = ephyviewer.TraceViewer(source=sig_source, name='signal {}'.format(i))
            view.params['scale_mode'] = 'same_for_all'
            view.params['display_labels'] = True
            view.auto_scale()
            if i==0:
                self.win_viewer.add_view(view)
            else:
                self.win_viewer.add_view(view, tabify_with='signal {}'.format(i-1))
        
        self.win_viewer.show()

    def open_probe_viewer(self):
        if self.dataio is None:
            return
        
        if self.probe_viewer is not None:
            self.probe_viewer.close()
            self.probe_viewer = None
        
        self.probe_viewer = ProbeGeometryView(channel_groups=self.dataio.channel_groups, parent=self)
        self.probe_viewer.setWindowFlags(QT.Qt.Window)
        self.probe_viewer.show()
    
    def export_spikes(self):
        if self.dataio is None:
            return
        
        possible_formats = [e.ext for e in export_list]
        params = [
            {'name': 'format', 'type': 'list', 'limits':possible_formats},
            {'name': 'split_by_cluster', 'type': 'bool', 'value':False},
            {'name': 'use_cell_label', 'type': 'bool', 'value':True},
        ]
        dialog = ParamDialog(params, parent=self)
        if not dialog.exec_():
            return
        p = dialog.get()
        self.dataio.export_spikes(export_path=None,
                split_by_cluster=p['split_by_cluster'],  use_cell_label=p['use_cell_label'], formats=p['format'])

    def generate_report(self):
        if self.dataio is None:
            return
        generate_report(self.dataio)

    def check_prb_file(self):
        fd = QT.QFileDialog(fileMode=QT.QFileDialog.AnyFile, acceptMode=QT.QFileDialog.AcceptOpen)
        fd.setNameFilters(['Probe geometry (*.PRB, *.prb)', 'All (*)'])
        fd.setViewMode( QT.QFileDialog.Detail )
        if fd.exec_():
            prb_filename = fd.selectedFiles()[0]
            #~ print(prb_filename)
            try:
                channel_groups = open_prb(prb_filename)
                self.probe_viewer = ProbeGeometryView(channel_groups=channel_groups, parent=self)
            except:
                self.warn('Error in PRB')
                return
            self.probe_viewer.setWindowFlags(QT.Qt.Window)
            self.probe_viewer.show()
    
    def open_gpuselector(self):
        if not HAVE_PYOPENCL:
            return
        if not self.dialog_gpuselector.exec_():
            return
        self.dialog_gpuselector.apply_cl_setting()

