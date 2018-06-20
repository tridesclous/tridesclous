from .myqt import QT
import pyqtgraph as pg

import time
import os
from collections import OrderedDict
import pickle
import webbrowser

from ..dataio import DataIO
from ..datasource import data_source_classes
from .tools import get_dict_from_group_param, ParamDialog, MethodDialog #, open_dialog_methods
from  ..datasets import datasets_info, download_dataset

from ..catalogueconstructor import CatalogueConstructor
from .cataloguewindow import CatalogueWindow
from ..peeler import Peeler
from .peelerwindow import PeelerWindow
from .initializedatasetwindow import InitializeDatasetWindow
from .probegeometryview import ProbeGeometryView

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
    def __init__(self):
        QT.QMainWindow.__init__(self)
        
        self.setWindowIcon(QT.QIcon(':/main_icon.png'))
        
        self.dataio = None
        self.catalogueconstructor = None
        
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
        
        
        
    

    def create_actions_and_menu(self):
        
        self.toolbar = QT.QToolBar(orientation=QT.Vertical)
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(QT.LeftToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))
        
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
            
        
        self.toolbar.addWidget(QT.QLabel('Select chanel group:'))
        self.combo_chan_grp = QT.QComboBox()
        self.toolbar.addWidget(self.combo_chan_grp)
        self.combo_chan_grp.currentIndexChanged .connect(self.on_chan_grp_change)

        do_init_cataloguewin = QT.QAction('Initialize Catalogue', self, icon=QT.QIcon(":autocorrection.svg"))
        do_init_cataloguewin.triggered.connect(self.initialize_catalogue)
        self.toolbar.addAction(do_init_cataloguewin)
        
        do_open_cataloguewin = QT.QAction('Open CatalogueWindow', self,  icon=QT.QIcon(":catalogwinodw.png"))
        do_open_cataloguewin.triggered.connect(self.open_cataloguewin)
        self.toolbar.addAction(do_open_cataloguewin)


        do_run_peeler = QT.QAction('Run Peeler', self,  icon=QT.QIcon(":configure-shortcuts.svg"))
        do_run_peeler.triggered.connect(self.run_peeler)
        self.toolbar.addAction(do_run_peeler)
        
        do_open_peelerwin = QT.QAction('Open PeelerWindow', self,  icon=QT.QIcon(":peelerwindow.png"))
        do_open_peelerwin.triggered.connect(self.open_peelerwin)
        self.toolbar.addAction(do_open_peelerwin)
        
        self.toolbar.addSeparator()

        do_refresh = QT.QAction(u'Refresh', self,checkable = False, icon=QT.QIcon(":/view-refresh.svg"))
        do_refresh.triggered.connect(self.refresh_with_reload)
        self.toolbar.addAction(do_refresh)
        
        help_act = QT.QAction('Help', self,checkable = False, icon=QT.QIcon(":main_icon.png"))
        help_act.triggered.connect(self.open_webbrowser_help)
        self.toolbar.addAction(help_act)
        
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
        txt1 = self.dataio.__repr__()
        txt2 = self.catalogueconstructor.__repr__()
        self.label_info.setText(txt1+'\n\n'+txt2)
    
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
            act = self.recetly_opened_menu.addAction(dirname)
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
        self.combo_chan_grp.addItems([str(k) for k in self.dataio.channel_groups.keys()])
        self.combo_chan_grp.blockSignals(False)
        self.on_chan_grp_change()
    
    def release_closed_windows(self):
        for win in list(self.open_windows):
            if not win.isVisible():
                self.open_windows.remove(win)
    
    def close_window_chan_grp(self, chan_grp):
        for win in list(self.open_windows):
            if win.controller.chan_grp == self.chan_grp:
                win.close()
                self.open_windows.remove(win)
        

    @property
    def chan_grp(self):
        return int(self.combo_chan_grp.currentText())
        
    def on_chan_grp_change(self, index=None):
        self.catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=self.chan_grp)
        self.refresh_info()
        
        # this set a the a by default method depending the number of channels
        n = self.dataio.nb_channel(chan_grp=self.chan_grp)
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
        self.close_window_chan_grp(self.chan_grp)
        
        #collect parals with UI
        #~ dia = ParamDialog(gui_params.fullchain_params)
        #~ dia.resize(450, 600)
        #~ if not dia.exec_():
            #~ return
        #~ fullchain_kargs = dia.get()
        #~ print(fullchain_kargs)
        
        #~ n = self.dataio.nb_channel(chan_grp=self.chan_grp)
        #~ if 1<=n<9:
            #~ method0 = 'global_pca'
        #~ else:
            #~ method0 = 'peak_max'
        
        #~ feat_method, feat_kargs = open_dialog_methods(gui_params.features_params_by_methods, self,
                        #~ title='Which feature method ?', selected_method=method0)
        
        #~ if feat_method is None:
            #~ return
        
        #~ selected_method = 'sawchaincut'
        #~ clust_method, clust_kargs = open_dialog_methods(gui_params.cluster_params_by_methods, self,
                        #~ title='Which cluster method ?', selected_method=selected_method)
        #~ if clust_method is None:
            #~ return
        
        
        if not self.dialog_fullchain_params.exec_():
            return
        if not self.dialog_method_features.exec_():
            return
        if not self.dialog_method_cluster.exec_():
            return
        
        fullchain_kargs = self.dialog_fullchain_params.get()
        
        feat_method = self.dialog_method_features.param_method['method']
        feat_kargs = get_dict_from_group_param(self.dialog_method_features.all_params[feat_method], cascade=True)

        clust_method = self.dialog_method_cluster.param_method['method']
        clust_kargs = get_dict_from_group_param(self.dialog_method_cluster.all_params[clust_method], cascade=True)
        
        print(fullchain_kargs)
        print('feat_method', feat_method)
        print('clust_method', clust_method)

        
        
        
        try:
        #~ if 1:

            apply_all_catalogue_steps(self.catalogueconstructor, fullchain_kargs, 
                feat_method, feat_kargs,clust_method, clust_kargs, verbose=True)            
            
        except Exception as e:
            print(e)
            self.errorToMessageBox(e)
                
        self.refresh_info()
    
    
    def open_cataloguewin(self):
        if self.dataio is None: return
        try:
        #~ if True:
            win = CatalogueWindow(self.catalogueconstructor)
            win.show()
            self.open_windows.append(win)
            #~ win.
        except Exception as e:
            print(e)
            self.errorToMessageBox(e)
    
    def run_peeler(self):
        if self.dataio is None: return
        
        #TODO find something better when several segment
        lengths = [ self.dataio.datasource.get_segment_shape(i)[0] for i in range(self.dataio.nb_segment)]
        duration = max(lengths)/self.dataio.sample_rate
        
        gui_params.peeler_params[1]['value'] = duration
        
        
        dia = ParamDialog(gui_params.peeler_params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            print(d)
            
            try:
            #~ if True:
                initial_catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grp)
                if initial_catalogue is None:
                    txt =  """Catalogue do not exists, please do:
    1. Initialize Catalogue
    2. Open CatalogueWindow
    3. Make catalogue for peeler
                    """
                    self.warn(txt)
                    return
                
                peeler = Peeler(self.dataio)
                peeler.change_params(catalogue=initial_catalogue)
                
                duration = d['duration'] if d['limit_duration'] else None
                
                t1 = time.perf_counter()
                peeler.run(duration=duration)
                t2 = time.perf_counter()
                print('peeler.run_loop', t2-t1)
                
            except Exception as e:
                print(e)
                self.errorToMessageBox(e)
    
    def open_peelerwin(self):
        if self.dataio is None: return
        try:
            initial_catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grp)
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
        
        sources = ephyviewer.get_source_from_neo(self.dataio.datasource.rawios[0])
        
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
        


# TODO put this somewhere else
def apply_all_catalogue_steps(catalogueconstructor, fullchain_kargs, 
                feat_method, feat_kargs,clust_method, clust_kargs, verbose=True):
    """
    used bu offline mainwinwod and OnlineWindow
    """
    
    p = {}
    p.update(fullchain_kargs['preprocessor'])
    p.update(fullchain_kargs['peak_detector'])
    catalogueconstructor.set_preprocessor_params(**p)
    dataio = catalogueconstructor.dataio
    
    #TODO offer noise esatimation duration somewhere
    noise_duration = min(10., fullchain_kargs['duration'], dataio.get_segment_length(seg_num=0)/dataio.sample_rate*.99)
    print('noise_duration', noise_duration)
    t1 = time.perf_counter()
    catalogueconstructor.estimate_signals_noise(seg_num=0, duration=noise_duration)
    t2 = time.perf_counter()
    if verbose:
        print('estimate_signals_noise', t2-t1)
    
    t1 = time.perf_counter()
    catalogueconstructor.run_signalprocessor(duration=fullchain_kargs['duration'])
    t2 = time.perf_counter()
    if verbose:
        print('run_signalprocessor', t2-t1)

    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(**fullchain_kargs['extract_waveforms'])
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_waveforms', t2-t1)

    t1 = time.perf_counter()
    #~ duration = d['duration'] if d['limit_duration'] else None
    #~ d['clean_waveforms']
    catalogueconstructor.clean_waveforms(**fullchain_kargs['clean_waveforms'])
    t2 = time.perf_counter()
    if verbose:
        print('clean_waveforms', t2-t1)
    
    #~ t1 = time.perf_counter()
    #~ n_left, n_right = catalogueconstructor.find_good_limits(mad_threshold = 1.1,)
    #~ t2 = time.perf_counter()
    #~ print('find_good_limits', t2-t1)

    t1 = time.perf_counter()
    catalogueconstructor.extract_some_noise(**fullchain_kargs['noise_snippet'])
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_noise', t2-t1)

    t1 = time.perf_counter()
    catalogueconstructor.extract_some_features(method=feat_method, **feat_kargs)
    t2 = time.perf_counter()
    if verbose:
        print('project', t2-t1)
    
    t1 = time.perf_counter()
    catalogueconstructor.find_clusters(method=clust_method, **clust_kargs)
    t2 = time.perf_counter()
    if verbose:
        print('find_clusters', t2-t1)
    