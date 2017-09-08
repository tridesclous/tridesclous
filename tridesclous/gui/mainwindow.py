from .myqt import QT
import pyqtgraph as pg

import time
import os
from collections import OrderedDict
from ..dataio import DataIO
from ..datasource import data_source_classes
from .tools import get_dict_from_group_param, ParamDialog

from ..catalogueconstructor import CatalogueConstructor
from .cataloguewindow import CatalogueWindow
from ..peeler import Peeler
from .peelerwindow import PeelerWindow
from .initializedatasetwindow import InitializeDatasetWindow

from . import icons

class MainWindow(QT.QMainWindow):
    def __init__(self):
        QT.QMainWindow.__init__(self)
        
        self.setWindowIcon(QT.QIcon(':/main_icon.png'))
        
        self.dataio = None
        self.catalogueconstructor = None
        
        self.resize(800, 600)

        self.create_actions_and_menu()

        w = QT.QWidget()
        self.setCentralWidget(w)
        mainlayout  = QT.QVBoxLayout()
        w.setLayout(mainlayout)
        
        self.label_info = QT.QLabel('Nothing loaded')
        mainlayout.addWidget(self.label_info)
        
        mainlayout.addStretch()
        
        self.open_windows = []
        
        

    def create_actions_and_menu(self):
        #~ self.actions = OrderedDict()
        
        self.toolbar = QT.QToolBar()
        self.toolbar.setToolButtonStyle(QT.Qt.ToolButtonTextUnderIcon)
        self.addToolBar(self.toolbar)
        self.toolbar.setIconSize(QT.QSize(60, 40))
        
        self.file_menu = self.menuBar().addMenu(self.tr("File"))
        
        do_open = QT.QAction('&Open', self, shortcut = "Ctrl+O")
        do_open.triggered.connect(self.open_dialog)
        self.file_menu.addAction(do_open)

        do_init = QT.QAction('&Initialize dataset', self, shortcut = "Ctrl+I")
        do_init.triggered.connect(self.initialize_dataset_dialog)
        self.file_menu.addAction(do_init)
        
        self.toolbar.addWidget(QT.QLabel('chan_grp:'))
        self.combo_chan_grp = QT.QComboBox()
        self.toolbar.addWidget(self.combo_chan_grp)
        self.combo_chan_grp.currentIndexChanged .connect(self.on_chan_grp_change)

        do_init_cataloguewin = QT.QAction('1- Initialize Catalogue', self)
        do_init_cataloguewin.triggered.connect(self.initialize_catalogue)
        self.toolbar.addAction(do_init_cataloguewin)
        
        do_open_cataloguewin = QT.QAction('2- open CatalogueWindow', self)
        do_open_cataloguewin.triggered.connect(self.open_cataloguewin)
        self.toolbar.addAction(do_open_cataloguewin)

        do_run_peeler = QT.QAction('4- run Peeler', self)
        do_run_peeler.triggered.connect(self.run_peeler)
        self.toolbar.addAction(do_run_peeler)
        
        do_open_peelerwin = QT.QAction('4- open PeelerWindow', self)
        do_open_peelerwin.triggered.connect(self.open_peelerwin)
        self.toolbar.addAction(do_open_peelerwin)
        
        self.toolbar.addSeparator()
        
        info_act = QT.QAction('Info', self,checkable = False, icon=QT.QIcon(":main_icon.png"))
        self.toolbar.addAction(info_act)
        
        

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
    
    def _open_dataio(self, dirname):
        for win in self.open_windows:
            win.close()
        self.open_windows = []
        
        self.dataio = DataIO(dirname=dirname)
        
        
        self.combo_chan_grp.clear()
        self.combo_chan_grp.addItems([str(k) for k in self.dataio.channel_groups.keys()])
        self.on_chan_grp_change()
    
    @property
    def chan_grp(self):
        return int(self.combo_chan_grp.currentText())
        
    def on_chan_grp_change(self, index=None):
        self.catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=self.chan_grp)
        self.refresh_info()
        
    
    def initialize_dataset_dialog(self):
        init_dia = InitializeDatasetWindow(parent=self)
        if init_dia.exec_():
            self._open_dataio(init_dia.dirname_created)


    
    def initialize_catalogue(self):
        params = [
            {'name':'duration', 'type': 'float', 'value':60., 'suffix': 's', 'siPrefix': True},
            {'name':'preprocessor', 'type':'group', 
                'children':[
                    {'name': 'highpass_freq', 'type': 'float', 'value':400., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
                    {'name': 'lowpass_freq', 'type': 'float', 'value':5000., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
                    {'name': 'smooth_size', 'type': 'int', 'value':0},
                    {'name': 'common_ref_removal', 'type': 'bool', 'value':True},
                    {'name': 'chunksize', 'type': 'int', 'value':1024, 'decilmals':5},
                    {'name': 'backward_chunksize', 'type': 'int', 'value':1280, 'decilmals':5},
                    
                    {'name': 'peakdetector_engine', 'type': 'list', 'values':['numpy', 'opencl']},
                    {'name': 'peak_sign', 'type': 'list', 'values':['-', '+']},
                    {'name': 'relative_threshold', 'type': 'float', 'value': 6., 'step': .1,},
                    {'name': 'peak_span', 'type': 'float', 'value':0.0005, 'step': 0.0001, 'suffix': 's', 'siPrefix': True},
                ]
            },
            {'name':'extract_waveforms', 'type':'group', 
                'children':[
                    {'name': 'n_left', 'type': 'int', 'value':-20},
                    {'name': 'n_right', 'type': 'int', 'value':30},
                    {'name': 'mode', 'type': 'list', 'values':['rand', 'all']},
                    {'name': 'nb_max', 'type': 'int', 'value':20000},
                    {'name': 'align_waveform', 'type': 'bool', 'value':True},
                    #~ {'name': 'subsample_ratio', 'type': 'int', 'value':20},
                ],
            },
            {'name':'project', 'type':'group', 
                'children':[
                    {'name': 'method', 'type': 'list', 'values':['pca']},
                    {'name' : 'n_components', 'type' : 'int', 'value' : 5},
                ],
            },
            {'name':'find_cluster', 'type':'group', 
                'children':[
                    {'name': 'method', 'type': 'list', 'values':['kmeans', 'gmm']},
                    {'name' : 'n_clusters', 'type' : 'int', 'value' : 8},
                ],
            },
            
            
        ]
        dia = ParamDialog(params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            print(d)
            try:
                #~ catalogueconstructor = CatalogueConstructor(dataio=self.dataio)
                self.catalogueconstructor.set_preprocessor_params(**d['preprocessor'])
                
                t1 = time.perf_counter()
                self.catalogueconstructor.estimate_signals_noise(seg_num=0, duration=10.)
                t2 = time.perf_counter()
                print('estimate_signals_noise', t2-t1)
                
                t1 = time.perf_counter()
                self.catalogueconstructor.run_signalprocessor(duration=d['duration'])
                t2 = time.perf_counter()
                print('run_signalprocessor', t2-t1)

                t1 = time.perf_counter()
                self.catalogueconstructor.extract_some_waveforms(**d['extract_waveforms'])
                t2 = time.perf_counter()
                print('extract_some_waveforms', t2-t1)

                #~ t1 = time.perf_counter()
                #~ n_left, n_right = catalogueconstructor.find_good_limits(mad_threshold = 1.1,)
                #~ t2 = time.perf_counter()
                #~ print('find_good_limits', t2-t1)

                t1 = time.perf_counter()
                self.catalogueconstructor.project(**d['project'])
                t2 = time.perf_counter()
                print('project', t2-t1)
                
                t1 = time.perf_counter()
                self.catalogueconstructor.find_clusters(**d['find_cluster'])
                t2 = time.perf_counter()
                print('find_clusters', t2-t1)
                
                print(self.catalogueconstructor)

                
                

            except Exception as e:
                print(e)
                
            self.refresh_info()
        
    
    def open_cataloguewin(self):
        if self.dataio is None: return
        try:
            win = CatalogueWindow(self.catalogueconstructor)
            win.show()
            self.open_windows.append(win)
        except Exception as e:
            print(e)
    
    def run_peeler(self):
        params = [
            {'name':'limit_duration', 'type': 'bool', 'value':True},
            {'name':'duration', 'type': 'float', 'value':300, 'suffix': 's', 'siPrefix': True},
            {'name': 'n_peel_level', 'type': 'int', 'value':2},
        ]
        
        dia = ParamDialog(params)
        dia.resize(450, 500)
        if dia.exec_():
            d = dia.get()
            print(d)
            
            try:
                initial_catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grp)
                peeler = Peeler(self.dataio)
                peeler.change_params(catalogue=initial_catalogue, n_peel_level=d['n_peel_level'])
                
                duration = d['duration'] if d['limit_duration'] else None
                
                t1 = time.perf_counter()
                peeler.run(chan_grp=self.chan_grp, duration=duration)
                t2 = time.perf_counter()
                print('peeler.run_loop', t2-t1)
                
            except Exception as e:
                print(e)
    
    def open_peelerwin(self):
        if self.dataio is None: return
        try:
            initial_catalogue = self.dataio.load_catalogue(chan_grp=self.chan_grp)
            win = PeelerWindow(dataio=self.dataio, catalogue=initial_catalogue)
            win.show()
            self.open_windows.append(win)
        except Exception as e:
            print(e)

    



