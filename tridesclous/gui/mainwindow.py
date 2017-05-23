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
from .peelerwindow import PeelerWindow
from .initializedatasetwindow import InitializeDatasetWindow


class MainWindow(QT.QMainWindow):
    def __init__(self):
        QT.QMainWindow.__init__(self)
        
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

        do_init_cataloguewin = QT.QAction('Initialize Catalogue', self)
        do_init_cataloguewin.triggered.connect(self.initialize_catalogue)
        self.toolbar.addAction(do_init_cataloguewin)
        
        do_open_cataloguewin = QT.QAction('CatalogueWindow', self)
        do_open_cataloguewin.triggered.connect(self.open_cataloguewin)
        self.toolbar.addAction(do_open_cataloguewin)
        
        do_open_peelerwin = QT.QAction('PeelerWindow', self)
        do_open_peelerwin.triggered.connect(self.open_peelerwin)
        self.toolbar.addAction(do_open_peelerwin)

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
            {'name':'preprocessor', 'type':'group', 
                'children':[
                        {'name': 'highpass_freq', 'type': 'float', 'value':400., 'step': 10., 'suffix': 'Hz', 'siPrefix': True},
                        {'name': 'common_ref_removal', 'type': 'bool', 'value':True},
                        {'name': 'chunksize', 'type': 'int', 'value':1024, 'decilmals':5},
                        {'name': 'backward_chunksize', 'type': 'int', 'value':1280, 'decilmals':5},
                        
                        {'name': 'peakdetector_engine', 'type': 'list', 'values':['numpy', 'opencl']},
                        {'name': 'peak_sign', 'type': 'list', 'values':['-', '+']},
                        {'name': 'relative_threshold', 'type': 'float', 'value': 6., 'step': .1,},
                        {'name': 'peak_span', 'type': 'float', 'value':0.0009, 'step': 0.0001, 'suffix': 's', 'siPrefix': True},
                ]},
            {'name':'duration', 'type': 'float', 'value':10., 'suffix': 's', 'siPrefix': True},
        ]
        dia = ParamDialog(params)
        dia.resize(300, 300)
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

            except Exception as e:
                print(e)
        
    
    def open_cataloguewin(self):
        if self.dataio is None: return
        try:
            win = CatalogueWindow(self.catalogueconstructor)
            win.show()
            self.open_windows.append(win)
        except Exception as e:
            print(e)
        
    def open_peelerwin(self):
        if self.dataio is None: return
        try:
            chan_grp= 0
            initial_catalogue = self.dataio.load_catalogue(chan_grp=chan_grp)
            win = PeelerWindow(dataio=self.dataio, catalogue=initial_catalogue)
            win.show()
            self.open_windows.append(win)
        except Exception as e:
            print(e)

    



