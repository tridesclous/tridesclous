from .myqt import QT
import pyqtgraph as pg

import os
from collections import OrderedDict
from ..dataio import DataIO
from ..datasource import data_source_classes

from ..catalogueconstructor import CatalogueConstructor
from .cataloguewindow import CatalogueWindow
from .peelerwindow import PeelerWindow
from .initializedatasetwindow import InitializeDatasetWindow


class MainWindow(QT.QMainWindow):
    def __init__(self):
        QT.QMainWindow.__init__(self)
        
        self.dataio = None
        
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
        txt = self.dataio.__repr__()
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
    
    def _open_dataio(self, dirname):
        for win in self.open_windows:
            win.close()
        self.open_windows = []
        
        self.dataio = DataIO(dirname=dirname)
        self.refresh_info()
        self.combo_chan_grp.clear()
        self.combo_chan_grp.addItems([str(k) for k in self.dataio.channel_groups.keys()])

    def initialize_dataset_dialog(self):
        
        init_dia = InitializeDatasetWindow(parent=self)
        
        if init_dia.exec_():
            self._open_dataio(init_dia.dirname_created)
    
    def initialize_catalogue(self):
        print('initialize_catalogue')
    
    def open_cataloguewin(self):
        if self.dataio is None: return
        try:
            chan_grp= 0
            catalogueconstructor = CatalogueConstructor(dataio=self.dataio, chan_grp=chan_grp)
            win = CatalogueWindow(catalogueconstructor)
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

    



