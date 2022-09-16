from .myqt import QT
import pyqtgraph as pg

import os
from collections import OrderedDict

import tempfile

from ..dataio import DataIO
from ..datasource import data_source_classes
from .tools import get_dict_from_group_param
from ..tools import download_probe, fix_prb_file_py2
from ..probe_list import probe_list
from .probegeometryview import ProbeGeometryView




class InitializeDatasetWindow(QT.QDialog):
    def __init__(self, parent=None):
        QT.QDialog.__init__(self, parent = parent)
        
        self.setWindowTitle('Initailize new tridesclous')
        self.setModal(True)
        
        self.resize(600, 300)

        layout = QT.QVBoxLayout()
        self.setLayout(layout)
        
        #for step 'source_type'
        self.tree_params = pg.parametertree.ParameterTree(parent=self)
        self.tree_params.header().hide()
        layout.addWidget(self.tree_params)
        self.tree_params.hide()
        
        # for filenames
        self.but_addfiles = QT.QPushButton('Add files')
        layout.addWidget(self.but_addfiles)
        self.but_addfiles.clicked.connect(self.on_addfiles)
        self.but_addfiles.hide()
        self.list_files = QT.QListWidget()
        layout.addWidget(self.list_files)
        self.list_files.hide()
        
        # for chan_grp
        self.changroup_widget = ChannelGroupWidget()
        layout.addWidget(self.changroup_widget)
        self.changroup_widget.hide()
        
        # for dirname
        self.but_dirname = QT.QPushButton('change path')
        layout.addWidget(self.but_dirname)
        self.but_dirname.clicked.connect(self.on_change_path)
        self.but_dirname.hide()
        self.label_path = QT.QLabel()
        layout.addWidget(self.label_path)
        self.label_path.hide()
        self.edit_dirname = QT.QLineEdit()
        layout.addWidget(self.edit_dirname)
        self.edit_dirname.hide()
        
        layout.addStretch()
        
        self.but_next = QT.QPushButton('Next >')
        layout.addWidget(self.but_next)
        self.but_next.clicked.connect(self.validate_step)
        
        self.steps = ['source_type', 'filenames', 'source_params', 'mk_chan_grp', 'dirname']
        self.display_step(self.steps[0])
        
        self.final_params = {}
    
    
    def display_step(self, step):
        self.actual_step = step
        
        if step=='source_type':
            source_types = list(data_source_classes.keys())
            source_types.remove('InMemory')
            #~ if 'RawBinarySignal' in source_types:
                #~ source_types.remove('RawData')
            
            params = [{'name': 'source_type', 'type': 'list', 'limits':source_types},
                            ]
            self.step_params = pg.parametertree.Parameter.create(name='Select source type', type='group', children = params)
            self.tree_params.setParameters(self.step_params, showTop=True)
            self.tree_params.show()
        
        elif step=='filenames':
            self.but_addfiles.show()
            self.list_files.show()
        
        elif step=='source_params':
            params = self.datasource_class.gui_params
            self.step_params = pg.parametertree.Parameter.create(name='Set params', type='group', children=params)
            self.tree_params.setParameters(self.step_params, showTop=True)
            self.tree_params.show()

        elif step=='mk_chan_grp':
            source_params = self._get_source_param()
            
            #~ try:
            if True:
                channel_names = self.datasource_class(**source_params).get_channel_names()
            #~ except:
                #~ #TYODO warning
                #~ print('Error file')
                #~ return
            print(channel_names)
            self.changroup_widget.set_channel_names(channel_names)
            self.changroup_widget.show()
        
        elif step=='dirname':
            self.but_dirname.show()
            self.label_path.show()
            self.edit_dirname.show()
            
            names = self.final_params['file_or_dir_names']
            name0 = names[0]
            suggested_dir = os.path.dirname(name0)
            if len(names)==1:
                name, _ = os.path.splitext(os.path.basename(name0))
                suggested_name = 'tdc_'+name
            else:
                suggested_name = 'tdc_'
            
            self.label_path.setText(suggested_dir)
            self.edit_dirname.setText(suggested_name)
        
    def _get_source_param(self):
        source_params = dict(self.final_params['source_params'])
        if self.datasource_class.mode =='one-file':
            source_params['filename'] = self.final_params['file_or_dir_names'][0]
        elif self.datasource_class.mode =='multi-file':
            source_params['filenames'] = self.final_params['file_or_dir_names']
        elif self.datasource_class.mode == 'one-dir':
            source_params['dirname'] = self.final_params['file_or_dir_names'][0]
        elif self.datasource_class.mode == 'multi-dir':
            source_params['dirnames'] = self.final_params['file_or_dir_names']
        print('source_params', source_params)
        return source_params
   
    def validate_step(self):
        #~ print('validate_step', self.actual_step)
        step = self.actual_step
        if step=='source_type':
            self.final_params['source_type'] = self.step_params['source_type']
            self.tree_params.hide()
            self.display_step('filenames')
            self.datasource_class = data_source_classes[self.final_params['source_type']]
            
        elif step=='filenames':
            file_or_dir_names = [self.list_files.item(i).text() for i in range(self.list_files.count())]
            if len(file_or_dir_names)==0:
                return
            if self.datasource_class.mode == 'one-file' and len(file_or_dir_names)!=1:
                return
            if self.datasource_class.mode == 'one-dir' and len(file_or_dir_names)!=1:
                return
            
            self.final_params['file_or_dir_names'] = file_or_dir_names
            self.but_addfiles.hide()
            self.list_files.hide()
            
            if self.datasource_class.gui_params is None:
                self.final_params['source_params'] = {}
                self.display_step('mk_chan_grp')
            else:
                self.display_step('source_params')
        
        elif step=='source_params':
            self.final_params['source_params'] = get_dict_from_group_param(self.step_params)
            self.tree_params.hide()
            self.display_step('mk_chan_grp')
        
        elif step=='mk_chan_grp':
            #~ print(self.final_params)
            self.final_params['channel_groups'] = self.changroup_widget.channel_groups
            self.changroup_widget.hide()
            self.display_step('dirname')
        
        elif step=='dirname':
            p1 = self.label_path.text()
            p2 = self.edit_dirname.text()
            dirname = os.path.join(p1, p2)
            if DataIO.check_initialized(dirname):
                #todo warning
                print('already exist')
                return
            self.final_params['dirname'] = dirname
            self.but_dirname.hide()
            self.label_path.hide()
            self.edit_dirname.hide()
            self.make_it()
            self.accept()
    
    def on_addfiles(self):
        print('on_addfiles')
        print(self.datasource_class)
        print(self.datasource_class.mode)
        
        if self.datasource_class.mode.endswith('-file'):
            fd = QT.QFileDialog(fileMode=QT.QFileDialog.ExistingFiles, acceptMode=QT.QFileDialog.AcceptOpen)
            fd.setNameFilters(['All (*)'])
        elif self.datasource_class.mode.endswith('-dir'):
            fd = QT.QFileDialog(fileMode=QT.QFileDialog.DirectoryOnly, acceptMode=QT.QFileDialog.AcceptOpen)
            #~ fd.setNameFilters(['All (*)'])
        
        fd.setViewMode(QT.QFileDialog.Detail)
        if fd.exec_():
            filenames = fd.selectedFiles()
            self.list_files.addItems(filenames)
    
    def on_change_path(self):
        dirname = self.label_path.text()
        fd = QT.QFileDialog(fileMode=QT.QFileDialog.DirectoryOnly, acceptMode=QT.QFileDialog.AcceptOpen)
        fd.setViewMode(QT.QFileDialog.Detail)
        fd.setDirectory(os.path.dirname(dirname))
        if fd.exec_():
            dirname = fd.selectedFiles()[0]
            self.label_path.setText(dirname)
            #~ self.list_files.addItems(filenames)
            
    def make_it(self):
        try:
            p = self.final_params
            
            source_params = self._get_source_param()
            
            dataio = DataIO(p['dirname'])
            dataio.set_data_source(type=p['source_type'], **source_params)
            dataio.set_channel_groups(p['channel_groups'])
            print(dataio)
            self.dirname_created = p['dirname']
        except Exception as e:
            print(e)



class ChannelGroupWidget(QT.QWidget):
    def __init__(self, parent=None):
        QT.QWidget.__init__(self, parent = parent)
        
        layout = QT.QHBoxLayout()
        self.setLayout(layout)
        
        self.list_channel = QT.QListWidget(selectionMode=QT.QAbstractItemView.ExtendedSelection, selectionBehavior=QT.QTreeView.SelectRows)
        layout.addWidget(self.list_channel)

        v = QT.QVBoxLayout()
        
        but = QT.QPushButton('clear')
        v.addWidget(but)
        but.clicked.connect(self.clear_table)
        
        layout.addLayout(v)
        but = QT.QPushButton('Set manual channel group from selection')
        v.addWidget(but)
        but.clicked.connect(self.add_chan_grp)

        but = QT.QPushButton('Set channel group from PRB file')
        v.addWidget(but)
        but.clicked.connect(self.open_prb_file)
        
        h = QT.QHBoxLayout()
        v.addLayout(h)
        but = QT.QPushButton('Download PRB file')
        but.clicked.connect(self.download_prb_file)
        h.addWidget(but)
        self.comboPrb = QT.QComboBox()
        h.addWidget(self.comboPrb)
        all_prb = ['']
        for origin, probe_names in probe_list.items():
            for probe_name in probe_names:
                all_prb.append('{}: {}'.format(origin, probe_name))
        self.comboPrb.addItems(all_prb)
        
        
        self.table_grp = QT.QTableWidget()
        v.addWidget(self.table_grp)

        but = QT.QPushButton('Show probes geometry')
        v.addWidget(but)
        but.clicked.connect(self.show_prb_geometry)
        
        self.channel_names = None
        self.channel_groups = None
        
        self.geometryview = None
    
    def set_channel_names(self, channel_names):
        self.n = len(channel_names)
        self.channel_names = channel_names
        self.list_channel.clear()
        self.list_channel.addItems(channel_names)
        
        channels = list(range(self.n))
        #~ geometry = { c: [0, i] for i, c in enumerate(channels) }
        geometry = None
        self.channel_groups = OrderedDict()
        self.channel_groups[0] = dict(channels=channels, geometry=geometry)
        
        self.refresh_table()
        

    def refresh_table(self):
        t = self.table_grp
        
        t.clear()
        t.setColumnCount(3)
        t.setRowCount(len(self.channel_groups))
        
        t.setHorizontalHeaderLabels(['key', 'nb_chan', 'channels'])
        for i, k in enumerate(self.channel_groups.keys()):
            channel_group = self.channel_groups[k]
            channels = channel_group['channels']
            t.setItem(i, 0,  QT.QTableWidgetItem('{}'.format(k)))
            t.setItem(i, 1,  QT.QTableWidgetItem('{}'.format(len(channels))))
            txt = ' '.join([str(c) for c in channels])
            t.setItem(i, 2,  QT.QTableWidgetItem('[{}]'.format(txt)))
        
        t.resizeColumnsToContents()
            
    def add_chan_grp(self):
        channels = [ind.row() for ind in self.list_channel.selectedIndexes()]
        if len(channels)==0: return
        k = len(self.channel_groups)
        #~ geometry = { c: [0, i] for i, c in enumerate(channels) }
        geometry = None
        self.channel_groups[k] = dict(channels=channels, geometry=geometry)
        self.refresh_table()
    
    def open_prb_file(self):
        fd = QT.QFileDialog(fileMode=QT.QFileDialog.ExistingFiles, acceptMode=QT.QFileDialog.AcceptOpen)
        fd.setNameFilters(['prb (*.prb *.PRB)', 'All (*)'])
        fd.setViewMode(QT.QFileDialog.Detail)
        if fd.exec_():
            prb_filename = fd.selectedFiles()[0]
            
            fix_prb_file_py2(prb_filename)
            
            d = {}
            exec(open(prb_filename).read(), None, d)
            
            self.channel_groups = OrderedDict()
            self.channel_groups.update(d['channel_groups'])
            self.refresh_table()
    
    def download_prb_file(self):
        probe_name = str(self.comboPrb.currentText())
        if len(probe_name) == 0:
            return
        origin, probe_name = probe_name.split(': ')
        #~ print(origin, probe_name)
        local_dirname = tempfile.gettempdir()
        #~ print(local_dirname)
        prb_filename = download_probe(local_dirname, probe_name, origin=origin)
        #~ print(prb_filename)

        d = {}
        exec(open(prb_filename).read(), None, d)
        
        #~ print()
        self.channel_groups = OrderedDict()
        self.channel_groups.update(d['channel_groups'])
        self.refresh_table()

    
    def clear_table(self):
        self.channel_groups = OrderedDict()
        self.refresh_table()

    def show_prb_geometry(self):
        if self.geometryview is not None:
            self.geometryview.close()
        
        self.geometryview = ProbeGeometryView(channel_groups=self.channel_groups, parent=self)
        #~ self.geometryview.setModal(True)
        self.geometryview.setWindowFlags(QT.Qt.Window)
        
        self.geometryview.show()
        
