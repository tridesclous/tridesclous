from .myqt import QT
import pyqtgraph as pg

import numpy as np
import  itertools

from .base import WidgetBase
from .tools import ParamDialog



class PairList(WidgetBase):
    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.combo = QT.QComboBox()
        self.layout.addWidget(self.combo)
        self.combo.addItems(['all pairs', 'similar amplitude ratio', ]) #'high similarity'
        self.combo.currentTextChanged.connect(self.refresh)
        
        but = QT.QPushButton('settings')
        self.layout.addWidget(but)
        but.clicked.connect(self.open_settings)
        
        

        self.table = QT.QTableWidget(selectionMode=QT.QAbstractItemView.SingleSelection,
                                                        selectionBehavior=QT.QAbstractItemView.SelectRows)
        self.table.setContextMenuPolicy(QT.Qt.CustomContextMenu)
        self.layout.addWidget(self.table)
        self.table.itemSelectionChanged.connect(self.on_item_selection_changed)
        self.table.customContextMenuRequested.connect(self.open_context_menu)
        
        self.menu = QT.QMenu()
        act = self.menu.addAction('Merge')
        act.triggered.connect(self.do_merge)

        act = self.menu.addAction('Tag same cell')
        act.triggered.connect(self.do_tag_same_cell)
        
        self.create_settings()
        
        self.refresh()

    def create_settings(self):
        _params = [
                          {'name': 'threshold', 'type': 'float', 'value' :.9},
                          ]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        
        self.params.sigTreeStateChanged.connect(self.refresh)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for waveforms viewer')
        self.tree_params.setWindowFlags(QT.Qt.Window)
        
        self.params.sigTreeStateChanged.connect(self.on_params_change)
    
    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()
    
    def on_params_change(self):
        self.refresh()
    
    def on_item_selection_changed(self):
        inds = self.table.selectedIndexes()
        if len(inds)!=2:
            return
        k1, k2 = self.pairs[inds[0].row()]
        for k in self.controller.cluster_visible:
            self.controller.cluster_visible[k] = k in (k1, k2)
        
        self.cluster_visibility_changed.emit()

    def open_context_menu(self):
        self.menu.popup(self.cursor().pos())
    
    def do_merge(self):
        if len(self.table.selectedIndexes())==0:
            return
        ind = self.table.selectedIndexes()[0].row()
        
        label_to_merge = list(self.pairs[ind])
        self.controller.merge_cluster(label_to_merge)
        self.refresh()
        self.spike_label_changed.emit()
    
    def do_tag_same_cell(self):
        if len(self.table.selectedIndexes())==0:
            return
        ind = self.table.selectedIndexes()[0].row()
        
        label_to_merge = list(self.pairs[ind])
        self.controller.tag_same_cell(label_to_merge)
        self.refresh()
        self.cluster_tag_changed.emit()
        
    
    def refresh(self):
        self.table.clear()
        labels = ['cluster_label_1', 'cluster_label_2', 'cell_label_1', 'cell_label_2' ]
        self.table.setColumnCount(len(labels))
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setColumnWidth(0, 100)
        self.table.setColumnWidth(1, 100)
        
        mode = self.combo.currentText()
        print('mode', mode)
        

        
        if mode == 'all pairs':
            labels = self.controller.positive_cluster_labels
            #~ labels = labels[labels>=0]
            self.pairs = list(itertools.combinations(labels, 2))
        elif mode == 'similar amplitude ratio':
            self.pairs = self.controller.detect_similar_waveform_ratio(threshold=self.params['threshold'])
            print(self.controller.cc.ratio_similarity[1, 5])
        #~ elif mode == 'high similarity':
            #~ self.pairs = self.controller.detect_high_similarity(threshold=0.9)
            #~ print(self.controller.cc.similarity[1, 5])
        
        self.table.setRowCount(len(self.pairs))
        
        for r in range(len(self.pairs)):
            k1, k2 = self.pairs[r]
            
            for c, k in enumerate((k1, k2)):
                color = self.controller.qcolors.get(k, QT.QColor( 'white'))
                pix = QT.QPixmap(16,16)
                pix.fill(color)
                icon = QT.QIcon(pix)
                
                name = '{} (nb={})'.format(k, self.controller.cluster_count[k])
                item = QT.QTableWidgetItem(name)
                item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
                self.table.setItem(r,c, item)
                item.setIcon(icon)
                
                
                cell_label = self.controller.cell_labels[self.controller.cluster_labels==k][0]
                name = '{}'.format(cell_label)
                item = QT.QTableWidgetItem(name)
                item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
                self.table.setItem(r,c+2, item)
                
                


        

    def on_spike_selection_changed(self):
        pass

    def on_spike_label_changed(self):
        self.refresh()
    
    def on_colors_changed(self):
        self.refresh()
    
    def on_cluster_visibility_changed(self):
        pass

