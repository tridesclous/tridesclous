from .myqt import QT
import pyqtgraph as pg

import numpy as np
import  itertools

from .base import WidgetBase
from .tools import ParamDialog



class PairList(WidgetBase):
    """
    **Pair list**  is an intuituitive list of pair of cluster : when you click on
    a pair this make visible only this 2 cluster on all others views.
    
    This help to validate in a fast way clusters.
    
    For convinience this paris can be filtered:
      * **all pairs** : all combinaison
      * **high similarity**: select only pairs that have similarity over a threshold (in settings)
      * **similarity amplitude ratio**! select only pairs that have "similarity ratio" over a threshold (in settings)
    
    And they can be sorted by:
      * label
      * similarity
      * ratio_similarity
    
    WIth a right click you can:
      * **merge** a pair
      * **tag same cell**, this keep 2 cluster but there tag as same cell.
    
    """
    _params = [{'name': 'threshold_similarity', 'type': 'float', 'value' :.9, 'step' : 0.01},
                    {'name': 'threshold_ratio_similarity', 'type': 'float', 'value' :.8, 'step' : 0.01},
                ]

    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)

        #~ h = QT.QHBoxLayout()
        #~ self.layout.addLayout(h)
        self.combo_select = QT.QComboBox()
        #~ h.addWidget(QT.QLabel('Select'))
        #~ h.addWidget(self.combo_select)
        self.combo_select.addItems(['all pairs', 'similar amplitude ratio', 'high similarity']) #
        #~ self.combo_select.currentTextChanged.connect(self.refresh)
        #~ h.addStretch()

        h = QT.QHBoxLayout()
        self.layout.addLayout(h)
        h.addWidget(QT.QLabel('Sort by'))
        self.combo_sort = QT.QComboBox()
        self.combo_sort.addItems(['label', 'similarity', 'ratio_similarity'])
        self.combo_sort.currentIndexChanged.connect(self.refresh)
        h.addWidget(self.combo_sort)
        h.addStretch()
        
        
        #~ but = QT.QPushButton('settings')
        #~ self.layout.addWidget(but)
        #~ but.clicked.connect(self.open_settings)
        
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
        
        self.refresh()
    
    def on_item_selection_changed(self):
        inds = self.table.selectedIndexes()
        if len(inds)!=self.table.columnCount():
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
        labels = ['cluster_label_1', 'cluster_label_2', 'cell_label_1', 'cell_label_2' , 'similarity', 'ratio_similarity']
        self.table.setColumnCount(len(labels))
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setColumnWidth(0, 100)
        self.table.setColumnWidth(1, 100)
        
        #select
        mode = self.combo_select.currentText()
        if mode == 'all pairs':
            labels = self.controller.positive_cluster_labels
            #~ labels = labels[labels>=0]
            self.pairs = list(itertools.combinations(labels, 2))
        elif mode == 'similar amplitude ratio':
            self.pairs = self.controller.detect_similar_waveform_ratio(threshold=self.params['threshold_ratio_similarity'])
        elif mode == 'high similarity':
            self.pairs = self.controller.detect_high_similarity(threshold=self.params['threshold_similarity'])
        
        #sort
        mode = self.combo_sort.currentText()
        order = np.arange(len(self.pairs))
        if mode == 'label':
            pass
        elif mode == 'similarity':
            if self.controller.cluster_similarity is not None:
                order = []
                for r in range(len(self.pairs)):
                    k1, k2 = self.pairs[r]
                    ind1 = self.controller.positive_cluster_labels.tolist().index(k1)
                    ind2 = self.controller.positive_cluster_labels.tolist().index(k2)
                    order.append(self.controller.cluster_similarity[ind1, ind2])
                order = np.argsort(order)[::-1]
        elif mode == 'ratio_similarity':
            if self.controller.cluster_ratio_similarity is not None:
                order = []
                for r in range(len(self.pairs)):
                    k1, k2 = self.pairs[r]
                    ind1 = self.controller.positive_cluster_labels.tolist().index(k1)
                    ind2 = self.controller.positive_cluster_labels.tolist().index(k2)
                    order.append(self.controller.cluster_ratio_similarity[ind1, ind2])
                order = np.argsort(order)[::-1]
        self.pairs = [self.pairs[i] for i in order ]
        
        self.table.setRowCount(len(self.pairs))
        
        for r in range(len(self.pairs)):
            k1, k2 = self.pairs[r]
            ind1 = self.controller.positive_cluster_labels.tolist().index(k1)
            ind2 = self.controller.positive_cluster_labels.tolist().index(k2)
            
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
        
            if self.controller.cluster_similarity is not None:
                if self.controller.cluster_similarity.shape[0] == self.controller.positive_cluster_labels.size:
                    name = '{}'.format(self.controller.cluster_similarity[ind1, ind2])
                    item = QT.QTableWidgetItem(name)
                    item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
                    self.table.setItem(r,4, item)

            if self.controller.cluster_ratio_similarity is not None:
                if self.controller.cluster_ratio_similarity.shape[0] == self.controller.positive_cluster_labels.size:
                    name = '{}'.format(self.controller.cluster_ratio_similarity[ind1, ind2])
                    item = QT.QTableWidgetItem(name)
                    item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
                    self.table.setItem(r,5, item)
        
                


        

    def on_spike_selection_changed(self):
        pass

    def on_spike_label_changed(self):
        self.refresh()
    
    def on_colors_changed(self):
        self.refresh()
    
    def on_cluster_visibility_changed(self):
        pass

