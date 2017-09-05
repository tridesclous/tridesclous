from .myqt import QT
import pyqtgraph as pg

import numpy as np
import  itertools
#~ import random

from .base import WidgetBase
from .tools import ParamDialog



class PairList(WidgetBase):
    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)

        self.table = QT.QTableWidget(selectionMode=QT.QAbstractItemView.SingleSelection,
                                                        selectionBehavior=QT.QAbstractItemView.SelectRows)
        self.layout.addWidget(self.table)
        self.table.itemSelectionChanged.connect(self.on_item_selection_changed)
        
        self.refresh()
    
    
    def on_item_selection_changed(self):
        inds = self.table.selectedIndexes()
        if len(inds)!=2:
            return
        k1, k2 = self.pairs[inds[0].row()]
        for k in self.controller.cluster_visible:
            self.controller.cluster_visible[k] = k in (k1, k2)
        
        self.cluster_visibility_changed.emit()
    
    def refresh(self):
        self.table.clear()
        labels = ['cluster1', 'cluster2']
        self.table.setColumnCount(len(labels))
        self.table.setHorizontalHeaderLabels(labels)
        #~ self.table.setMinimumWidth(100)
        #~ self.table.setColumnWidth(0,60)
        self.table.setContextMenuPolicy(QT.Qt.CustomContextMenu)
        #~ self.table.customContextMenuRequested.connect(self.open_context_menu)
        self.table.setSelectionMode(QT.QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QT.QAbstractItemView.SelectRows)
        
        labels = self.controller.cluster_labels
        labels = labels[labels>0]
        self.pairs = list(itertools.combinations(labels, 2))
        
        self.table.setRowCount(len(self.pairs))
        
        
        for r in range(len(self.pairs)):
            k1, k2 = self.pairs[r]
            
            for c, k in enumerate((k1, k2)):
                color = self.controller.qcolors.get(k, QT.QColor( 'white'))
                pix = QT.QPixmap(10,10)
                pix.fill(color)
                icon = QT.QIcon(pix)
                
                name = '{}'.format(k)
                item = QT.QTableWidgetItem(name)
                item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
                self.table.setItem(r,0+c, item)
                item.setIcon(icon)
        
    
    #~ def open_context_menu(self):
        #~ pass