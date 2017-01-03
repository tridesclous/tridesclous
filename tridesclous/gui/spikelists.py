import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np

from .base import WidgetBase
from .tools import ParamDialog


class SpikeModel(QtCore.QAbstractItemModel):
    def __init__(self, parent =None, controller=None):
        QtCore.QAbstractItemModel.__init__(self,parent)
        self.controller = controller
        self.refresh_colors()
    
    def columnCount(self , parentIndex):
        return 6
    
    def rowCount(self, parentIndex):
        #~ if not parentIndex.isValid() and self.cc.peak_label is not None:
        if not parentIndex.isValid():
            self.visible_ind, = np.nonzero(self.controller.spikes['visible'])
            return self.visible_ind.size
            
        else :
            return 0
    
    def index(self, row, column, parentIndex):
        if not parentIndex.isValid():
            if column==0:
                childItem = row
            return self.createIndex(row, column, None)
        else:
            return QtCore.QModelIndex()
    
    def parent(self, index):
        return QtCore.QModelIndex()
    
    def data(self, index, role):
        if not index.isValid():
            return None
        
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.DecorationRole):
            return
        
        col = index.column()
        row = index.row()
        
        #~ t_start = 0.
        
        abs_ind = self.visible_ind[row]
        spike = self.controller.spikes[abs_ind]
        
        spike_time = (spike['index']+ spike['jitter'])/self.controller.dataio.sample_rate 
        
        if role ==QtCore.Qt.DisplayRole :
            if col == 0:
                return '{}'.format(abs_ind)
            elif col == 1:
                return '{}'.format(spike['segment'])
            elif col == 2:
                return '{}'.format(spike['index'])
            elif col == 3:
                return '{:.2f}'.format(spike['jitter'])
            elif col == 4:
                return '{:.4f}'.format(spike_time)
            elif col == 5:
                return '{}'.format(spike['label'])
            else:
                return None
        elif role == QtCore.Qt.DecorationRole :
            if col != 0: return None
            if spike['label'] in self.icons:
                return self.icons[spike['label']]
            else:
                return None
        else :
            return None
    
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable #| Qt.ItemIsDragEnabled

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return  ['num', 'seg_num', 'index', 'jitter', 'time', 'cluster_label'][section]
        return
    
    def refresh_colors(self):
        self.icons = { }
        for k, color in self.controller.qcolors.items():
            pix = QtGui.QPixmap(10,10 )
            pix.fill(color)
            self.icons[k] = QtGui.QIcon(pix)
        #~ self.icons[-1] = QIcon(':/user-trash.png')
        self.layoutChanged.emit()
        

class SpikeList(WidgetBase):
    def __init__(self,controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        self.controller = controller
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.layout.addWidget(QtGui.QLabel('<b>All spikes</b>') )
        
        self.tree = QtGui.QTreeView(minimumWidth = 100, uniformRowHeights = True,
                    selectionMode= QtGui.QAbstractItemView.ExtendedSelection, selectionBehavior = QtGui.QTreeView.SelectRows,
                    contextMenuPolicy = QtCore.Qt.CustomContextMenu,)
        
        self.layout.addWidget(self.tree)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        
        self.model = SpikeModel(controller=self.controller)
        self.tree.setModel(self.model)
        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)

        for i in range(self.model.columnCount(None)):
            self.tree.resizeColumnToContents(i)
        self.tree.setColumnWidth(0,80)
    
    def refresh(self):
        self.model.refresh_colors()
    
    def on_tree_selection(self):
        self.controller.spikes['selected'][:] = False
        for index in self.tree.selectedIndexes():
            if index.column() == 0:
                ind = self.model.visible_ind[index.row()]
                self.controller.spikes['selected'][ind] = True
        self.spike_selection_changed.emit()
    
    def on_spike_selection_changed(self):
        self.tree.selectionModel().selectionChanged.disconnect(self.on_tree_selection)
        
        row_selected, = np.nonzero(self.controller.spikes['selected'][self.model.visible_ind])
        
        if row_selected.size>100:#otherwise this is verry slow
            row_selected = row_selected[:10]
        
        # change selection
        self.tree.selectionModel().clearSelection()
        flags = QtCore.QItemSelectionModel.Select #| QItemSelectionModel.Rows
        itemsSelection = QtCore.QItemSelection()
        for r in row_selected:
            for c in range(2):
                index = self.tree.model().index(r,c,QtCore.QModelIndex())
                ir = QtCore.QItemSelectionRange( index )
                itemsSelection.append(ir)
        self.tree.selectionModel().select(itemsSelection , flags)

        # set selection visible
        if len(row_selected)>=1:
            index = self.tree.model().index(row_selected[0],0,QtCore.QModelIndex())
            self.tree.scrollTo(index)

        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)        


    def open_context_menu(self):
        pass
        #~ menu = QtGui.QMenu()
        #~ act = menu.addAction('Move selection to trash')
        #~ act.triggered.connect(self.move_selection_to_trash)
        #~ menu.exec_(self.cursor().pos())
    
    def move_selection_to_trash(self):
        #TODO
        pass
        #~ self.cc.peak_label[self.cc.peak_selection] = -1
        #~ self.cc.on_new_cluster()
        #~ self.cc.refresh_colors(reset = False)
        #~ self.refresh()
        #~ self.spike_label_changed.emit()


class ClusterSpikeList(WidgetBase):
    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        self.table = QtGui.QTableWidget()
        self.layout.addWidget(self.table)
        self.table.itemChanged.connect(self.on_item_changed)
        
        self.refresh()

    def refresh(self):
        #~ self.cc._check_plot_attributes()
        
        self.table.itemChanged.disconnect(self.on_item_changed)
        
        self.table.clear()
        labels = ['label', 'show/hide', 'nb_peaks']
        self.table.setColumnCount(len(labels))
        self.table.setHorizontalHeaderLabels(labels)
        #~ self.table.setMinimumWidth(100)
        #~ self.table.setColumnWidth(0,60)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.open_context_menu)
        self.table.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        
        self.table.setRowCount(self.controller.cluster_labels.size)
        
        for i, k in enumerate(self.controller.cluster_labels):
            color = self.controller.qcolors.get(k, QtGui.QColor( 'white'))
            pix = QtGui.QPixmap(10,10)
            pix.fill(color)
            icon = QtGui.QIcon(pix)
            
            name = '{}'.format(k)
            item = QtGui.QTableWidgetItem(name)
            item.setFlags(QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable)
            self.table.setItem(i,0, item)
            item.setIcon(icon)
            
            item = QtGui.QTableWidgetItem('')
            item.setFlags(QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsUserCheckable)
            
            item.setCheckState({ False: QtCore.Qt.Unchecked, True : QtCore.Qt.Checked}[self.controller.cluster_visible[k]])
            self.table.setItem(i,1, item)

            item = QtGui.QTableWidgetItem('{}'.format(self.controller.cluster_count[k]))
            item.setFlags(QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable)
            self.table.setItem(i,2, item)
        
        for i in range(3):
            self.table.resizeColumnToContents(i)
        self.table.itemChanged.connect(self.on_item_changed)        

    def on_item_changed(self, item):
        if item.column() != 1: return
        sel = {QtCore.Qt.Unchecked : False, QtCore.Qt.Checked : True}[item.checkState()]
        k = self.controller.cluster_labels[item.row()]
        self.controller.cluster_visible[k] = bool(item.checkState())
        self.cluster_visibility_changed.emit()
    
    def selected_cluster(self):
        selected = []
        for index in self.table.selectedIndexes():
            if index.column() !=0: continue
            selected.append(self.controller.cluster_labels[index.row()])
        return selected
    
    def _selected_spikes(self):
        selection = np.zeros(self.controller.spike_label.shape[0], dtype = bool)
        for k in self.selected_cluster():
            selection |= self.controller.spike_label == k
        return selection
    
    def open_context_menu(self):
        n = len(self.selected_cluster())
        menu = QtGui.QMenu()

        if n>=0: 
            act = menu.addAction('Reset colors')
            act.triggered.connect(self.reset_colors)
            act = menu.addAction('Show all')
            act.triggered.connect(self.show_all)
            act = menu.addAction('Hide all')
            act.triggered.connect(self.hide_all)
            #~ act = menu.addAction('Order cluster by power')
            #~ act.triggered.connect(self.order_clusters)
            
        if n>=1:
            #~ act = menu.addAction('Move selection to trash')
            #~ act.triggered.connect(self.move_selection_to_trash)
            #~ act = menu.addAction('Merge selection')
            #~ act.triggered.connect(self.merge_selection)
            act = menu.addAction('Select')
            act.triggered.connect(self.select_peaks_of_clusters)
        
        self.menu = menu
        menu.popup(self.cursor().pos())
        #~ menu.exec_(self.cursor().pos())
    
    def reset_colors(self):
        self.controller.refresh_colors(reset = True)
        self.refresh()
        self.colors_changed.emit()
    
    def show_all(self):
        for k in self.controller.cluster_visible:
            self.controller.cluster_visible[k] = True
        self.refresh()
        self.cluster_visibility_changed.emit()
    
    def hide_all(self):
        for k in self.controller.cluster_visible:
            self.controller.cluster_visible[k] = False
        self.refresh()
        self.cluster_visibility_changed.emit()
    

    def move_selection_to_trash(self):
        pass
        #~ for k in self.selected_cluster():
            #~ mask = self.controller.spike_label == k
            #~ self.controller.change_spike_label(mask, -1)
        #~ self.refresh()
        #~ self.spike_label_changed.emit()
    
    def merge_selection(self):
        pass
        #~ label_to_merge = self.selected_cluster()
        #~ self.controller.merge_cluster(label_to_merge)
        #~ self.refresh()
        #~ self.spike_label_changed.emit()
    
    def select_peaks_of_clusters(self):
        pass
        #TODO
        #~ self.controller.spike_selection[:] = self._selected_spikes()
        #~ self.refresh()
        #~ self.spike_selection_changed.emit()



