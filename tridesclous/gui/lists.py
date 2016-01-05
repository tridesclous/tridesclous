import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .base import WidgetBase
from .tools import ParamDialog


class PeakModel(QtCore.QAbstractItemModel):
    def __init__(self, parent =None, spikesorter = None):
        QtCore.QAbstractItemModel.__init__(self,parent)
        self.spikesorter = spikesorter
        self.io = self.spikesorter.dataio
        self.refresh_colors()
    
    def columnCount(self , parentIndex):
        return 4
        
    def rowCount(self, parentIndex):
        if not parentIndex.isValid() and self.spikesorter.peak_labels is not None:
            v = self.spikesorter.cluster_visible[self.spikesorter.cluster_visible]
            self.visible_peak_labels = self.spikesorter.peak_labels[self.spikesorter.peak_labels.isin(v.index)]
            return self.visible_peak_labels.shape[0]
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
        ind = self.visible_peak_labels.index[row]
        label =  self.visible_peak_labels.iloc[row]
        
        if role ==QtCore.Qt.DisplayRole :
            if col == 0:
                return '{}'.format(row)
            elif col == 1:
                return '{}'.format(int(ind[0]))
            elif col == 2:
                return '{:.4f}'.format(ind[1])
            elif col == 3:
                return '{}'.format(label)
            else:
                return None
        elif role == QtCore.Qt.DecorationRole :
            if col != 0: return None
            if label in self.icons:
                return self.icons[label]
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
            return  ['num', 'seg_num', 'time', 'cluster_label'][section]
        return
    
    def refresh_colors(self):
        self.icons = { }
        for k in self.spikesorter.qcolors:
            color = self.spikesorter.qcolors.get(k, QtGui.QColor( 'white'))
            pix = QtGui.QPixmap(10,10 )
            pix.fill(color)
            self.icons[k] = QtGui.QIcon(pix)
        
        #~ self.icons[-1] = QIcon(':/user-trash.png')
        
        self.layoutChanged.emit()
        
        
class PeakList(WidgetBase):
    def __init__(self, spikesorter = None, parent=None):
        WidgetBase.__init__(self, parent)
        
        self.spikesorter = spikesorter
        self.dataio = self.spikesorter.dataio
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.layout.addWidget(QtGui.QLabel('<b>All spikes</b>') )
        
        self.tree = QtGui.QTreeView(minimumWidth = 100, uniformRowHeights = True,
                    selectionMode= QtGui.QAbstractItemView.ExtendedSelection, selectionBehavior = QtGui.QTreeView.SelectRows,
                    contextMenuPolicy = QtCore.Qt.CustomContextMenu,)
        
        self.layout.addWidget(self.tree)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        
        self.model = PeakModel(spikesorter = spikesorter)
        self.tree.setModel(self.model)
        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)

        for i in range(self.model.columnCount(None)):
            self.tree.resizeColumnToContents(i)
        self.tree.setColumnWidth(0,80)
    
    def refresh(self):
        self.model.refresh_colors()
    
    def on_tree_selection(self):
        self.spikesorter.peak_selection[:] = False
        for index in self.tree.selectedIndexes():
            if index.column() == 0:
                ind = self.model.visible_peak_labels.index[index.row()]
                self.spikesorter.peak_selection.loc[ind] = True
        self.peak_selection_changed.emit()
    
    def on_peak_selection_changed(self):
        self.tree.selectionModel().selectionChanged.disconnect(self.on_tree_selection)
        
        v = self.spikesorter.cluster_visible[self.spikesorter.cluster_visible]
        selected_peaks = self.spikesorter.peak_selection.iloc[self.spikesorter.peak_labels.isin(v.index).values]
        selected_peaks = selected_peaks[selected_peaks]
        
        if selected_peaks.shape[0]>100:#otherwise this is verry slow
            selected_peaks = selected_peaks.iloc[:10]
        rows = [self.model.visible_peak_labels.index.get_loc(ind) for ind in selected_peaks.index]
        
        # change selection
        self.tree.selectionModel().clearSelection()
        flags = QtGui.QItemSelectionModel.Select #| QItemSelectionModel.Rows
        itemsSelection = QtGui.QItemSelection()
        for r in rows:
            for c in range(2):
                index = self.tree.model().index(r,c,QtCore.QModelIndex())
                ir = QtGui.QItemSelectionRange( index )
                itemsSelection.append(ir)
        self.tree.selectionModel().select(itemsSelection , flags)

        # set selection visible
        if len(rows)>=1:
            index = self.tree.model().index(rows[0],0,QtCore.QModelIndex())
            self.tree.scrollTo(index)

        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)        


    def open_context_menu(self):
        menu = QtGui.QMenu()
        act = menu.addAction('Move selection to trash')
        act.triggered.connect(self.move_selection_to_trash)
        ##menu.popup(self.cursor().pos())
        menu.exec_(self.cursor().pos())
    
    def move_selection_to_trash(self):
        self.spikesorter.peak_labels[self.spikesorter.peak_selection] = -1
        self.spikesorter.on_new_cluster()
        self.spikesorter.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()


class ClusterList(WidgetBase):
    
    def __init__(self, spikesorter = None, parent=None):
        WidgetBase.__init__(self, parent)
        
        self.spikesorter = spikesorter
        self.dataio = self.spikesorter.dataio
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        self.table = QtGui.QTableWidget()
        self.layout.addWidget(self.table)
        self.table.itemChanged.connect(self.on_item_changed)
        
        self.refresh()

    def refresh(self):
        self.table.itemChanged.disconnect(self.on_item_changed)
        sps = self.spikesorter
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
        
        self.table.setRowCount(self.spikesorter.cluster_labels.size)
        
        for i, k in enumerate(self.spikesorter.cluster_labels):
            color = self.spikesorter.qcolors.get(k, QtGui.QColor( 'white'))
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
            item.setCheckState({ False: QtCore.Qt.Unchecked, True : QtCore.Qt.Checked}[self.spikesorter.cluster_visible[k]])
            self.table.setItem(i,1, item)

            item = QtGui.QTableWidgetItem('{}'.format(self.spikesorter.cluster_count[k]))
            item.setFlags(QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable)
            self.table.setItem(i,2, item)
        
        for i in range(3):
            self.table.resizeColumnToContents(i)
        self.table.itemChanged.connect(self.on_item_changed)        

    def on_item_changed(self, item):
        if item.column() != 1: return
        sel = {QtCore.Qt.Unchecked : False, QtCore.Qt.Checked : True}[item.checkState()]
        k = self.spikesorter.cluster_labels[item.row()]
        self.spikesorter.cluster_visible[k] = item.checkState()
        self.cluster_visibility_changed.emit()
    
    def selected_cluster(self):
        selected = []
        for index in self.table.selectedIndexes():
            if index.column() !=0: continue
            selected.append(self.spikesorter.cluster_labels[index.row()])
        return selected

    def open_context_menu(self):
        n = len(self.selected_cluster())
        menu = QtGui.QMenu()

        if n>=0: 
            act = menu.addAction('Reset colors')
            act.triggered.connect(self.reset_colors)
            act = menu.addAction('Sort by ascending waveform power')
            act.triggered.connect(self.sort_clusters)
            act = menu.addAction('Show all')
            act.triggered.connect(self.show_all)
            act = menu.addAction('Hide all')
            act.triggered.connect(self.hide_all)
            act = menu.addAction('Order cluster by power')
            act.triggered.connect(self.order_clusters)
            
        if n>=1:
            act = menu.addAction('PCA projection with all')
            act.triggered.connect(self.pca_project_all)
            act = menu.addAction('PCA projection with selection')
            act.triggered.connect(self.pca_project_selection)
            act = menu.addAction('Move selection to trash')
            act.triggered.connect(self.move_selection_to_trash)
            act = menu.addAction('Merge selection')
            act.triggered.connect(self.merge_selection)
            act = menu.addAction('Select')
            act.triggered.connect(self.select_peaks_of_clusters)
        
        if n == 1:
            act = menu.addAction('Split selection')
            act.triggered.connect(self.split_selection)
        
        self.menu = menu
        menu.popup(self.cursor().pos())
        #~ menu.exec_(self.cursor().pos())
        
    
    def reset_colors(self):
        self.spikesorter.refresh_colors(reset = True)
        self.refresh()
        self.colors_changed.emit()
    
    def sort_clusters(self):
        pass
        
    def show_all(self):
        self.spikesorter.cluster_visible[:] = True
        self.refresh()
        self.cluster_visibility_changed.emit()
    
    def hide_all(self):
        self.spikesorter.cluster_visible[:] = False
        self.refresh()
        self.cluster_visibility_changed.emit()
    
    def order_clusters(self):
        self.spikesorter.order_clusters()
        self.spikesorter.on_new_cluster()
        self.spikesorter.refresh_colors(reset = True)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def pca_project_all(self):
        self.spikesorter.clustering.project(method = 'pca', n_components = self.spikesorter.clustering._pca.n_components)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def pca_project_selection(self):
        selection = np.zeros(self.spikesorter.peak_labels.shape[0], dtype = bool)
        for k in self.selected_cluster():
            selection |= self.spikesorter.peak_labels == k
        self.spikesorter.clustering.project(method = 'pca', n_components = self.spikesorter.clustering._pca.n_components,
                                        selection = selection)
        self.peak_cluster_changed.emit()
    
    def move_selection_to_trash(self):
        for k in self.selected_cluster():
            take = self.spikesorter.peak_labels == k
            self.spikesorter.peak_labels[take] = -1
        self.spikesorter.on_new_cluster()
        self.spikesorter.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def merge_selection(self):
        new_label = max(self.spikesorter.cluster_labels)+1
        for k in self.selected_cluster():
            take = self.spikesorter.peak_labels == k
            self.spikesorter.peak_labels[take] = new_label
        self.spikesorter.on_new_cluster()
        self.spikesorter.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def split_selection(self):
        k = self.selected_cluster()[0]
        
        _params = [{'name' : 'method', 'type' : 'list', 'values' : ['kmeans', 'gmm']}]
        dialog1 = ParamDialog(_params, title = 'Which method ?', parent = self)
        if not dialog1.exec_():
            return

        method = dialog1.params['method']
        
        if  method=='kmeans':
            _params =  [{'name' : 'n', 'type' : 'int', 'value' : 2}]
            dialog2 = ParamDialog(_params, title = 'kmeans parameters', parent = self)
            if not dialog2.exec_():
                return
            kargs = dialog2.get()
        
        elif method=='gmm':
            _params =  [{'name' : 'n', 'type' : 'int', 'value' : 2},
                                {'name' : 'covariance_type', 'type' : 'list', 'values' : ['full']},
                                {'name' : 'n_init', 'type' : 'int', 'value' : 10},
                                ]
            dialog2 = ParamDialog(_params, title = 'kmeans parameters', parent = self)
            if not dialog2.exec_():
                return
            kargs = dialog2.get()
        
        n = kargs.pop('n')
        self.spikesorter.clustering.split_cluster(k, n, method=method, order_clusters = True, **kargs)

        self.spikesorter.on_new_cluster()
        self.spikesorter.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()

    
    def select_peaks_of_clusters(self):
        self.spikesorter.peak_selection[:] = False
        for k in self.selected_cluster():
            self.spikesorter.peak_selection[self.spikesorter.peak_labels == k] = True
            
        self.refresh()
        self.peak_selection_changed.emit()

