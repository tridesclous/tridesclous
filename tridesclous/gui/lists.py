import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .base import WidgetBase
from .tools import ParamDialog


class PeakModel(QtCore.QAbstractItemModel):
    def __init__(self, parent =None, catalogueconstructor=None):
        QtCore.QAbstractItemModel.__init__(self,parent)
        self.cc = self.catalogueconstructor = catalogueconstructor
        self.refresh_colors()
    
    def columnCount(self , parentIndex):
        return 4
        
    def rowCount(self, parentIndex):
        if not parentIndex.isValid() and self.cc.peak_labels is not None:
            visibles = np.array([k for k, v in self.cc.cluster_visible.items() if v ])
            self.visible_mask = np.in1d(self.cc.peak_labels, visibles)
            self.visible_ind, = np.nonzero(self.visible_mask)
            #~ self.visible_peak_labels = self.cc.peak_labels[self.visible_peak_mask]
            
            #~ v = self.cc.cluster_visible[self.cc.cluster_visible]
            #~ self.visible_peak_labels = self.cc.peak_labels[self.cc.peak_labels.isin(v.index)]
            #~ return self.visible_peak_labels.shape[0]
            return np.sum(self.visible_mask)
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
        #~ ind = self.visible_peak_labels.index[row]
        #~ label =  self.visible_peak_labels.iloc[row]
        #~ t_start = 0.
        
        abs_ind = self.visible_ind[row]
        
        seg_num = self.cc.peak_segment[abs_ind]
        peak_pos = self.cc.peak_pos[abs_ind]
        peak_time = peak_pos/self.cc.dataio.sample_rate
        peak_label = self.cc.peak_labels[abs_ind]
        
        if role ==QtCore.Qt.DisplayRole :
            if col == 0:
                return '{}'.format(abs_ind)
            elif col == 1:
                return '{}'.format(seg_num)
            elif col == 2:
                return '{:.4f}'.format(peak_time)
            elif col == 3:
                return '{}'.format(peak_label)
            else:
                return None
        elif role == QtCore.Qt.DecorationRole :
            if col != 0: return None
            if peak_label in self.icons:
                return self.icons[peak_label]
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
        for k in self.cc.qcolors:
            color = self.cc.qcolors.get(k, QtGui.QColor( 'white'))
            pix = QtGui.QPixmap(10,10 )
            pix.fill(color)
            self.icons[k] = QtGui.QIcon(pix)
        
        #~ self.icons[-1] = QIcon(':/user-trash.png')
        
        self.layoutChanged.emit()
        
        
class PeakList(WidgetBase):
    def __init__(self, catalogueconstructor=None, parent=None):
        WidgetBase.__init__(self, parent)
        
        self.cc = self.catalogueconstructor = catalogueconstructor
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.layout.addWidget(QtGui.QLabel('<b>All spikes</b>') )
        
        self.tree = QtGui.QTreeView(minimumWidth = 100, uniformRowHeights = True,
                    selectionMode= QtGui.QAbstractItemView.ExtendedSelection, selectionBehavior = QtGui.QTreeView.SelectRows,
                    contextMenuPolicy = QtCore.Qt.CustomContextMenu,)
        
        self.layout.addWidget(self.tree)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        
        self.model = PeakModel(catalogueconstructor = catalogueconstructor)
        self.tree.setModel(self.model)
        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)

        for i in range(self.model.columnCount(None)):
            self.tree.resizeColumnToContents(i)
        self.tree.setColumnWidth(0,80)
    
    def refresh(self):
        self.model.refresh_colors()
    
    def on_tree_selection(self):
        self.cc.peak_selection[:] = False
        for index in self.tree.selectedIndexes():
            if index.column() == 0:
                ind = self.model.visible_ind[index.row()]
                self.cc.peak_selection[ind] = True
        self.peak_selection_changed.emit()
    
    def on_peak_selection_changed(self):
        self.tree.selectionModel().selectionChanged.disconnect(self.on_tree_selection)
        
        row_selected, = np.nonzero(self.cc.peak_selection[self.model.visible_mask])
        
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
        menu = QtGui.QMenu()
        act = menu.addAction('Move selection to trash')
        act.triggered.connect(self.move_selection_to_trash)
        ##menu.popup(self.cursor().pos())
        menu.exec_(self.cursor().pos())
    
    def move_selection_to_trash(self):
        self.cc.peak_labels[self.cc.peak_selection] = -1
        self.cc.on_new_cluster()
        self.cc.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()


class ClusterList(WidgetBase):
    
    def __init__(self, catalogueconstructor=None, parent=None):
        WidgetBase.__init__(self, parent)
        
        self.cc = self.catalogueconstructor = catalogueconstructor
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        self.table = QtGui.QTableWidget()
        self.layout.addWidget(self.table)
        self.table.itemChanged.connect(self.on_item_changed)
        
        self.refresh()

    def refresh(self):
        self.cc._check_plot_attributes()
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
        
        self.table.setRowCount(self.cc.cluster_labels.size)
        
        for i, k in enumerate(self.cc.cluster_labels):
            color = self.cc.qcolors.get(k, QtGui.QColor( 'white'))
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
            
            item.setCheckState({ False: QtCore.Qt.Unchecked, True : QtCore.Qt.Checked}[self.cc.cluster_visible[k]])
            self.table.setItem(i,1, item)

            item = QtGui.QTableWidgetItem('{}'.format(self.cc.cluster_count[k]))
            item.setFlags(QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable)
            self.table.setItem(i,2, item)
        
        for i in range(3):
            self.table.resizeColumnToContents(i)
        self.table.itemChanged.connect(self.on_item_changed)        

    def on_item_changed(self, item):
        if item.column() != 1: return
        sel = {QtCore.Qt.Unchecked : False, QtCore.Qt.Checked : True}[item.checkState()]
        k = self.cc.cluster_labels[item.row()]
        self.cc.cluster_visible[k] = bool(item.checkState())
        self.cluster_visibility_changed.emit()
    
    def selected_cluster(self):
        selected = []
        for index in self.table.selectedIndexes():
            if index.column() !=0: continue
            selected.append(self.cc.cluster_labels[index.row()])
        return selected
    
    def _selected_spikes(self):
        selection = np.zeros(self.cc.peak_labels.shape[0], dtype = bool)
        for k in self.selected_cluster():
            selection |= self.cc.peak_labels == k
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
            act = menu.addAction('Order cluster by power')
            act.triggered.connect(self.order_clusters)
            
        if n>=1:
            act = menu.addAction('PC projection with all')
            act.triggered.connect(self.pc_project_all)
            act = menu.addAction('PC projection with selection')
            act.triggered.connect(self.pc_project_selection)
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
        self.cc.refresh_colors(reset = True)
        self.refresh()
        self.colors_changed.emit()
    
    def order_clusters(self):
        self.cc.order_clusters()
        self.cc.refresh_colors(reset = True)
        self.refresh()
        self.colors_changed.emit()
        
        
    def show_all(self):
        for k in self.cc.cluster_visible:
            self.cc.cluster_visible[k] = True
        self.refresh()
        self.cluster_visibility_changed.emit()
    
    def hide_all(self):
        for k in self.cc.cluster_visible:
            self.cc.cluster_visible[k] = False
        self.refresh()
        self.cluster_visibility_changed.emit()
    
    def order_clusters(self):
        self.cc.order_clusters()
        self.cc.on_new_cluster()
        self.cc.refresh_colors(reset = True)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def _dialog_methods(self, methods, _params_by_method):
        _params = [{'name' : 'method', 'type' : 'list', 'values' : methods}]
        dialog1 = ParamDialog(_params, title = 'Which method ?', parent = self)
        if not dialog1.exec_():
            return None, None

        method = dialog1.params['method']
        
        _params =  _params_by_method[method]
        dialog2 = ParamDialog(_params, title = '{} parameters'.format(method), parent = self)
        if not dialog2.exec_():
            return None, None
        kargs = dialog2.get()
        
        return method, kargs
        

    
    def pc_project_all(self, selection=None):

        
        methods = ['pca', 'peak_max']
        _params_by_method = {
            'pca' : [{'name' : 'n_components', 'type' : 'int', 'value' : 5},],
            'peak_max' : [],
        }
        method, kargs = self._dialog_methods(methods, _params_by_method)
        if method is None: return
        
        self.cc.project(method=method, selection=selection, **kargs)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def pc_project_selection(self):
        self.pc_project_all(selection=self._selected_spikes())
    
    def move_selection_to_trash(self):
        for k in self.selected_cluster():
            take = self.cc.peak_labels == k
            self.cc.peak_labels[take] = -1
        self.cc.on_new_cluster(label_changed=self.selected_cluster()+[-1])
        self.cc.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def merge_selection(self):
        label_to_merge = self.selected_cluster()
        self.cc.merge_cluster(label_to_merge, order_clusters =False)
        self.cc.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()
    
    def split_selection(self):
        label_to_split = self.selected_cluster()[0]
        
        methods = ['kmeans', 'gmm']
        _params_by_method = {
            'kmeans' : [{'name' : 'n', 'type' : 'int', 'value' : 2}],
            'gmm' : [{'name' : 'n', 'type' : 'int', 'value' : 2},
                                {'name' : 'covariance_type', 'type' : 'list', 'values' : ['full']},
                                {'name' : 'n_init', 'type' : 'int', 'value' : 10},],
        }
        method, kargs = self._dialog_methods(methods, _params_by_method)
        if method is None: return
        
        n = kargs.pop('n')
        
        self.cc.split_cluster(label_to_split, n, method=method, order_clusters=True, **kargs)
        self.cc.refresh_colors(reset = False)
        self.refresh()
        self.peak_cluster_changed.emit()

    
    def select_peaks_of_clusters(self):
        self.cc.peak_selection[:] = self._selected_spikes()
        self.refresh()
        self.peak_selection_changed.emit()
