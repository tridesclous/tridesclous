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
        return 4
    
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
        seg_num = spike['segment']
        peak_pos = spike['index']
        peak_time = peak_pos/self.controller.dataio.sample_rate
        peak_label = spike['label']
 
        
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
        #TODO
        pass
        self.controller.spikes['selected'][:] = False
        for index in self.tree.selectedIndexes():
            if index.column() == 0:
                ind = self.model.visible_ind[index.row()]
                self.controller.spikes['selected'][ind] = True
        self.peak_selection_changed.emit()
    
    def on_peak_selection_changed(self):
        #TODO
        pass
        #~ self.tree.selectionModel().selectionChanged.disconnect(self.on_tree_selection)
        
        #~ row_selected, = np.nonzero(self.cc.peak_selection[self.model.visible_mask])
        
        #~ if row_selected.size>100:#otherwise this is verry slow
            #~ row_selected = row_selected[:10]
        
        #~ # change selection
        #~ self.tree.selectionModel().clearSelection()
        #~ flags = QtCore.QItemSelectionModel.Select #| QItemSelectionModel.Rows
        #~ itemsSelection = QtCore.QItemSelection()
        #~ for r in row_selected:
            #~ for c in range(2):
                #~ index = self.tree.model().index(r,c,QtCore.QModelIndex())
                #~ ir = QtCore.QItemSelectionRange( index )
                #~ itemsSelection.append(ir)
        #~ self.tree.selectionModel().select(itemsSelection , flags)

        #~ # set selection visible
        #~ if len(row_selected)>=1:
            #~ index = self.tree.model().index(row_selected[0],0,QtCore.QModelIndex())
            #~ self.tree.scrollTo(index)

        #~ self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)        


    def open_context_menu(self):
        menu = QtGui.QMenu()
        act = menu.addAction('Move selection to trash')
        act.triggered.connect(self.move_selection_to_trash)
        ##menu.popup(self.cursor().pos())
        menu.exec_(self.cursor().pos())
    
    def move_selection_to_trash(self):
        #TODO
        pass
        #~ self.cc.peak_label[self.cc.peak_selection] = -1
        #~ self.cc.on_new_cluster()
        #~ self.cc.refresh_colors(reset = False)
        #~ self.refresh()
        #~ self.peak_cluster_changed.emit()
