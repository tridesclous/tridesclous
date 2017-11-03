from .myqt import QT
import pyqtgraph as pg

import numpy as np

from .. import labelcodes
from .base import WidgetBase
from .tools import ParamDialog, open_dialog_methods
from . import gui_params


class PeakModel(QT.QAbstractItemModel):
    def __init__(self, parent =None, controller=None):
        QT.QAbstractItemModel.__init__(self,parent)
        self.controller = controller
        self.refresh_colors()
    
    def columnCount(self , parentIndex):
        return 4
        
    def rowCount(self, parentIndex):
        if not parentIndex.isValid() and self.controller.spike_label is not None:
            self.visible_ind, = np.nonzero(self.controller.spike_visible)
            n = self.visible_ind.size
            return n
        else :
            return 0
        
    def index(self, row, column, parentIndex):
        if not parentIndex.isValid():
            if column==0:
                childItem = row
            return self.createIndex(row, column, None)
        else:
            return QT.QModelIndex()
    
    def parent(self, index):
        return QT.QModelIndex()
    
    def data(self, index, role):
        if not index.isValid():
            return None
        
        if role not in (QT.Qt.DisplayRole, QT.Qt.DecorationRole):
            return
        
        col = index.column()
        row = index.row()
        #~ ind = self.visible_peak_labels.index[row]
        #~ label =  self.visible_peak_labels.iloc[row]
        #~ t_start = 0.
        
        abs_ind = self.visible_ind[row]
        
        seg_num = self.controller.spike_segment[abs_ind]
        peak_pos = self.controller.spike_index[abs_ind]
        peak_time = peak_pos/self.controller.dataio.sample_rate
        peak_label = self.controller.spike_label[abs_ind]
        
        if role ==QT.Qt.DisplayRole :
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
        elif role == QT.Qt.DecorationRole :
            if col != 0: return None
            if peak_label in self.icons:
                return self.icons[peak_label]
            else:
                return None
        else :
            return None
    
    def flags(self, index):
        if not index.isValid():
            return QT.Qt.NoItemFlags
        return QT.Qt.ItemIsEnabled | QT.Qt.ItemIsSelectable #| Qt.ItemIsDragEnabled

    def headerData(self, section, orientation, role):
        if orientation == QT.Qt.Horizontal and role == QT.Qt.DisplayRole:
            return  ['num', 'seg_num', 'time', 'cluster_label'][section]
        return
    
    def refresh_colors(self):
        self.icons = { }
        for k in self.controller.qcolors:
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            pix = QT.QPixmap(10,10 )
            pix.fill(color)
            self.icons[k] = QT.QIcon(pix)
        
        #~ self.icons[-1] = QIcon(':/user-trash.png')
        
        self.layoutChanged.emit()
        
        
class PeakList(WidgetBase):
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label_title = QT.QLabel('')
        self.layout.addWidget(self.label_title)
        
        self.tree = QT.QTreeView(minimumWidth = 100, uniformRowHeights = True,
                    selectionMode= QT.QAbstractItemView.ExtendedSelection, selectionBehavior = QT.QTreeView.SelectRows,
                    contextMenuPolicy = QT.Qt.CustomContextMenu,)
        
        self.layout.addWidget(self.tree)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        
        self.model = PeakModel(controller = controller)
        self.tree.setModel(self.model)
        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)

        for i in range(self.model.columnCount(None)):
            self.tree.resizeColumnToContents(i)
        self.tree.setColumnWidth(0,80)
        
        self.refresh()
    
    def refresh(self):
        self.model.refresh_colors()
        nb_peak = self.controller.spikes.size
        if self.controller.some_waveforms is not None:
            nb_wf = self.controller.some_waveforms.shape[0]
        else:
            nb_wf = 0
        self.label_title.setText('<b>All peaks {} - Nb waveforms {}</b>'.format(nb_peak, nb_wf))
    
    def on_tree_selection(self):
        self.controller.spike_selection[:] = False
        for index in self.tree.selectedIndexes():
            if index.column() == 0:
                ind = self.model.visible_ind[index.row()]
                self.controller.spike_selection[ind] = True
        self.spike_selection_changed.emit()
    
    def on_spike_selection_changed(self):
        self.tree.selectionModel().selectionChanged.disconnect(self.on_tree_selection)
        
        row_selected, = np.nonzero(self.controller.spike_selection[self.model.visible_ind])
        
        if row_selected.size>100:#otherwise this is verry slow
            row_selected = row_selected[:10]
        
        # change selection
        self.tree.selectionModel().clearSelection()
        flags = QT.QItemSelectionModel.Select #| QItemSelectionModel.Rows
        itemsSelection = QT.QItemSelection()
        for r in row_selected:
            for c in range(2):
                index = self.tree.model().index(r,c,QT.QModelIndex())
                ir = QT.QItemSelectionRange( index )
                itemsSelection.append(ir)
        self.tree.selectionModel().select(itemsSelection , flags)

        # set selection visible
        if len(row_selected)>=1:
            index = self.tree.model().index(row_selected[0],0,QT.QModelIndex())
            self.tree.scrollTo(index)

        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)        


    def open_context_menu(self):
        menu = QT.QMenu()
        act = menu.addAction('Move selection to trash')
        act.triggered.connect(self.move_selection_to_trash)
        act = menu.addAction('Make cluster with selection')
        act.triggered.connect(self.make_new_cluster)

        ##menu.popup(self.cursor().pos())
        menu.exec_(self.cursor().pos())
    
    def move_selection_to_trash(self):
        self.controller.change_spike_label(self.controller.spike_selection, -1)
        self.refresh()
        self.spike_label_changed.emit()

    def make_new_cluster(self):
        self.controller.change_spike_label(self.controller.spike_selection, max(self.controller.cluster_labels)+1)
        self.refresh()
        self.spike_label_changed.emit()
    
    


class ClusterPeakList(WidgetBase):
    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        h = QT.QHBoxLayout()
        self.layout.addLayout(h)
        h.addWidget(QT.QLabel('sort by'))
        self.combo_sort = QT.QComboBox()
        self.combo_sort.addItems(['label', 'max_on_channel', 'max_peak_amplitude', 'waveform_rms'])
        self.combo_sort.currentIndexChanged.connect(self.refresh)
        h.addWidget(self.combo_sort)
        h.addStretch()
        
        self.table = QT.QTableWidget()
        self.layout.addWidget(self.table)
        self.table.itemChanged.connect(self.on_item_changed)
        
        self.make_menu()
        
        self.refresh()

    def make_menu(self):
        self.menu = QT.QMenu()

        act = self.menu.addAction('Reset colors')
        act.triggered.connect(self.reset_colors)
        act = self.menu.addAction('Show all')
        act.triggered.connect(self.show_all)
        act = self.menu.addAction('Hide all')
        act.triggered.connect(self.hide_all)
        act = self.menu.addAction('Re-label cluster by rms')
        act.triggered.connect(self.order_clusters)
        
        act = self.menu.addAction('PC projection with all')
        act.triggered.connect(self.pc_project_all)
        act = self.menu.addAction('PC projection with selection')
        act.triggered.connect(self.pc_project_selection)
        act = self.menu.addAction('Move selection to trash')
        act.triggered.connect(self.move_selection_to_trash)
        act = self.menu.addAction('Merge selection')
        act.triggered.connect(self.merge_selection)
        act = self.menu.addAction('Select')
        act.triggered.connect(self.select_peaks_of_clusters)
        act = self.menu.addAction('Tag selection as same cell')
        act.triggered.connect(self.selection_tag_same_cell)
        
        act = self.menu.addAction('Split selection')
        act.triggered.connect(self.split_selection)        

    def refresh(self):
        self.table.itemChanged.disconnect(self.on_item_changed)
        
        self.table.clear()
        labels = ['cluster_label', 'show/hide', 'nb_peaks', 'max_on_channel', 'cell_label',]
        self.table.setColumnCount(len(labels))
        self.table.setHorizontalHeaderLabels(labels)
        #~ self.table.setMinimumWidth(100)
        #~ self.table.setColumnWidth(0,60)
        self.table.setContextMenuPolicy(QT.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.open_context_menu)
        self.table.setSelectionMode(QT.QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QT.QAbstractItemView.SelectRows)
        
        #~ cluster_labels = self.controller.cluster_labels
        #~ cluster_labels = [labelcodes.LABEL_NOISE] + cluster_labels.tolist()
        _special_label = [labelcodes.LABEL_UNCLASSIFIED, labelcodes.LABEL_NOISE, labelcodes.LABEL_TRASH]

        sort_mode = str(self.combo_sort.currentText())
        
        clusters = self.controller.clusters
        clusters = clusters[clusters['cluster_label']>=0]
        if sort_mode=='label':
            order =np.arange(clusters.size)
        elif sort_mode=='max_on_channel':
            order = np.argsort(clusters['max_on_channel'])
        elif sort_mode=='max_peak_amplitude':
            order = np.argsort(np.abs(clusters['max_peak_amplitude']))[::-1]
        elif sort_mode=='waveform_rms':
            order = np.argsort(clusters['waveform_rms'])[::-1]
        
        cluster_labels =_special_label + self.controller.positive_cluster_labels[order].tolist()
        
        self.table.setRowCount(len(cluster_labels))
        
        for i, k in enumerate(cluster_labels):
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            pix = QT.QPixmap(16,16)
            pix.fill(color)
            icon = QT.QIcon(pix)
            
            if k<0:
                name = '{} ({})'.format(k, labelcodes.to_name[k])
            else:
                name = '{}'.format(k)
            item = QT.QTableWidgetItem(name)
            item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
            self.table.setItem(i,0, item)
            item.setIcon(icon)
            
            item = QT.QTableWidgetItem('')
            item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable|QT.Qt.ItemIsUserCheckable)
            
            item.setCheckState({ False: QT.Qt.Unchecked, True : QT.Qt.Checked}[self.controller.cluster_visible.get(k, False)])
            self.table.setItem(i,1, item)
            item.label = k

            item = QT.QTableWidgetItem('{}'.format(self.controller.cluster_count.get(k, 0)))
            item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
            self.table.setItem(i,2, item)
            
            c = self.controller.get_max_on_channel(k)
            if c is not None:
                item = QT.QTableWidgetItem('{}: {}'.format(c, self.controller.channel_names[c]))
                item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
                self.table.setItem(i,3, item)
            
            if k>=0:
                cell_label = self.controller.cell_labels[self.controller.cluster_labels==k][0]
                item = QT.QTableWidgetItem('{}'.format(cell_label))
                item.setFlags(QT.Qt.ItemIsEnabled|QT.Qt.ItemIsSelectable)
                self.table.setItem(i,4, item)
            
        for i in range(5):
            self.table.resizeColumnToContents(i)
        self.table.itemChanged.connect(self.on_item_changed)        

    def on_item_changed(self, item):
        if item.column() != 1: return
        sel = {QT.Qt.Unchecked : False, QT.Qt.Checked : True}[item.checkState()]
        #~ k = self.controller.cluster_labels[item.row()]
        k = item.label
        self.controller.cluster_visible[k] = bool(item.checkState())
        self.cluster_visibility_changed.emit()
    
    def selected_cluster(self):
        selected = []
        #~ for index in self.table.selectedIndexes():
        for item in self.table.selectedItems():
            #~ if index.column() !=1: continue
            if item.column() != 1: continue
            #~ selected.append(self.controller.cluster_labels[index.row()])
            selected.append(item.label)
        return selected
    
    def _selected_spikes(self):
        selection = np.zeros(self.controller.spike_label.shape[0], dtype = bool)
        for k in self.selected_cluster():
            selection |= self.controller.spike_label == k
        return selection
    
    def open_context_menu(self):
        #~ n = len(self.selected_cluster())

        
        self.menu.popup(self.cursor().pos())
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
    
    def order_clusters(self):
        self.controller.order_clusters()
        self.controller.on_new_cluster()
        self.refresh()
        self.spike_label_changed.emit()
    
    def pc_project_all(self, selection=None):
        method, kargs = open_dialog_methods(gui_params.features_params_by_methods, self)
        
        if method is None: return
        
        self.controller.project(method=method, selection=selection, **kargs)
        self.refresh()
        self.spike_label_changed.emit()
    
    def pc_project_selection(self):
        self.pc_project_all(selection=self._selected_spikes())
    
    def move_selection_to_trash(self):
        for k in self.selected_cluster():
            mask = self.controller.spike_label == k
            self.controller.change_spike_label(mask, -1)
        self.refresh()
        self.spike_label_changed.emit()
    
    def merge_selection(self):
        label_to_merge = self.selected_cluster()
        self.controller.merge_cluster(label_to_merge)
        self.refresh()
        self.spike_label_changed.emit()
    
    def split_selection(self):
        #TODO bug when not n_clusters
        
        n = len(self.selected_cluster())
        if n!=1: return
        label_to_split = self.selected_cluster()[0]
        
        method, kargs = open_dialog_methods(gui_params.cluster_params_by_methods, self)
        
        if method is None: return
        
        n = kargs.pop('n_clusters')
        
        self.controller.split_cluster(label_to_split, n, method=method,  **kargs) #order_clusters=True,
        self.refresh()
        self.spike_label_changed.emit()

    def selection_tag_same_cell(self):
        labels_to_group = self.selected_cluster()
        self.controller.tag_same_cell(labels_to_group)
        self.refresh()
        self.cluster_tag_changed.emit()
    
    def select_peaks_of_clusters(self):
        self.controller.spike_selection[:] = self._selected_spikes()
        self.refresh()
        self.spike_selection_changed.emit()
