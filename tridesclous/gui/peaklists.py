from .myqt import QT
import pyqtgraph as pg

import numpy as np

from .. import labelcodes
from .base import WidgetBase
from .baselist import ClusterBaseList

from .tools import ParamDialog, open_dialog_methods
from . import gui_params


class PeakModel(QT.QAbstractItemModel):
    def __init__(self, parent =None, controller=None):
        QT.QAbstractItemModel.__init__(self,parent)
        self.controller = controller
        self.refresh_colors()
    
    def columnCount(self , parentIndex):
        return 6
        
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
        peak_chan = self.controller.spike_channel[abs_ind]
        peak_ampl = self.controller.spikes[abs_ind]['extremum_amplitude']
        
        
        if role ==QT.Qt.DisplayRole :
            if col == 0:
                return '{}'.format(abs_ind)
            elif col == 1:
                return '{}'.format(seg_num)
            elif col == 2:
                return '{:.4f}'.format(peak_time)
            elif col == 3:
                return '{}'.format(peak_label)
            elif col == 4:
                return '{}'.format(peak_chan)
            elif col == 5:
                return '{:.1f}'.format(peak_ampl)
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
            return  ['num', 'seg_num', 'time', 'cluster_label', 'channel', 'amplitude'][section]
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
    """
    **Peak List** show all detected peak for the catalogue construction.
    
    Here pintentionally peaks are not spikes already (even most of them are spikes)
    because supperposition of spikes are done here in catalogue in Peeler.
    
    Note:
      * If there are to many peaks, not all of them will have a extracted waveform.
        This why some peak are not labeled (-10) and nb_peak != nb_wveforms
        sometimes.
      * Peaks can belong to diffrents segment, a column indicate it. This is th full list
        of all peaks of all segment.
      * A right click open a ontext menu:
        * move one or several selected spike to trash
        * create a new cluster with one or several spikes
      * When you select one spike, this will auto zoom on **Trace View**,  auto select
        the appriopriate segment and hilight the point on **ND Scatetr**. And vice versa.
    """
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

        # This is very slow!!!!!
        #~ for i in range(self.model.columnCount(None)):
            #~ print(i)
            #~ self.tree.resizeColumnToContents(i)
        self.tree.setColumnWidth(0,80)
        
        self.refresh()
    
    def refresh(self):
        self.model.refresh_colors()
        nb_peak = self.controller.spikes.size
        if self.controller.some_peaks_index is not None:
            nb_sampled = self.controller.some_peaks_index.shape[0]
        else:
            nb_sampled = 0
        self.label_title.setText('<b>All peaks {} - Nb sampled {}</b>'.format(nb_peak, nb_sampled))
    
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
        act = menu.addAction('Move peak selection to trash')
        act.triggered.connect(self.move_peak_selection_to_trash)
        act = menu.addAction('Make cluster with selection')
        act.triggered.connect(self.make_new_cluster)

        ##menu.popup(self.cursor().pos())
        menu.exec_(self.cursor().pos())
    
    def move_peak_selection_to_trash(self):
        self.controller.change_spike_label(self.controller.spike_selection, -1)
        self.refresh()
        self.spike_label_changed.emit()

    def make_new_cluster(self):
        self.controller.change_spike_label(self.controller.spike_selection, max(self.controller.cluster_labels)+1)
        self.refresh()
        self.spike_label_changed.emit()
    
    
class ClusterPeakList(ClusterBaseList):
    """
    **Cluster list** is the central widget for actions for clusters :
    make them visible, merge, trash, sort, split, change color, ...
    
    A context menu with right propose:
      * **Reset colors**
      * **Show all**
      * **Hide all**
      * **Re-label cluster by rms**: this re order cluster so that 0 have the bigger rms
         and N the lowest.
      * **Feature projection with all**: this re compute feature projection (equivalent to left toolbar)
      * **Feature projection with selection**: this re compute feature projection but take
        in account only selected usefull when you have doubt on small cluster and want a specifit
        PCA because variance is absord by big ones.
      * **Move selection to trash**
      * **Merge selection**: re label spikes in the same cluster.
      * **Select on peak list**: a spike  as selected for theses clusters.
      * **Tag selection as same cell**: in case of burst some cell
        can have diffrent waveform shape leading to diferents cluster but
        with same ratio. If you have that do not merge clusters because the
        Peeler wll fail. Prefer tag 2 cluster as same cell.
      * **Split selection**: try to split only selected cluster.
    
    Double click on a row make invisible all others except this one.
    
    Cluster can be visualy ordered by some criteria (rms, amplitude, nb peak, ...)
    This is useless to explore cluster few peaks, or big amplitudes, ...
    
    Negative labels are reserved:
      * -1 is thrash
      * -2 is noise snippet
      * -10 unclassified (because no waveform associated)
      * -9 Alien

    """
    _special_label = sorted(list(labelcodes.to_name.keys()))

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
        
        act = self.menu.addAction('Feature projection with all')
        act.triggered.connect(self.pc_project_all)
        act = self.menu.addAction('Feature projection with selection')
        act.triggered.connect(self.pc_project_selection)
        act = self.menu.addAction('Move cluster selection to trash')
        act.triggered.connect(self.move_cluster_selection_to_trash)
        act = self.menu.addAction('Merge selection')
        act.triggered.connect(self.merge_selection)
        act = self.menu.addAction('Select on peak list')
        act.triggered.connect(self.select_peaks_of_clusters)
        act = self.menu.addAction('Tag selected clusters as same cell')
        act.triggered.connect(self.selection_tag_same_cell)
        
        act = self.menu.addAction('Split selection')
        act.triggered.connect(self.split_selection)

        act = self.menu.addAction('Change color/annotation/tag')
        act.triggered.connect(self.change_color_annotation_tag)

    def _selected_spikes(self):
        selection = np.zeros(self.controller.spike_label.shape[0], dtype = bool)
        for k in self.selected_cluster():
            selection |= self.controller.spike_label == k
        return selection
    
    def reset_colors(self):
        self.controller.refresh_colors(reset = True)
        self.refresh()
        self.colors_changed.emit()
    
    def order_clusters(self):
        self.controller.order_clusters()
        self.refresh()
        self.spike_label_changed.emit()
    
    def pc_project_all(self, selection=None):
        
        params = gui_params.features_params_by_methods
        if selection is not None:
            params = params.copy()
            params['global_lda'] = []
        
        method, kargs = open_dialog_methods(params, self)
        
        if method is None:
            return
        
        self.controller.extract_some_features(method=method, selection=selection, **kargs)
        self.refresh()
        self.spike_label_changed.emit()
    
    def pc_project_selection(self):
        print('pc_project_selection', np.sum(self._selected_spikes()))
        self.pc_project_all(selection=self._selected_spikes())
    
    def move_cluster_selection_to_trash(self):
        mask = np.zeros(self.controller.spike_label.size, dtype='bool')
        for k in self.selected_cluster():
            #~ mask = self.controller.spike_label == k
            mask |= self.controller.spike_label == k
        self.controller.change_spike_label(mask, labelcodes.LABEL_TRASH)
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
        
        self.controller.split_cluster(label_to_split, method=method,  **kargs) #order_clusters=True,
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
    
    def change_color_annotation_tag(self):
        labels = self.selected_cluster()
        n = len(labels)
        if n!=1: return
        k = labels[0]
        clusters = self.controller.clusters
        ## ind = np.searchsorted(clusters['cluster_label'], k)  ## wrong because searchsortedmust be ordered
        ind = np.nonzero(clusters['cluster_label'] == k)[0][0]
        
        color = QT.QColor(self.controller.qcolors[k])
        annotations = str(clusters[ind]['annotations'])
        tag = str(clusters[ind]['tag'])
        
        params_ = [
            {'name': 'color', 'type': 'color', 'value': color},
            {'name': 'annotations', 'type': 'str', 'value': annotations},
            {'name': 'tag', 'type': 'list', 'value': tag, 'limits':gui_params.possible_tags},
        ]        
        
        dia = ParamDialog(params_, title = 'Cluster {}'.format(k), parent=self)
        if dia.exec_():
            d = dia.get()
            self.controller.set_cluster_attributes(k, **d)
            
            self.colors_changed.emit()
            self.refresh()

