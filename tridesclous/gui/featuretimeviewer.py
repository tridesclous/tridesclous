from .myqt import QT
import pyqtgraph as pg

import numpy as np
import matplotlib.cm
import matplotlib.colors


from .base import WidgetBase
from .tools import ParamDialog
from ..tools import median_mad

import time

#~ enableMenu
class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    gain_zoom = QT.pyqtSignal(float)
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    #~ def wheelEvent(self, ev, axis=None):
        #~ if ev.modifiers() == QT.Qt.ControlModifier:
            #~ z = 10 if ev.delta()>0 else 1/10.
        #~ else:
            #~ z = 1.3 if ev.delta()>0 else 1/1.3
        #~ self.gain_zoom.emit(z)
        #~ ev.accept()
    def raiseContextMenu(self, ev):
        #for some reasons enableMenu=False is not taken (bug ????)
        pass


class FeatureTimeViewer(WidgetBase):

    _params = [
                    {'name': 'metric', 'type': 'list', 'values' : ['extremum_amplitude', 'geometry0', 'geometry1']},
                    {'name': 'alpha', 'type': 'float', 'value' : 0.5, 'limits':(0, 1.), 'step':0.05 },
                    {'name': 'spread', 'type': 'float', 'value' : 2.5, 'limits':(0, 50.), 'step':0.1 },
        ]
    
    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)

        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.combo_seg = QT.QComboBox()
        self.layout.addWidget(self.combo_seg)
        self.combo_seg.addItems([ 'Segment {}'.format(seg_num) for seg_num in range(self.controller.dataio.nb_segment) ])
        self._seg_pos = 0
        self.seg_num = self._seg_pos
        self.combo_seg.currentIndexChanged.connect(self.refresh)
        
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        
        self.create_settings()
        
        self.initialize_plot()
        self.similarity = None
        
        self.on_params_changed()#this do refresh
    
    
    def on_params_changed(self, ): #params, changes
        self.refresh()
        
    
    
    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        #~ self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons() 


    def refresh(self):
        
        self.plot.clear()
        if self.controller.some_peaks_index is None:
            return
        
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in cluster_visible.items() if v ]

        seg_index =  self.combo_seg.currentIndex()
        

        if self.controller.spikes is None:
            return
        
        some_spikes = self.controller.spikes[self.controller.some_peaks_index]
        selected = some_spikes['segment'] == seg_index
        if len(selected)==0:
            return
        spikes = some_spikes[selected]
        all_times = spikes['index'].astype('float64') / self.controller.dataio.sample_rate
        
        #~ selected = self.controller.spike_segment[self.controller.some_peaks_index]==seg_index
        #~ all_index = self.controller.spike_index[self.controller.some_peaks_index][selected]
        #~ all_times = all_index.astype('float64')/self.controller.dataio.sample_rate
        #~ all_labels = self.controller.spike_label[self.controller.some_peaks_index][selected]
        
        #TODO if None
        # this is a hack to speedup some_waveforms[selected]
        # because boolean selection is slow here ???
        #~ ind_selected, = np.nonzero(selected)
        #~ selected_slice = slice(np.min(ind_selected), np.max(ind_selected)+1)
        
        if self.params['metric'] == 'extremum_amplitude':
            #~ extremum_amplitude = spikes['extremum_amplitude']
            values = spikes['extremum_amplitude']
            
        elif self.params['metric'] == 'geometry0':
            geometry0 = self.controller.geometry[:, 0]
            values = geometry0[spikes['channel']]
            values += np.random.randn(values.size) * self.params['spread']
        elif self.params['metric'] == 'geometry1':
            geometry0 = self.controller.geometry[:, 1]
            values = geometry0[spikes['channel']]
            values += np.random.randn(values.size) * self.params['spread']
            
        else:
            raise NotImplementedError


        for k in visibles:
            #~ self.controller.some
            keep = spikes['cluster_label'] == k
            
            x = all_times[keep]
            y = values[keep]
            
            color = QT.QColor(self.controller.qcolors.get(k, QT.QColor( 'white')))
            color.setAlpha(int(self.params['alpha']*255))
            curve = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(color, width=1), brush=color)
            
            self.plot.addItem(curve)
            #~ self.curves.append(curve)
        
        
        self.plot.setXRange(0, all_times[-1])
        #~ if self.params['metric'] == 'extremum_amplitude':
            #~ self.plot.setYRange(-30, 30)
        #~ else:
        self.plot.setYRange(np.min(values), np.max(values))

    def on_spike_selection_changed(self):
        pass

    def on_spike_label_changed(self):
        self.refresh()
        
    def on_colors_changed(self):
        self.refresh()
    
    def on_cluster_visibility_changed(self):
        self.refresh()
    
    def on_cluster_tag_changed(self):
        pass



