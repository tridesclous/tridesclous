from .myqt import QT
import pyqtgraph as pg

import numpy as np
import matplotlib.cm
import matplotlib.colors


from .base import WidgetBase
from .tools import ParamDialog
from ..tools import median_mad

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
        


class FeatureTimeViewer(WidgetBase):

    _params = [
                    {'name': 'metric', 'type': 'list', 'values' : ['max_peak_value', 'feat_0'] },
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
        
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in cluster_visible.items() if v ]

        
        seg_index =  self.combo_seg.currentIndex()
        
        selected = self.controller.spike_segment[self.controller.some_peaks_index]==seg_index
        all_index = self.controller.spike_index[self.controller.some_peaks_index][selected]
        all_times = all_index.astype('float64')/self.controller.dataio.sample_rate
        all_labels = self.controller.spike_label[self.controller.some_peaks_index][selected]
        
        #TODO if None
        if self.params['metric'] == 'max_peak_value':
            if self.controller.some_waveforms is None:
                return
            else:
                all_waveforms = self.controller.some_waveforms[selected]
        if self.params['metric'] == 'feat_0':
            if self.controller.some_features is None:
                return
            else:
                all_features = self.controller.some_features[selected]
        
        #~ print(all_waveforms.shape)
        

        d = self.controller.info['params_waveformextractor']
        n_left, n_right = d['n_left'], d['n_right']
        
        for k in visibles:
            #~ self.controller.some
            keep = all_labels==k
            
            x = all_times[keep]
            
            if self.params['metric'] == 'max_peak_value':
                c = self.controller.get_max_on_channel(k)
                if c is None:
                    continue
                y = all_waveforms[keep, -n_left, c]
            elif self.params['metric'] == 'feat_0':
                y = all_features[keep, 0]
            
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            
            curve = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(color, width=2), brush=color)
            self.plot.addItem(curve)
            #~ self.curves.append(curve)
        
        
        
        self.plot.setXRange(0, all_times[-1])
        if self.params['metric'] == 'max_peak_value':
            self.plot.setYRange(-30, 30)
        else:
            self.plot.setYRange(min(y), max(y))

