from .myqt import QT
import pyqtgraph as pg

import numpy as np
import matplotlib.cm
import matplotlib.colors

import time

from .base import WidgetBase
from .tools import ParamDialog
from ..tools import median_mad
from .. import labelcodes

class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    gain_zoom = QT.pyqtSignal(float)
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() == QT.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()
    def raiseContextMenu(self, ev):
        #for some reasons enableMenu=False is not taken (bug ????)
        pass
        

class WaveformHistViewer(WidgetBase):
    """
    **Waveform histogram viewer** is also a important thing.
    
    It is equivalent to **Waveform veiwer** in **flatten** mode but with
    a 2d histogram that show the density (probability) of a cluster.
    So waveforms are flatten from (nb_peak, nb_sample, nb_channel) to
    (nb_peak, nb_channel*nb_sample) and are binarized on a 2d histogram.
    Then this is plotted as a map. The color code the density.
    
    This is the best friend to see if two cluster are well discrimitated somewhere or
    if one cluster must be split.
    
    Important:
      * use right click for X/Y zoom
      * use left clik to move
      * use **mouse wheel** for color zoom.Really important to play with this
        to discover low density
      * intentionnaly not all cluster are displayed other we see nothing. The best is to plot
        2 by 2. Furthermore it faster to plot with few cluster.
      * don't forget to display the **noise snippet** to validate that the mad is 1 for all channel.
    
    Settings:
      * **colormap** hot is good because loaw density are black like background.
      * **data** choose waveforms or features
      * **bin_min** y limts of histogram
      * **bin_max** y limts of histogram
      * **bin_size**
      * **display_threshold**
      * **max_label** maximum number of labels displayed simulteneously 
        (2 by default but you can set more)
    
    """
    _params = [
                      {'name': 'colormap', 'type': 'list', 'values' : ['hot', 'viridis', 'jet', 'gray',  ] },
                      {'name': 'data', 'type': 'list', 'values' : ['waveforms', 'features', ] },
                      {'name': 'bin_min', 'type': 'float', 'value' : -20. },
                      {'name': 'bin_max', 'type': 'float', 'value' : 8. },
                      {'name': 'bin_size', 'type': 'float', 'value' : .1 },
                      {'name': 'display_threshold', 'type': 'bool', 'value' : True },
                      {'name': 'max_label', 'type': 'int', 'value' : 2 },
                      ]
    
    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)

        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        h = QT.QHBoxLayout()
        self.layout.addLayout(h)
        but = QT.QPushButton('Show 1D dist', checkable=True)
        h.addWidget(but)
        but.clicked.connect(self.show_hide_1d_dist)
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)

        self.graphicsview2 = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview2)
        self.graphicsview2.hide()
        
        self.create_settings()
        
        self.initialize_plot()
        self.similarity = None
        
        self.on_params_changed()#this do refresh
    
    
    def on_params_changed(self, ): #params, changes
        #~ for param, change, data in changes:
            #~ if change != 'value': continue
            #~ if param.name()=='data':
        
        N = 512
        cmap_name = self.params['colormap']
        cmap = matplotlib.cm.get_cmap(cmap_name , N)
        
        lut = []
        for i in range(N):
            r,g,b,_ =  matplotlib.colors.ColorConverter().to_rgba(cmap(i))
            lut.append([r*255,g*255,b*255])
        self.lut = np.array(lut, dtype='uint8')

        self._x_range = None
        self._y_range = None


        
        
        
        self.refresh()
    
    def initialize_plot(self):
        if self.controller.some_waveforms is None:
            return
        
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        
        #~ self.curve1 = pg.PlotCurveItem()
        #~ self.plot.addItem(self.curve1)
        #~ self.curve2 = pg.PlotCurveItem()
        #~ self.plot.addItem(self.curve2)
        
        self.curves = []
        
        thresh = self.controller.get_threshold()
        self.thresh_line = pg.InfiniteLine(pos=thresh, angle=0, movable=False, pen = pg.mkPen('w'))
        self.plot.addItem(self.thresh_line)

        self.params.blockSignals(True)
        #~ self.params['bin_min'] = np.min(self.controller.some_waveforms)
        #~ self.params['bin_max'] = np.max(self.controller.some_waveforms)
        
        
        #~ print(self.controller.some_waveforms.shape)
        #~ print(self.controller.some_peaks_index.shape)
        #~ print(self.controller.some_peaks_index)
        #~ print(self.controller.spike_label[self.controller.some_peaks_index].shape)
        keep = self.controller.spike_label[self.controller.some_peaks_index]>=0
        wfs = self.controller.some_waveforms[keep, :, :]
        if wfs.shape[0]>0:
            self.params['bin_min'] = np.percentile(wfs, .001)
            self.params['bin_max'] = np.percentile(wfs, 99.999)
            
        self.params.blockSignals(False)
                

        

        
        
    def gain_zoom(self, v):
        #~ print('v', v)
        levels = self.image.getLevels()
        if levels is not None:
            self.image.setLevels(levels * v, update=True)

    def refresh(self):
        if not hasattr(self, 'viewBox'):
            self.initialize_plot()
        
        if not hasattr(self, 'viewBox'):
            return
        
        if self._x_range is not None:
            #~ self._x_range = self.plot.getXRange()
            #~ self._y_range = self.plot.getYRange()
            #this may change with pyqtgraph
            self._x_range = tuple(self.viewBox.state['viewRange'][0])
            self._y_range = tuple(self.viewBox.state['viewRange'][1])
        
        
        cluster_visible = self.controller.cluster_visible
        visibles = [k for k, v in cluster_visible.items() if v and k>=-1 ]

        #remove old curves
        for curve in self.curves:
            self.plot.removeItem(curve)
        self.curves = []
        
        if len(visibles)>self.params['max_label'] or len(visibles)==0:
            self.image.hide()
            return
        
        #~ if  len(visibles)==1:
            #~ self.curve2.hide()
        
        
        if self.controller.some_peaks_index is None:
            self.plot.clear()
            return

        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep = np.in1d(labels, visibles)
        
        if self.params['data']=='waveforms':
            wf = self.controller.some_waveforms
            if wf is None:
                self.plot.clear()
                return
            data_kept = wf[keep].copy()
            if data_kept.size == 0:
                self.plot.clear()
                return
            data_kept = data_kept.swapaxes(1,2).reshape(data_kept.shape[0], -1)
        
        elif self.params['data']=='features':
            data = self.controller.some_features
            data_kept = data[keep]
            if data is None:
                self.plot.clear()
                return
        
        #TODO change for PCA
        if self.params['data']=='waveforms':
            bin_min, bin_max = self.params['bin_min'], self.params['bin_max']
            bin_size = self.params['bin_size']
            bins = np.arange(bin_min, bin_max, self.params['bin_size'])
            
        elif self.params['data']=='features':
            bin_min, bin_max = np.min(data_kept), np.max(data_kept)
            #~ n = 500
            bins = np.linspace(bin_min, bin_max, 500)
            bin_size = bins[1]  - bins[0]
            #~ med, mad = median_mad(data_kept, axis=0)
            #~ min, max = np.min(med-10*mad), np.max(med+10*mad)
            #~ n = self.params['nb_bin']
            #~ bin = (max-min)/(n-1)
        n = bins.size

        hist2d = np.zeros((data_kept.shape[1], bins.size))
        indexes0 = np.arange(data_kept.shape[1])
        
        data_bined = np.floor((data_kept-bin_min)/bin_size).astype('int32')
        data_bined = data_bined.clip(0, bins.size-1)
        
        for d in data_bined:
            hist2d[indexes0, d] += 1
        
        if self.controller.cluster_visible[labelcodes.LABEL_NOISE] and self.controller.some_noise_snippet is not None:
            #~ print('labelcodes.LABEL_NOISE in cluster_visible', labelcodes.LABEL_NOISE in cluster_visible, cluster_visible)
            if self.params['data']=='waveforms':
                noise = self.controller.some_noise_snippet
                noise = noise.swapaxes(1,2).reshape(noise.shape[0], -1)
                noise_bined = np.floor((noise-bin_min)/bin_size).astype('int32')
                noise_bined = noise_bined.clip(0, bins.size-1)
                for d in noise_bined:
                    hist2d[indexes0, d] += 1
            #~ elif self.params['data']=='features':
            

        self.image.setImage(hist2d, lut=self.lut)#, levels=[0, self._max])
        self.image.setRect(QT.QRectF(-0.5, bin_min, data_kept.shape[1], bin_max-bin_min))
        self.image.show()
        
        
        #~ for k, curve in zip(visibles, [self.curve1, self.curve2]):
        for k in visibles:
            
            median = self.controller.get_waveform_centroid(k, 'median')
            if median is None:
                continue
            if self.params['data']=='waveforms':
                y = median.T.flatten()
            else:
                y = np.median(data[labels==k], axis=0)
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            
            curve = pg.PlotCurveItem(x=indexes0, y=y, pen=pg.mkPen(color, width=2))
            self.plot.addItem(curve)
            self.curves.append(curve)
            #~ curve.setData()
            #~ curve.setPen()
            #~ curve.show()
        
        if self.params['display_threshold'] and self.params['data']=='waveforms' :
            self.thresh_line.show()
        else:
            self.thresh_line.hide()
        
        
        if self._x_range is None:
            self._x_range = 0, indexes0[-1] #hist2d.shape[1]
            self._y_range = bin_min, bin_max
        

        self.plot.setXRange(*self._x_range, padding = 0.0)
        self.plot.setYRange(*self._y_range, padding = 0.0)
    
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

    def show_hide_1d_dist(self, v=None):
        #~ print(v)
        if v:
            self.graphicsview2.show()
        else:
            self.graphicsview2.hide()





