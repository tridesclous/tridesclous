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
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() == QT.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()
        

class WaveformHistViewer(WidgetBase):
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)

        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        
        self.create_settings()
        
        self.initialize_plot()
        self.similarity = None
        
        self.on_params_change()#this do refresh
        #~ self.refresh()

    def create_settings(self):
        _params = [
                          {'name': 'colormap', 'type': 'list', 'values' : ['hot', 'viridis', 'jet', 'gray',  ] },
                          {'name': 'data', 'type': 'list', 'values' : ['waveforms', 'features', ] },
                          {'name': 'bin_min', 'type': 'float', 'value' : -20. },
                          {'name': 'bin_max', 'type': 'float', 'value' : 8. },
                          {'name': 'bin_size', 'type': 'float', 'value' : .1 },
                          {'name': 'display_threshold', 'type': 'bool', 'value' : True },
                          {'name': 'max_label', 'type': 'int', 'value' : 2 },
                          ]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        
        self.params.sigTreeStateChanged.connect(self.refresh)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for waveforms hist viewer')
        self.tree_params.setWindowFlags(QT.Qt.Window)
        
        self.params.sigTreeStateChanged.connect(self.on_params_change)

    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()
            
    def on_params_change(self, ): #params, changes
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
        self.params['bin_min'] = np.percentile(self.controller.some_waveforms, .05)
        self.params['bin_max'] = np.percentile(self.controller.some_waveforms, 99.95)
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
        visibles = [c for c, v in cluster_visible.items() if v ]

        #remove old curves
        for curve in self.curves:
            self.plot.removeItem(curve)
        self.curves = []
        
        if len(visibles)>self.params['max_label'] or len(visibles)==0:
            self.image.hide()
            return
        
        #~ if  len(visibles)==1:
            #~ self.curve2.hide()

        if self.params['data']=='waveforms':
            wf = self.controller.some_waveforms
            data = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
        elif self.params['data']=='features':
            data = self.controller.some_features
        
        
        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep = np.in1d(labels, visibles)
        data_kept = data[keep]
        
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

        self.image.setImage(hist2d, lut=self.lut)#, levels=[0, self._max])
        self.image.setRect(QT.QRectF(-0.5, bin_min, data_kept.shape[1], bin_max-bin_min))
        self.image.show()
        
        
        #~ for k, curve in zip(visibles, [self.curve1, self.curve2]):
        for k in visibles:
            if k not in self.controller.centroids:
                continue
            if self.params['data']=='waveforms':
                data = self.controller.centroids[k]['median'].T.flatten()
            else:
                data = np.median(data[labels==k], axis=0)
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            
            curve = pg.PlotCurveItem(x=indexes0, y=data, pen=pg.mkPen(color, width=2))
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
        

    def on_cluster_visibility_changed(self):
        self.refresh()

