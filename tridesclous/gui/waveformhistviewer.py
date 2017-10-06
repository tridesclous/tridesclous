from .myqt import QT
import pyqtgraph as pg

import numpy as np
import matplotlib.cm
import matplotlib.colors


from .base import WidgetBase
from .tools import ParamDialog

class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    gain_zoom = QT.pyqtSignal(float)
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def wheelEvent(self, ev):
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
                          {'name': 'nb_bin', 'type': 'int', 'value' : 500 },
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
            
    def on_params_change(self):
        
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
        self.viewBox = MyViewBox()
        self.viewBox.doubleclicked.connect(self.open_settings)
        self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        
        self.curve1 = pg.PlotCurveItem()
        self.plot.addItem(self.curve1)
        self.curve2 = pg.PlotCurveItem()
        self.plot.addItem(self.curve2)

 
        
    def gain_zoom(self, v):
        #~ print('v', v)
        levels = self.image.getLevels()*v
        self.image.setLevels(levels, update=True)

    def refresh(self):

        if self._x_range is not None:
            #~ self._x_range = self.plot.getXRange()
            #~ self._y_range = self.plot.getYRange()
            #this may change with pyqtgraph
            self._x_range = tuple(self.viewBox.state['viewRange'][0])
            self._y_range = tuple(self.viewBox.state['viewRange'][1])
        
            
        cluster_visible = self.controller.cluster_visible
        visibles = [c for c, v in self.controller.cluster_visible.items() if v and c>=0]

        if len(visibles) not in [1, 2]:
            self.image.hide()
            self.curve1.hide()
            self.curve2.hide()
            return
        
        if  len(visibles)==1:
            self.curve2.hide()

        if self.params['data']=='waveforms':
            wf = self.controller.some_waveforms
            data = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
        if self.params['data']=='features':
            data = self.controller.some_features
        
        
        labels = self.controller.spike_label[self.controller.some_peaks_index]
        keep = np.in1d(labels, visibles)
        data_kept = data[keep]
        
        min, max = np.min(data_kept), np.max(data_kept)
        n = self.params['nb_bin']
        bin = (max-min)/(n-1)
        
        hist2d = np.zeros((data_kept.shape[1], n))
        indexes0 = np.arange(data_kept.shape[1])
        
        data_bined = np.floor((data_kept-min)/bin).astype('int32')
        for d in data_bined:
            hist2d[indexes0, d] += 1

        self.image.setImage(hist2d, lut=self.lut)#, levels=[0, self._max])
        self.image.show()
        
        
        for k, curve in zip(visibles, [self.curve1, self.curve2]):
            
            if self.params['data']=='waveforms':
                wf0 = self.controller.centroids[k]['median'].T.flatten()
                curve_bined = np.ceil((wf0-min)/bin).astype('int32')
            else:
                feat0 = np.median(data[labels==k], axis=0)
                curve_bined = np.ceil((feat0-min)/bin).astype('int32')
            
            
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            curve.setData(x=indexes0+.5, y=curve_bined)
            curve.setPen(pg.mkPen(color, width=2))
            
            curve.show()
        
        if self._x_range is None:
            self._x_range = 0, hist2d.shape[0]
            self._y_range = 0, hist2d.shape[1]
        

        self.plot.setXRange(*self._x_range, padding = 0.0)
        self.plot.setYRange(*self._y_range, padding = 0.0)
        

    def on_cluster_visibility_changed(self):
        self.refresh()

