from .myqt import QT
import pyqtgraph as pg

import numpy as np
import pandas as pd

from .base import WidgetBase


class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()
    gain_zoom = QT.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #~ self.disableAutoRange()
    def mouseClickEvent(self, ev):
        ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    #~ def mouseDragEvent(self, ev):
        #~ ev.ignore()
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() == QT.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()



class WaveformViewerBase(WidgetBase):
    #base for both WaveformViewer (Catalogue) and PeelerWaveformViewer
    

    
    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent=parent, controller=controller)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        #~ self.create_settings()
        
        self.create_toolbar()
        self.layout.addWidget(self.toolbar)

        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
        
        self.alpha = 60
        self.refresh()
    
    def create_toolbar(self):
        tb = self.toolbar = QT.QToolBar()
        
        #Mode flatten or geometry
        self.combo_mode = QT.QComboBox()
        tb.addWidget(self.combo_mode)
        #~ self.mode = 'flatten'
        #~ self.combo_mode.addItems([ 'flatten', 'geometry'])
        self.mode = 'geometry'
        self.combo_mode.addItems([ 'geometry', 'flatten'])
        self.combo_mode.currentIndexChanged.connect(self.on_combo_mode_changed)
        tb.addSeparator()
        
        
        but = QT.QPushButton('settings')
        but.clicked.connect(self.open_settings)
        tb.addWidget(but)

        but = QT.QPushButton('scale')
        but.clicked.connect(self.zoom_range)
        tb.addWidget(but)

        but = QT.QPushButton('refresh')
        but.clicked.connect(self.refresh)
        tb.addWidget(but)
    
    def on_combo_mode_changed(self):
        self.mode = str(self.combo_mode.currentText())
        self.initialize_plot()
        self.refresh()
    
    def on_params_changed(self, params, changes):
        for param, change, data in changes:
            if change != 'value': continue
            if param.name()=='flip_bottom_up':
                self.initialize_plot()
        self.refresh()

    def initialize_plot(self):
        #~ print('WaveformViewer.initialize_plot', self.controller.some_waveforms)
        if self.controller.get_waveform_left_right()[0] is None:
            return
            
        self.viewBox1 = MyViewBox()
        self.viewBox1.disableAutoRange()

        grid = pg.GraphicsLayout(border=(100,100,100))
        self.graphicsview.setCentralItem(grid)
        
        
        self.plot1 = grid.addPlot(row=0, col=0, rowspan=2, viewBox=self.viewBox1)
        self.plot1.hideButtons()
        self.plot1.showAxis('left', True)

        self.curve_one_waveform = pg.PlotCurveItem([], [], pen=pg.mkPen(QT.QColor( 'white'), width=1), connect='finite')
        self.plot1.addItem(self.curve_one_waveform)
        
        if self.mode=='flatten':
            grid.nextRow()
            grid.nextRow()
            self.viewBox2 = MyViewBox()
            self.viewBox2.disableAutoRange()
            self.plot2 = grid.addPlot(row=2, col=0, rowspan=1, viewBox=self.viewBox2)
            self.plot2.hideButtons()
            self.plot2.showAxis('left', True)
            self.viewBox2.setXLink(self.viewBox1)
            self.factor_y = 1.
            
            self._common_channels_flat = None

        elif self.mode=='geometry':
            self.plot2 = None
            
            chan_grp = self.controller.chan_grp
            channel_group = self.controller.dataio.channel_groups[chan_grp]
            #~ print(channel_group['geometry'])
            if channel_group['geometry'] is None:
                print('no geometry')
                self.xvect = None
            else:

                n_left, n_right = self.controller.get_waveform_left_right()
                width = n_right - n_left
                nb_channel = self.controller.nb_channel
                
                #~ self.xvect = np.zeros(shape[0]*shape[1], dtype='float32')
                #~ self.xvect = np.zeros((shape[1], shape[0]), dtype='float32')
                self.xvect = np.zeros((nb_channel, width), dtype='float32')

                self.arr_geometry = []
                for i, chan in enumerate(self.controller.channel_indexes):
                    x, y = channel_group['geometry'][chan]
                    self.arr_geometry.append([x, y])
                self.arr_geometry = np.array(self.arr_geometry, dtype='float64')
                
                if self.params['flip_bottom_up']:
                    self.arr_geometry[:, 1] *= -1.
                
                xpos = self.arr_geometry[:,0]
                ypos = self.arr_geometry[:,1]
                
                if np.unique(xpos).size>1:
                    self.delta_x = np.min(np.diff(np.sort(np.unique(xpos))))
                else:
                    self.delta_x = np.unique(xpos)[0]
                if np.unique(ypos).size>1:
                    self.delta_y = np.min(np.diff(np.sort(np.unique(ypos))))
                else:
                    self.delta_y = max(np.unique(ypos)[0], 1)
                self.factor_y = .3
                if self.delta_x>0.:
                    #~ espx = self.delta_x/2. *.95
                    espx = self.delta_x/2.5
                else:
                    espx = .5
                for i, chan in enumerate(channel_group['channels']):
                    x, y = channel_group['geometry'][chan]
                    self.xvect[i, :] = np.linspace(x-espx, x+espx, num=width)
        
        self.wf_min, self.wf_max = self.controller.get_min_max_centroids()
        
        
        self._x_range = None
        self._y1_range = None
        self._y2_range = None
        
        self.viewBox1.gain_zoom.connect(self.gain_zoom)
        
        self.viewBox1.doubleclicked.connect(self.open_settings)
        
        #~ self.viewBox.xsize_zoom.connect(self.xsize_zoom)    
    

    def gain_zoom(self, factor_ratio):
        self.factor_y *= factor_ratio
        
        self.refresh(keep_range=True)
    
    def zoom_range(self):
        self._x_range = None
        self._y1_range = None
        self._y2_range = None
        self.refresh(keep_range=False)
    
    def refresh(self, keep_range=False):
        
        if not hasattr(self, 'viewBox1'):
            self.initialize_plot()
        
        if not hasattr(self, 'viewBox1'):
            return
        
        n_selected = np.sum(self.controller.spike_selection)
        
        if self.params['show_only_selected_cluster'] and n_selected==1:
            cluster_visible = {k:False for k in self.controller.cluster_visible}
            ind, = np.nonzero(self.controller.spike_selection)
            ind = ind[0]
            k = self.controller.spikes[ind]['cluster_label']
            cluster_visible[k] = True
        else:
            cluster_visible = self.controller.cluster_visible
        
        if self.mode=='flatten':
            self.refresh_mode_flatten(cluster_visible, keep_range)
        elif self.mode=='geometry':
            self.refresh_mode_geometry(cluster_visible, keep_range)
        
        self._refresh_one_spike(n_selected)
    
    
    def refresh_mode_flatten(self, cluster_visible, keep_range):
        if self._x_range is not None and keep_range:
            #this may change with pyqtgraph
            self._x_range = tuple(self.viewBox1.state['viewRange'][0])
            self._y1_range = tuple(self.viewBox1.state['viewRange'][1])
            self._y2_range = tuple(self.viewBox2.state['viewRange'][1])
        
        
        
        self.plot1.clear()
        self.plot2.clear()
        self.plot1.addItem(self.curve_one_waveform)
        
        if self.controller.spike_index ==[]:
            return

        nb_channel = self.controller.nb_channel
        n_left, n_right = self.controller.get_waveform_left_right()
        width = n_right - n_left
        
        sparse = self.controller.have_sparse_template and self.params['sparse_display']
        visibles = [k for k, v in cluster_visible.items() if v and k>=-1 ]
        
        if sparse:
            if len(visibles) > 0:
                common_channels = self.controller.get_common_sparse_channels(visibles)
            else:
                #~ common_channels = np.array([], dtype='int64')
                return
        else:
            common_channels = self.controller.channels
        
        self._common_channels_flat = common_channels
        
        #lines
        def addSpan(plot):
            white = pg.mkColor(255, 255, 255, 20)
            #~ for i in range(nb_channel):
            for i, c in enumerate(common_channels):
                if i%2==1:
                    region = pg.LinearRegionItem([width*i, width*(i+1)-1], movable = False, brush = white)
                    plot.addItem(region, ignoreBounds=True)
                    for l in region.lines:
                        l.setPen(white)
                vline = pg.InfiniteLine(pos = -n_left + width*i, angle=90, movable=False, pen = pg.mkPen('w'))
                plot.addItem(vline)
        
        if self.params['plot_limit_for_flatten']:
            addSpan(self.plot1)
            addSpan(self.plot2)
        
        if self.params['display_threshold']:
            thresh = self.controller.get_threshold()
            thresh_line = pg.InfiniteLine(pos=thresh, angle=0, movable=False, pen = pg.mkPen('w'))
            self.plot1.addItem(thresh_line)

            
            
        
        
        #waveforms
        
        if self.params['metrics']=='median/mad':
            key1, key2 = 'median', 'mad'
        elif self.params['metrics']=='mean/std':
            key1, key2 = 'mean', 'std'
        
        #~ shape = self.controller.get_waveforms_shape()
        #~ if shape is None:
            #~ return
        n_left, n_right = self.controller.get_waveform_left_right()
        if n_left is None:
            return
        width = n_right - n_left

        shape = (width, len(common_channels))
        xvect = np.arange(shape[0]*shape[1])
        
        #~ for i,k in enumerate(self.controller.centroids):
        for k in cluster_visible:
            #~ if not self.controller.cluster_visible[k]:
            if not cluster_visible[k]:
                continue
            
            #~ wf0 = self.controller.centroids[k][key1].T.flatten()
            #~ mad = self.controller.centroids[k][key2].T.flatten()
            wf0, chans = self.controller.get_waveform_centroid(k, key1, channels=common_channels)
            if wf0 is None: continue
            wf0 = wf0.T.flatten()
            
            mad, chans = self.controller.get_waveform_centroid(k, key2, channels=common_channels)
            
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            curve = pg.PlotCurveItem(xvect, wf0, pen=pg.mkPen(color, width=2))
            self.plot1.addItem(curve)
            
            
            if self.params['fillbetween'] and mad is not None:
                mad = mad.T.flatten()
                color2 = QT.QColor(color)
                color2.setAlpha(self.alpha)
                curve1 = pg.PlotCurveItem(xvect, wf0+mad, pen=color2)
                curve2 = pg.PlotCurveItem(xvect, wf0-mad, pen=color2)
                self.plot1.addItem(curve1)
                self.plot1.addItem(curve2)
                
                fill = pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=color2)
                self.plot1.addItem(fill)
            
            if mad is not None:
                curve = pg.PlotCurveItem(xvect, mad, pen=color)
                self.plot2.addItem(curve)        

        if self.params['show_channel_num']:
            cn = self.controller.channel_indexes_and_names
            for i, c in enumerate(common_channels):
                # chan i sabsolut chan
                chan, name = cn[c]
            #~ for i, (chan, name) in enumerate(self.controller.channel_indexes_and_names):
                itemtxt = pg.TextItem('{}: {}'.format(i, name), anchor=(.5,.5), color='#FFFF00')
                itemtxt.setFont(QT.QFont('', pointSize=12))
                self.plot1.addItem(itemtxt)
                itemtxt.setPos(width*i-n_left, 0)

        
        if self._x_range is None or not keep_range :
            if xvect.size>0:
                self._x_range = xvect[0], xvect[-1]
                self._y1_range = self.wf_min*1.1, self.wf_max*1.1
                self._y2_range = 0., 5.
        
        if self._x_range is not None:
            self.plot1.setXRange(*self._x_range, padding = 0.0)
            self.plot1.setYRange(*self._y1_range, padding = 0.0)
            self.plot2.setYRange(*self._y2_range, padding = 0.0)

        

    def refresh_mode_geometry(self, cluster_visible, keep_range):
        if self._x_range is not None and keep_range:
            #this may change with pyqtgraph
            self._x_range = tuple(self.viewBox1.state['viewRange'][0])
            self._y1_range = tuple(self.viewBox1.state['viewRange'][1])

        self.plot1.clear()
        
        if self.xvect is None:
            return

        sparse = self.controller.have_sparse_template and self.params['sparse_display']
        visibles = [k for k, v in cluster_visible.items() if v and k>=-1 ]
        
        #~ if sparse:
            #~ if len(visibles) > 0:
                #~ common_channels = self.controller.get_common_sparse_channels(visibles)
            #~ else:
                #~ common_channels = np.array([], dtype='int64')
                #~ return
        #~ else:
            #~ common_channels = self.controller.channels
        
        
        n_left, n_right = self.controller.get_waveform_left_right()
        if n_left is None:
            return
        width = n_right - n_left
        #~ shape = self.controller.get_waveforms_shape()
        #~ if shape is None:
            #~ return
        
        # if n_left/n_right have change need new xvect
        #~ if self.xvect.size != shape[0] * shape[1]:
            #~ self.initialize_plot()
        if width != self.xvect.shape[1]:
            self.initialize_plot()
        #~ shape = (shape[0], len(common_channels))
        
        self.plot1.addItem(self.curve_one_waveform)

        
        
        if self.params['metrics']=='median/mad':
            key1, key2 = 'median', 'mad'
        elif self.params['metrics']=='mean/std':
            key1, key2 = 'mean', 'std'

        #~ ypos = self.arr_geometry[:,1]
        #~ ypos = self.arr_geometry[common_channels,1]
        
        #~ xvect = self.xvect.reshape(self.controller.nb_channel, -1)[common_channels, :].flatten()
        for k in cluster_visible:
            if not cluster_visible[k]:
                continue
            
            
            wf, chans = self.controller.get_waveform_centroid(k, key1, sparse=sparse)
            
            if wf is None: continue
            
            ypos = self.arr_geometry[chans,1]
            
            wf = wf*self.factor_y*self.delta_y + ypos[None, :]
            #wf[0,:] = np.nan
            
            
            connect = np.ones(wf.shape, dtype='bool')
            connect[0, :] = 0
            connect[-1, :] = 0
            
            xvect = self.xvect[chans, :]
            
            color = self.controller.qcolors.get(k, QT.QColor( 'white'))
            
            curve = pg.PlotCurveItem(xvect.flatten(), wf.T.flatten(), pen=pg.mkPen(color, width=2), connect=connect.T.flatten())
            self.plot1.addItem(curve)
        
        if self.params['show_channel_num']:
            chan_grp = self.controller.chan_grp
            channel_group = self.controller.dataio.channel_groups[chan_grp]            
            for i, (chan, name) in enumerate(self.controller.channel_indexes_and_names):
                x, y = self.arr_geometry[i, : ]
                itemtxt = pg.TextItem('{}: {}'.format(i, name), anchor=(.5,.5), color='#FFFF00')
                itemtxt.setFont(QT.QFont('', pointSize=12))
                self.plot1.addItem(itemtxt)
                itemtxt.setPos(x, y)
        
        #~ if self._x_range is None:
        if self._x_range is None or not keep_range :
            self._x_range = np.min(self.xvect), np.max(self.xvect)
            self._y1_range = np.min(self.arr_geometry[:,1])-self.delta_y*2, np.max(self.arr_geometry[:,1])+self.delta_y*2
        
        self.plot1.setXRange(*self._x_range, padding = 0.0)
        self.plot1.setYRange(*self._y1_range, padding = 0.0)
        
    
    def _refresh_one_spike(self, n_selected):
        #TODO peak the selected peak if only one
        
        if n_selected!=1 or not self.params['plot_selected_spike']: 
            self.curve_one_waveform.setData([], [])
            return
        
        ind, = np.nonzero(self.controller.spike_selection)
        ind = ind[0]
        seg_num = self.controller.spike_segment[ind]
        peak_ind = self.controller.spike_index[ind]
        
        n_left, n_right = self.controller.get_waveform_left_right()
        
        wf = self.controller.dataio.get_signals_chunk(seg_num=seg_num, chan_grp=self.controller.chan_grp,
                i_start=peak_ind+n_left, i_stop=peak_ind+n_right,
                signal_type='processed')
        
        if wf.shape[0]==(n_right-n_left):
            #this avoid border bugs
            if self.mode=='flatten':
                if self._common_channels_flat is None:
                    self.curve_one_waveform.setData([], [])
                    return
                
                wf = wf[:, self._common_channels_flat].T.flatten()
                xvect = np.arange(wf.size)
                self.curve_one_waveform.setData(xvect, wf)
            elif self.mode=='geometry':
                ypos = self.arr_geometry[:,1]
                wf = wf*self.factor_y*self.delta_y + ypos[None, :]
                
                connect = np.ones(wf.shape, dtype='bool')
                connect[0, :] = 0
                connect[-1, :] = 0

                self.curve_one_waveform.setData(self.xvect.flatten(), wf.T.flatten(), connect=connect.T.flatten())
    
    def on_spike_selection_changed(self):
        #~ n_selected = np.sum(self.controller.spike_selection)
        #~ self._refresh_one_spike(n_selected)
        self.refresh(keep_range=True)




class WaveformViewer(WaveformViewerBase):
    """
    **Waveform viewer** is undoubtedly the view to inspect waveforms.
    
    Note that in some aspect **Waveform hist viewer** can be a better firend.
    
    All centroid (median or mean) of visible cluster are plotted here.
    
    2 main modes:
      * **geometry** waveforms are organized with 2d geometry given by PRB file.
      * **flatten** each chunk of each channel is put side by side in channel order
        than it can be ploted in 1d. The bottom view is th mad. On good cluster the mad
        must as close as possible from the value 1 because 1 is the normalized noise.
    
    The **geometry** mode is more intuitive and help users about spatial
    information. But the  **flatten**  mode is really important because is give information 
    about the variance (mad or std) for each point and about peak alignement.
    
    The centoid is dfine by median+mad but you can also check with mean+std.
    For healthy cluster it should more or less the same.
    
    Important for zooming:
      *  **geometry** : zoomXY geometry = right click, move = left click and mouse wheel = zoom waveforms
      * **flatten**: zoomXY = right click and move = left click
    
    
    Settings:
      * **plot_selected_spike**: superimposed one slected peak on centroid
      * **show_only_selected_cluster**: this auto hide all cluster except the one of selected spike
      * **plot_limit_for_flatten**: for flatten mode this plot line for delimiting channels.
        Plotting is important but it slow down the zoom.
      * **metrics**: choose median+mad or mean+std.
      * *show_channel_num**: what could it be ?
      * **flip_bottom_up**: in geometry this flip bottom up the channel geometry.
      * **display_threshold**: what could it be ?
    """
    _params = [{'name': 'plot_selected_spike', 'type': 'bool', 'value': False },
                        {'name': 'show_only_selected_cluster', 'type': 'bool', 'value': False},
                      {'name': 'plot_limit_for_flatten', 'type': 'bool', 'value': True },
                      {'name': 'metrics', 'type': 'list', 'limits': ['median/mad', 'mean/std'] },
                      {'name': 'fillbetween', 'type': 'bool', 'value': True },
                      {'name': 'show_channel_num', 'type': 'bool', 'value': False},
                      {'name': 'flip_bottom_up', 'type': 'bool', 'value': False},
                      {'name': 'display_threshold', 'type': 'bool', 'value' : True },
                      {'name': 'sparse_display', 'type': 'bool', 'value' : True },
                      ]
        

class PeelerWaveformViewer(WaveformViewerBase):
    """
    **Waveform viewer** 
    """
    _params = [{'name': 'plot_selected_spike', 'type': 'bool', 'value': True },
                        {'name': 'show_only_selected_cluster', 'type': 'bool', 'value': True},
                      {'name': 'plot_limit_for_flatten', 'type': 'bool', 'value': True },
                      {'name': 'metrics', 'type': 'list', 'limits': ['median/mad'] },
                      {'name': 'fillbetween', 'type': 'bool', 'value': True },
                      {'name': 'show_channel_num', 'type': 'bool', 'value': False},
                      {'name': 'flip_bottom_up', 'type': 'bool', 'value': False},
                      {'name': 'display_threshold', 'type': 'bool', 'value' : True },
                      {'name': 'sparse_display', 'type': 'bool', 'value' : True },
                      ]
