from .myqt import QT
import pyqtgraph as pg

import numpy as np

class MyViewBox(pg.ViewBox):
    pass
    

class ProbeGeometryView(QT.QWidget):
    
    def __init__(self, parent = None, channel_groups=None):
        QT.QWidget.__init__(self, parent)
        
        self.channel_groups = channel_groups
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        h = QT.QHBoxLayout()
        self.layout.addLayout(h)
        
        self.combo_chan_grp = QT.QComboBox()
        h.addWidget(self.combo_chan_grp)
        self.combo_chan_grp.clear()
        self.combo_chan_grp.addItems([str(k) for k in self.channel_groups.keys()])
        self.combo_chan_grp.currentIndexChanged .connect(self.on_chan_grp_change)
        
        self.checkbox = QT.QCheckBox('flip_bottom_up')
        h.addWidget(self.checkbox)
        self.checkbox.stateChanged.connect(self.refresh)
        
        #~ self.combo_chan_grp.blockSignals(True)
        #~ self.combo_chan_grp.blockSignals(False)
        #~ self.on_chan_grp_change()

        
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        

        self.viewBox = MyViewBox()
        self.viewBox.disableAutoRange()
        
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        self.plot.showAxis('left', False)
        
        self.refresh()

    
    def on_chan_grp_change(self):
        self.refresh()
        
    def refresh(self, v=None):
        self.plot.clear()
        
        flip_bottom_up = self.checkbox.checkState()
        
        
        chan_grp = int(self.combo_chan_grp.currentText())
        channel_group = self.channel_groups[chan_grp]
        if channel_group['geometry'] is None:
            return
        
        
        geometry = [ channel_group['geometry'][chan] for chan in channel_group['channels'] ]
        geometry = np.array(geometry, dtype='float64')
        
        if flip_bottom_up:
            geometry[:, 1] *= -1.
        
        for c, chan in enumerate(channel_group['channels']):
            x, y = geometry[c]
            
            #~ name = '{}: {}'.format(c, chan)
            name = '{}'.format(chan)
            itemtxt = pg.TextItem(name, anchor=(.5,.5))
            self.plot.addItem(itemtxt)
            itemtxt.setPos(x, y)

        margin = 100
        
        self.plot.setXRange(np.min(geometry[:, 0])-margin, np.max(geometry[:, 0])+margin)
        self.plot.setYRange(np.min(geometry[:, 1])-margin, np.max(geometry[:, 1])+margin)
            

    #~ for c, chan in enumerate(channels):
        #~ x, y = geometry[c]
        #~ ax.plot([x], [y], marker='o', color='r')
        #~ ax.text(x, y, '{}: {}'.format(c, chan),  size=20)

