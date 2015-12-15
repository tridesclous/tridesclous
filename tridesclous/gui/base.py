import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class WidgetBase(QtGui.QWidget):
    peak_selection_changed = QtCore.pyqtSignal()
    peak_cluster_changed = QtCore.pyqtSignal()
    colors_changed = QtCore.pyqtSignal()
    cluster_visibility_changed = QtCore.pyqtSignal()
    
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
    
    def refresh(self):
        raise(NotImplementedError)
    
    def on_peak_selection_changed(self):
        self.refresh()

    def on_peak_cluster_changed(self):
        self.refresh()
        
    def on_colors_changed(self):
        self.refresh()
    
    def on_cluster_visibility_changed(self):
        self.refresh()
