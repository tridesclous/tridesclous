import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class ControllerBase(QtCore.QObject):
    peak_selection_changed = QtCore.pyqtSignal()
    peak_cluster_changed = QtCore.pyqtSignal()
    colors_changed = QtCore.pyqtSignal()
    cluster_visibility_changed = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent=parent)
        self._views = []
    
    def declare_a_view(self, new_view):
        assert new_view not in self._views, 'view already declared {}'.format(self)
        self._views.append(new_view)
        
        new_view.peak_selection_changed.connect(self.on_peak_selection_changed)
        new_view.peak_cluster_changed.connect(self.on_peak_cluster_changed)
        new_view.colors_changed.connect(self.on_colors_changed)
        new_view.cluster_visibility_changed.connect(self.on_cluster_visibility_changed)
        
        #~ for view in self._views:
            #~ view.peak_selection_changed.connect(new_view.on_peak_selection_changed)
            #~ new_view.peak_selection_changed.connect(view.on_peak_selection_changed)

            #~ view.peak_cluster_changed.connect(new_view.on_peak_cluster_changed)
            #~ new_view.peak_cluster_changed.connect(view.on_peak_cluster_changed)

            #~ view.colors_changed.connect(new_view.on_colors_changed)
            #~ new_view.colors_changed.connect(view.on_colors_changed)

            #~ view.cluster_visibility_changed.connect(new_view.on_cluster_visibility_changed)
            #~ new_view.cluster_visibility_changed.connect(view.on_cluster_visibility_changed)

    def on_peak_selection_changed(self):
        for view in self._views:
            if view==self.sender(): continue
            view.on_peak_selection_changed()

    def on_peak_cluster_changed(self):
        for view in self._views:
            if view==self.sender(): continue
            view.on_peak_cluster_changed()
    
    def on_colors_changed(self):
        for view in self._views:
            if view==self.sender(): continue
            view.on_colors_changed()
    
    def on_cluster_visibility_changed(self):
        for view in self._views:
            if view==self.sender(): continue
            view.on_cluster_visibility_changed()



class WidgetBase(QtGui.QWidget):
    peak_selection_changed = QtCore.pyqtSignal()
    peak_cluster_changed = QtCore.pyqtSignal()
    colors_changed = QtCore.pyqtSignal()
    cluster_visibility_changed = QtCore.pyqtSignal()
    
    def __init__(self, parent = None, controller=None):
        QtGui.QWidget.__init__(self, parent)
        self.controller = controller
        if self.controller is not None:
            self.controller.declare_a_view(self)
    
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
