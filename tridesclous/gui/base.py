from .myqt import QT
import pyqtgraph as pg


class ControllerBase(QT.QObject):
    spike_selection_changed = QT.pyqtSignal()
    spike_label_changed = QT.pyqtSignal()
    colors_changed = QT.pyqtSignal()
    cluster_visibility_changed = QT.pyqtSignal()
    
    def __init__(self, parent=None):
        QT.QObject.__init__(self, parent=parent)
        self.views = []
    
    def declare_a_view(self, new_view):
        assert new_view not in self.views, 'view already declared {}'.format(self)
        self.views.append(new_view)
        
        new_view.spike_selection_changed.connect(self.on_spike_selection_changed)
        new_view.spike_label_changed.connect(self.on_spike_label_changed)
        new_view.colors_changed.connect(self.on_colors_changed)
        new_view.cluster_visibility_changed.connect(self.on_cluster_visibility_changed)
        
    def on_spike_selection_changed(self):
        for view in self.views:
            if view==self.sender(): continue
            view.on_spike_selection_changed()

    def on_spike_label_changed(self):
        for view in self.views:
            if view==self.sender(): continue
            view.on_spike_label_changed()
    
    def on_colors_changed(self):
        for view in self.views:
            if view==self.sender(): continue
            view.on_colors_changed()
    
    def on_cluster_visibility_changed(self):
        for view in self.views:
            if view==self.sender(): continue
            view.on_cluster_visibility_changed()



class WidgetBase(QT.QWidget):
    spike_selection_changed = QT.pyqtSignal()
    spike_label_changed = QT.pyqtSignal()
    colors_changed = QT.pyqtSignal()
    cluster_visibility_changed = QT.pyqtSignal()
    
    def __init__(self, parent = None, controller=None):
        QT.QWidget.__init__(self, parent)
        self.controller = controller
        if self.controller is not None:
            self.controller.declare_a_view(self)
    
    def refresh(self):
        raise(NotImplementedError)
    
    def on_spike_selection_changed(self):
        self.refresh()

    def on_spike_label_changed(self):
        self.refresh()
        
    def on_colors_changed(self):
        self.refresh()
    
    def on_cluster_visibility_changed(self):
        self.refresh()
