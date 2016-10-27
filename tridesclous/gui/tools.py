import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np

class TimeSeeker(QtGui.QWidget) :
    
    time_changed = QtCore.pyqtSignal(float)
    
    def __init__(self, parent = None, show_slider = True, show_spinbox = True) :
        QtGui.QWidget.__init__(self, parent)
        
        self.layout = QtGui.QHBoxLayout()
        self.setLayout(self.layout)
        
        if show_slider:
            self.slider = QtGui.QSlider(orientation=QtCore.Qt.Horizontal, minimum=0, maximum=999)
            self.layout.addWidget(self.slider)
            self.slider.valueChanged.connect(self.slider_changed)
        else:
            self.slider = None
            
        if show_spinbox:
            self.spinbox = pg.SpinBox(decimals = 4, bounds=[-np.inf, np.inf], suffix = 's', siPrefix = True, 
                            step = 0.1, dec = True, minStep = 0.001)
            self.layout.addWidget(self.spinbox)
            self.spinbox.valueChanged.connect(self.spinbox_changed)
        else:
            self.spinbox = None

        self.t = 0 #  s
        self.set_start_stop(0., 10.)

    def set_start_stop(self, t_start, t_stop, seek = True):
        if np.isnan(t_start) or np.isnan(t_stop): return
        assert t_stop>t_start
        self.t_start = t_start
        self.t_stop = t_stop
        
        if seek:
            self.seek(self.t_start)
        
        if self.spinbox is not None:
            self.spinbox.setMinimum(t_start)
            self.spinbox.setMaximum(t_stop)

    def slider_changed(self, pos):
        t = pos/1000.*(self.t_stop - self.t_start)+self.t_start
        self.seek(t, set_slider = False)
    
    def spinbox_changed(self, val):
        self.seek(val, set_spinbox = False)
        
    def seek(self, t, set_slider = True, set_spinbox = True, emit = True):
        self.t = t
        
        if self.slider is not None and set_slider:
            self.slider.valueChanged.disconnect(self.slider_changed)
            pos = int((self.t - self.t_start)/(self.t_stop - self.t_start)*1000.)
            self.slider.setValue(pos)
            self.slider.valueChanged.connect(self.slider_changed)
        
        if self.spinbox is not None and set_spinbox:
            self.spinbox.valueChanged.disconnect(self.spinbox_changed)
            self.spinbox.setValue(t)
            self.spinbox.valueChanged.connect(self.spinbox_changed)
        
        if emit:
            self.time_changed.emit(float(self.t))

def get_dict_from_group_param(param, cascade = False):
    assert param.type() == 'group'
    d = {}
    for p in param.children():
        if p.type() == 'group':
            if cascade:
                d[p.name()] = get_dict_from_group_param(p, cascade = True)
            continue
        else:
            d[p.name()] = p.value()
    return d

class ParamDialog(QtGui.QDialog):
    def __init__(self,   params, title = '', parent = None):
        QtGui.QDialog.__init__(self, parent = parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        
        self.params = pg.parametertree.Parameter.create( name=title, type='group', children = params)
        
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        #~ self.tree_params.setWindowFlags(QtCore.Qt.Window)
        layout.addWidget(self.tree_params)

        but = QtGui.QPushButton('OK')
        layout.addWidget(but)
        but.clicked.connect(self.accept)

    def get(self):
        return get_dict_from_group_param(self.params)
    

if __name__=='__main__':
    app = pg.mkQApp()
    #~ timeseeker =TimeSeeker()
    #~ timeseeker.show()
    #~ app.exec_()
    params = [{'name' : 'a', 'value' : 1., 'type' : 'float'}]
    dialog = ParamDialog(params, title = 'yep')
    dialog.exec_()
    print(dialog.get())
    
    


