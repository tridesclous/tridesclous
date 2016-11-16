import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import seaborn as sns



class CatalogueController(QtCore.QObject):
    def __init__(self, parent=None, dataio=None, catalogueconstructor=None):
        QtCore.QObject.__init__(self, parent=parent)
        self.dataio=dataio
        self.cc = catalogueconstructor = catalogueconstructor
        
        
        #~ self.init_plot_attributes()
        #~ self.refresh_colors()