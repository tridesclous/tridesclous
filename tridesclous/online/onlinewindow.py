import numpy as np
from ..gui import QT
import pyqtgraph as pg

from pyacq.core import WidgetNode


from onlinepeeler import OnlinePeeler
from .onlinetraceviewer import OnlineTraceViewer




class OnlineWindow(WidgetNode):
    def _init__(self, parent=None):
       WidgetNode.__inti__(self, parent=parent)
        
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        
        
    
    