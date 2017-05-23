from .myqt import QT
import pyqtgraph as pg




class MainWindow(QT.QMainWindow):
    def __init__(self):
        QT.QMainWindow.__init__(self)
        
        self.resize(800, 600)



