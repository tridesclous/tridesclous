from tridesclous import *
from tridesclous.gui import GpuSelector
import  pyqtgraph as pg




def test_GpuSelector():
    
    app = mkQApp()

    appname = 'tridesclous'
    settings_name = 'settings'
    settings = QT.QSettings(appname, settings_name)
    
    view = GpuSelector(settings=settings)
    
    
    if __name__ == '__main__':
        #~ app.exec_()
        if view.exec_():
            view.apply_cl_setting()
            d = view.get()
            
            print(d)

    


if __name__ == '__main__':
    test_GpuSelector()
