from tridesclous import *
import  pyqtgraph as pg




def test_MainWindow():
    app = pg.mkQApp()
    win = MainWindow()
    win.show()
    if __name__ == '__main__':
        app.exec_()



    
if __name__ == '__main__':
    test_MainWindow()

