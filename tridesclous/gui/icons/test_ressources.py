 
import sys






if __name__ == '__main__' :
    from tridesclous.gui.myqt import QT, mkQApp, QT_MODE
    import  tridesclous.gui.icons
    print('QT_MODE', QT_MODE)
    
    app = mkQApp()

    w = QT.QWidget()
    w.show()
    w.setWindowIcon(QT.QIcon(':/main_icon.png'))

    app.exec_()
