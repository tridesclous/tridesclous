 
import sys
from tridesclous.gui.myqt import QT, mkQApp, QT_MODE
print('QT_MODE', QT_MODE)

import  tridesclous.gui.icons


if __name__ == '__main__' :
	app = mkQApp()
	
	w = QT.QWidget()
	w.show()
	w.setWindowIcon(QT.QIcon(':/main_icon.png'))
	
	app.exec_()
	
	
