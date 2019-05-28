from tridesclous import *
import  pyqtgraph as pg

import pytest
from tridesclous.tests.testingtools import ON_CI_CLOUD


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_MainWindow():
    app = pg.mkQApp()
    win = MainWindow()
    win.show()
    if __name__ == '__main__':
        app.exec_()



    
if __name__ == '__main__':
    test_MainWindow()

