from tridesclous import *

import pytest
from tridesclous.tests.testingtools import ON_CI_CLOUD
from tridesclous.gui.tests.testingguitools import HAVE_QT5

if HAVE_QT5:
    import  pyqtgraph as pg
    from tridesclous.gui import *


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_MainWindow():
    app = mkQApp()
    win = MainWindow()
    win.show()
    if __name__ == '__main__':
        app.exec_()



    
if __name__ == '__main__':
    test_MainWindow()

