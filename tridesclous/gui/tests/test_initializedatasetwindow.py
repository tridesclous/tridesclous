from tridesclous import *

import pytest
from tridesclous.tests.testingtools import ON_CI_CLOUD
from tridesclous.gui.tests.testingguitools import HAVE_QT5

if HAVE_QT5:
    import  pyqtgraph as pg
    from tridesclous.gui import *


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_InitializeDatasetWindow():
    app = mkQApp()
    win = InitializeDatasetWindow()
    win.show()
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ChannelGroupWidget():
    app = mkQApp()
    win = ChannelGroupWidget()
    channel_names = ['ch{}'.format(i) for i in range(32)]
    win.set_channel_names(channel_names)
    win.show()
    
    if __name__ == '__main__':
        app.exec_()
    
    

    
if __name__ == '__main__':
    test_InitializeDatasetWindow()
    test_ChannelGroupWidget()

