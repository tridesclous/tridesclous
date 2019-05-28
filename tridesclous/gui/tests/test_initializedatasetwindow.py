from tridesclous import *
import  pyqtgraph as pg

import pytest
from tridesclous.tests.testingtools import ON_CI_CLOUD


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_InitializeDatasetWindow():
    app = pg.mkQApp()
    win = InitializeDatasetWindow()
    win.show()
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ChannelGroupWidget():
    app = pg.mkQApp()
    win = ChannelGroupWidget()
    channel_names = ['ch{}'.format(i) for i in range(32)]
    win.set_channel_names(channel_names)
    win.show()
    
    if __name__ == '__main__':
        app.exec_()
    
    

    
if __name__ == '__main__':
    test_InitializeDatasetWindow()
    test_ChannelGroupWidget()

