from tridesclous import *
import  pyqtgraph as pg



def test_InitializeDatasetWindow():
    app = pg.mkQApp()
    win = InitializeDatasetWindow()
    win.show()
    app.exec_()


def test_ChannelGroupWidget():
    app = pg.mkQApp()
    win = ChannelGroupWidget()
    channel_names = ['ch{}'.format(i) for i in range(32)]
    win.set_channel_names(channel_names)
    win.show()
    app.exec_()
    
    

    
if __name__ == '__main__':
    #~ test_InitializeDatasetWindow()
    test_ChannelGroupWidget()

