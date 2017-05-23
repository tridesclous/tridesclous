from tridesclous import *
import  pyqtgraph as pg




def test_MainWindow():
    app = pg.mkQApp()
    win = MainWindow()
    win.show()
    app.exec_()

def test_InitializeWindow():
    app = pg.mkQApp()
    win = InitializeWindow()
    win.show()
    app.exec_()


def test_ChannelGroupWidget():
    app = pg.mkQApp()
    win = ChannelGroupWidget()
    channel_names = ['ch{}'.format(i) for i in range(16)]
    win.set_channel_names(channel_names)
    win.show()
    app.exec_()
    
    

    
if __name__ == '__main__':
    test_MainWindow()
    #~ test_InitializeWindow()
    #~ test_ChannelGroupWidget()

