from tridesclous import *
import  pyqtgraph as pg
from matplotlib import pyplot


def get_spikesorter():
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    #~ spikesorter = SpikeSorter(dirname = '../../tests/datatest_neo')
    print(spikesorter.summary(level=1))
    spikesorter.refresh_colors(reset=True, palette = 'husl')
    return spikesorter


def test_traceviewer():
    app = pg.mkQApp()
    
    spikesorter = get_spikesorter()
    
    traceviewer = PeelerTraceViewer(spikesorter=spikesorter, mode = 'memory', signal_type = 'filtered')
    traceviewer.show()
    traceviewer.resize(800,600)
    
    app.exec_()
    
def test_peelerwindow():
    app = pg.mkQApp()
    spikesorter = get_spikesorter()
    
    win = PeelerWindow(spikesorter)
    win.show()
    
    app.exec_()

    
    
if __name__ == '__main__':
    #~ test_traceviewer()
    
    test_peelerwindow()
    