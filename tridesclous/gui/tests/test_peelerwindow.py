from tridesclous import *
import  pyqtgraph as pg
from matplotlib import pyplot


def get_spikesorter():
    #~ spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    spikesorter = SpikeSorter(dirname = '../../tests/datatest_neo')
    print(spikesorter.summary(level=1))
    spikesorter.detect_peaks_extract_waveforms(seg_nums = 'all',  threshold=-6.,
                            peak_sign = '-', n_span = 2,  n_left=-30, n_right=50)
    #~ print(spikesorter.summary(level=1))
    spikesorter.project(method = 'pca', n_components = 5)
    spikesorter.find_clusters(7)
    spikesorter.refresh_colors(reset=True, palette = 'husl')
    #~ print(spikesorter.summary(level=1))
    spikesorter.construct_catalogue()
    spikesorter.appy_peeler(seg_nums = 'all',  levels = [0, 1])

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
    test_traceviewer()
    
    #~ test_peelerwindow()
    