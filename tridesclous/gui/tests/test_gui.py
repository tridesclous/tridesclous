from tridesclous import *
import  pyqtgraph as pg


def get_spikesorter():
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    spikesorter.detect_peaks_extract_waveforms(seg_nums = 'all',  threshold=-4, peak_sign = '-', n_span = 2,  n_left=-30, n_right=50)
    print(spikesorter.summary(level=1))
    spikesorter.project(method = 'pca', n_components = 5)
    spikesorter.find_clusters(7)
    return spikesorter


def test_traceviewer():
    app = pg.mkQApp()
    
    spikesorter = get_spikesorter()
    
    traceviewer = TraceViewer(spikesorter=spikesorter, mode = 'memory', )
    traceviewer.show()
    traceviewer.resize(800,600)
    
    app.exec_()
    

def test_traceviewer_linked():
    app = pg.mkQApp()
    spikesorter = get_spikesorter()
    
    traceviewer0 = TraceViewer(spikesorter=spikesorter, mode = 'memory', )
    traceviewer0.show()
    traceviewer0.resize(800,600)

    traceviewer1 = TraceViewer(spikesorter=spikesorter, shared_view_with = [traceviewer0])
    traceviewer1.show()
    traceviewer1.resize(800,600)
    traceviewer0.shared_view_with.append(traceviewer1)
    
    app.exec_()


def test_peaklist():
    app = pg.mkQApp()
    spikesorter = get_spikesorter()
    
    peaklist = PeakList(spikesorter = spikesorter)
    peaklist.show()
    peaklist.resize(800,400)
    
    app.exec_()

def test_ndviewer():
    app = pg.mkQApp()
    spikesorter = get_spikesorter()
    
    ndscatter = NDScatter(spikesorter)
    ndscatter.show()
    
    app.exec_()
    


def test_mainwindow():
    app = pg.mkQApp()
    spikesorter = get_spikesorter()
    
    win = SpikeSortingWindow(spikesorter)
    win.show()
    
    app.exec_()
    
    
if __name__ == '__main__':
    #~ test_traceviewer()
    #~ test_traceviewer_linked()
    #~ test_peaklist()
    test_ndviewer()
    #~ test_mainwindow()
