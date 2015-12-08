from tridesclous import *
import  pyqtgraph as pg



def test_traceviewer():
    app = pg.mkQApp()
    
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    
    traceviewer = TraceViewer(spikesorter=spikesorter, mode = 'memory', )
    traceviewer.show()
    traceviewer.resize(800,600)
    
    app.exec_()
    

def test_traceviewer_linked():
    app = pg.mkQApp()
    
    
    #~ print(spikesorter.dataio.segments)
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    
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
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    
    peaklist = PeakList(spikesorter = spikesorter)
    peaklist.show()
    peaklist.resize(800,400)
    
    app.exec_()


def test_mainwindow():
    app = pg.mkQApp()
    
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    
    peaklist = SpikeSortingWindow(spikesorter)
    peaklist.show()
    
    app.exec_()
    
    
if __name__ == '__main__':
    #~ test_traceviewer()
    #~ test_traceviewer_linked()
    #~ test_peaklist()
    test_mainwindow()
