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
    
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    #~ print(spikesorter.dataio.segments)
    
    traceviewer0 = TraceViewer(spikesorter=spikesorter, mode = 'memory', )
    traceviewer0.show()
    traceviewer0.resize(800,600)

    traceviewer1 = TraceViewer(spikesorter=spikesorter, shared_view_with = [traceviewer0])
    traceviewer1.show()
    traceviewer1.resize(800,600)
    traceviewer0.shared_view_with.append(traceviewer1)
    
    app.exec_()

    
    
    
if __name__ == '__main__':
    test_traceviewer()
    #~ test_traceviewer_linked()

