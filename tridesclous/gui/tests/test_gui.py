from tridesclous import *
import  pyqtgraph as pg

def test_traceviewer():
    app = pg.mkQApp()
    
    spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    #~ print(spikesorter)
    #~ print(spikesorter.dataio.segments)
    
    traceviewer = TraceViewer(spikesorter=spikesorter, mode = 'memory', )
    #~ traceviewer = TraceViewer(spikesorter=spikesorter, mode = 'file', )
    traceviewer.show()
    traceviewer.resize(800,600)

    #~ traceviewer1 = TraceViewer(spikesorter=spikesorter, shared_view_with = [traceviewer])
    #~ traceviewer1.show()
    #~ traceviewer1.resize(800,600)
    
    #~ traceviewer.shared_view_with.append(traceviewer1)
    
    
    app.exec_()
    
    
    
    
if __name__ == '__main__':
    test_traceviewer()

