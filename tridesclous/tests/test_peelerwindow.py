from tridesclous import *
import  pyqtgraph as pg
from matplotlib import pyplot



def test_traceviewer():
    dataio = RawDataIO(dirname='test_peeler')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    initial_catalogue = catalogueconstructor.load_catalogue()
    
    
    app = pg.mkQApp()
    traceviewer = PeelerTraceViewer(catalogue=initial_catalogue, dataio=dataio)
    traceviewer.show()
    traceviewer.resize(800,600)
    app.exec_()
    
#~ def test_peelerwindow():
    #~ app = pg.mkQApp()
    #~ spikesorter = get_spikesorter()
    
    #~ win = PeelerWindow(spikesorter)
    #~ win.show()
    
    #~ app.exec_()

    
    
if __name__ == '__main__':
    test_traceviewer()
    
    #~ test_peelerwindow()
    