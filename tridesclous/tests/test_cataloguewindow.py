from tridesclous import *
import  pyqtgraph as pg
from matplotlib import pyplot

# run test_catalogueconstructor.py before this

def get_catalogueconstructor():
    dataio = RawDataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    catalogueconstructor.on_new_cluster()
    
    return catalogueconstructor


def test_cataloguetraceviewer():
    app = pg.mkQApp()
    
    catalogueconstructor = get_catalogueconstructor()
    
    traceviewer = CatalogueTraceViewer(catalogueconstructor=catalogueconstructor, signal_type = 'processed')
    traceviewer.show()
    traceviewer.resize(800,600)
    
    app.exec_()
    


def test_peaklist():
    app = pg.mkQApp()
    catalogueconstructor = get_catalogueconstructor()
    
    peaklist = PeakList(catalogueconstructor = catalogueconstructor)
    peaklist.show()
    peaklist.resize(800,400)
    
    app.exec_()

def test_clusterlist():
    app = pg.mkQApp()
    catalogueconstructor = get_catalogueconstructor()
    
    clusterlist = ClusterList(catalogueconstructor = catalogueconstructor)
    clusterlist.show()
    clusterlist.resize(800,400)

    app.exec_()

def test_ndscatter():
    app = pg.mkQApp()
    catalogueconstructor = get_catalogueconstructor()
    
    #TODO: remove this
    catalogueconstructor.project()
    
    ndscatter = NDScatter(catalogueconstructor)
    ndscatter.show()
    
    app.exec_()

def test_waveformviewer():
    app = pg.mkQApp()
    catalogueconstructor = get_catalogueconstructor()
    
    waveformviewer = WaveformViewer(catalogueconstructor)
    waveformviewer.show()
    
    app.exec_()






def test_cataloguewindow():
    app = pg.mkQApp()
    catalogueconstructor = get_catalogueconstructor()

    #TODO: remove this
    catalogueconstructor.project(method='pca', n_components=12)
    catalogueconstructor.find_clusters(method='kmeans', n_clusters=12)
    #~ catalogueconstructor.project(method='pca', n_components=5)
    #~ catalogueconstructor.find_clusters(method='gmm', n_clusters=1)
    
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()



    
    
if __name__ == '__main__':
    #~ test_cataloguetraceviewer()
    #~ test_peaklist()
    #~ test_clusterlist()
    #~ test_ndscatter()
    #~ test_waveformviewer()
    
    test_cataloguewindow()

