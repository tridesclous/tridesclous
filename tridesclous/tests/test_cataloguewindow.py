from tridesclous import *
import  pyqtgraph as pg
from matplotlib import pyplot

# run test_catalogueconstructor.py before this


def get_controller():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    controller = CatalogueController(catalogueconstructor=catalogueconstructor)
    return controller


def test_CatalogueController():
    controller = get_controller()
    assert controller.cluster_labels is not None
    #~ print(controller.cluster_labels)



def test_CatalogueTraceViewer():
    controller = get_controller()
    app = mkQApp()
    traceviewer = CatalogueTraceViewer(controller=controller, signal_type = 'processed')
    traceviewer.show()
    traceviewer.resize(800,600)
    
    app.exec_()
    


def test_PeakList():
    controller = get_controller()
    
    app = mkQApp()
    peaklist = PeakList(controller=controller)
    peaklist.show()
    peaklist.resize(800,400)
    
    app.exec_()

def test_ClusterPeakList():
    controller = get_controller()
    
    app = mkQApp()
    clusterlist = ClusterPeakList(controller=controller)
    clusterlist.show()
    clusterlist.resize(800,400)
    
    app.exec_()

def test_NDScatter():
    controller = get_controller()
    controller.project()
    
    app = mkQApp()
    ndscatter = NDScatter(controller=controller)
    ndscatter.show()
    
    app.exec_()

def test_WaveformViewer():
    controller = get_controller()
    
    app = mkQApp()
    waveformviewer = WaveformViewer(controller=controller)
    waveformviewer.show()
    
    app.exec_()


def test_SimilarityView():
    controller = get_controller()
    
    app = mkQApp()
    similarityview = SimilarityView(controller=controller)
    similarityview.show()
    app.exec_()

def test_PairList():
    controller = get_controller()
    
    app = mkQApp()
    similarityview = PairList(controller=controller)
    similarityview.show()
    app.exec_()


def test_Silhouette():
    controller = get_controller()
    
    app = mkQApp()
    similarityview = Silhouette(controller=controller)
    similarityview.show()
    app.exec_()

def test_WaveformHistViewer():
    controller = get_controller()
    for k in controller.cluster_labels:
        controller.cluster_visible[k] = False
    for k in controller.cluster_labels[:2]:
        controller.cluster_visible[k] = True
    
    app = mkQApp()
    similarityview = WaveformHistViewer(controller=controller)
    similarityview.show()
    app.exec_()


def test_CatalogueWindow():
    dataio = DataIO(dirname='test_catalogueconstructor')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    
    app = mkQApp()
    win = CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()



    
    
if __name__ == '__main__':
    #~ test_CatalogueController()
    
    #~ test_CatalogueTraceViewer()
    #~ test_PeakList()
    #~ test_ClusterPeakList()
    #~ test_NDScatter()
    test_WaveformViewer()
    #~ test_SimilarityView()
    #~ test_PairList()
    #~ test_Silhouette()
    #~ test_WaveformHistViewer()
    
    #~ test_CatalogueWindow()



