from tridesclous import *
from matplotlib import pyplot

# run test_catalogueconstructor.py before this


import pytest
from tridesclous.tests.testingtools import ON_CI_CLOUD, setup_catalogue
from tridesclous.gui.tests.testingguitools import HAVE_QT5

if HAVE_QT5:
    import  pyqtgraph as pg
    from tridesclous.gui import *


def setup_module():
    dirname = 'test_cataloguewindow'
    setup_catalogue(dirname, dataset_name='olfactory_bulb')
    

def get_controller():
    dataio = DataIO(dirname='test_cataloguewindow')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    controller = CatalogueController(catalogueconstructor=catalogueconstructor)
    return controller


def test_CatalogueController():
    controller = get_controller()
    assert controller.cluster_labels is not None
    #~ print(controller.cluster_labels)


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_CatalogueTraceViewer():
    controller = get_controller()
    app = mkQApp()
    traceviewer = CatalogueTraceViewer(controller=controller, signal_type = 'processed')
    traceviewer.show()
    traceviewer.resize(800,600)
    
    if __name__ == '__main__':
        app.exec_()
    

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_PeakList():
    controller = get_controller()
    
    app = mkQApp()
    peaklist = PeakList(controller=controller)
    peaklist.show()
    peaklist.resize(800,400)
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ClusterPeakList():
    controller = get_controller()
    
    app = mkQApp()
    clusterlist = ClusterPeakList(controller=controller)
    clusterlist.show()
    clusterlist.resize(800,400)
    
    if __name__ == '__main__':
        app.exec_()


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_NDScatter():
    controller = get_controller()
    
    app = mkQApp()
    ndscatter = NDScatter(controller=controller)
    ndscatter.show()
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_WaveformViewer():
    controller = get_controller()
    
    app = mkQApp()
    waveformviewer = WaveformViewer(controller=controller)
    waveformviewer.show()
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_SpikeSimilarityView():
    controller = get_controller()
    #~ controller.compute_spike_waveforms_similarity()
    app = mkQApp()
    similarityview = SpikeSimilarityView(controller=controller)
    similarityview.show()
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ClusterSimilarityView():
    controller = get_controller()
    controller.compute_cluster_similarity()
    app = mkQApp()
    similarityview = ClusterSimilarityView(controller=controller)
    similarityview.show()
    
    if __name__ == '__main__':
        app.exec_()


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ClusterRatioSimilarityView():
    controller = get_controller()
    controller.compute_cluster_ratio_similarity()
    app = mkQApp()
    similarityview = ClusterRatioSimilarityView(controller=controller)
    similarityview.show()
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_PairList():
    controller = get_controller()
    
    app = mkQApp()
    similarityview = PairList(controller=controller)
    similarityview.show()
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_Silhouette():
    controller = get_controller()
    
    controller.compute_spike_silhouette()
    
    app = mkQApp()
    similarityview = Silhouette(controller=controller)
    similarityview.show()
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_WaveformHistViewer():
    controller = get_controller()
    for k in controller.cluster_labels:
        controller.cluster_visible[k] = False
    for k in controller.cluster_labels[:2]:
        controller.cluster_visible[k] = True
    
    app = mkQApp()
    similarityview = WaveformHistViewer(controller=controller)
    similarityview.show()
    
    if __name__ == '__main__': 
        app.exec_()


@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_FeatureTimeViewer():
    controller = get_controller()
    app = mkQApp()
    view = FeatureTimeViewer(controller=controller)
    view.show()
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_CatalogueWindow():
    dataio = DataIO(dirname='test_cataloguewindow')

    cc = CatalogueConstructor(dataio=dataio)
    
    #~ sel = cc.all_peaks['cluster_label'] == -9
    #~ print(np.sum(sel))
    
    app = mkQApp()
    win = CatalogueWindow(cc)
    win.show()
    
    if __name__ == '__main__':
        app.exec_()



    
    
if __name__ == '__main__':
    setup_module()
    
    #~ test_CatalogueController()
    
    #~ test_CatalogueTraceViewer()
    #~ test_PeakList()
    #~ test_ClusterPeakList()
    #~ test_NDScatter()
    #~ test_WaveformViewer()
    #~ test_SpikeSimilarityView()
    #~ test_ClusterSimilarityView()
    #~ test_ClusterRatioSimilarityView()
    #~ test_PairList()
    #~ test_Silhouette()
    #~ test_WaveformHistViewer()
    #~ test_FeatureTimeViewer()
    
    test_CatalogueWindow()



