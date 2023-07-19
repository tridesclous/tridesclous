from tridesclous import *

from matplotlib import pyplot
import time

import pytest
from tridesclous.tests.testingtools import ON_CI_CLOUD, setup_catalogue
from tridesclous.gui.tests.testingguitools import HAVE_QT5



if HAVE_QT5:
    import  pyqtgraph as pg
    from tridesclous.gui import *

def setup_module():
    dirname = 'test_peelerwindow'
    setup_catalogue(dirname, dataset_name='olfactory_bulb')
    
    dataio = DataIO(dirname=dirname)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)
    peeler = Peeler(dataio)
    peeler.change_params(catalogue=initial_catalogue, engine='geometrical',
                    chunksize=1024)
    t1 = time.perf_counter()
    peeler.run(progressbar=False)
    t2 = time.perf_counter()
    print('peeler.run_loop', t2-t1)



def get_controller():
    dataio = DataIO(dirname='test_peelerwindow')
    catalogueconstructor = CatalogueConstructor(dataio=dataio)
    initial_catalogue = dataio.load_catalogue()
    controller = PeelerController(dataio=dataio,catalogue=initial_catalogue)
    return controller


def test_Peelercontroller():
    controller = get_controller()
    assert controller.cluster_labels is not None

    
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_PeelerTraceViewer():
    controller = get_controller()
    
    app = pg.mkQApp()
    traceviewer = PeelerTraceViewer(controller=controller)
    traceviewer.show()
    traceviewer.resize(800,600)
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_SpikeList():
    controller = get_controller()
    
    app = pg.mkQApp()
    traceviewer = SpikeList(controller)
    traceviewer.show()
    traceviewer.resize(800,600)
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ClusterSpikeList():
    controller = get_controller()
    
    app = pg.mkQApp()
    traceviewer = ClusterSpikeList(controller)
    traceviewer.show()
    traceviewer.resize(800,600)
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_PeelerWaveformViewer():
    controller = get_controller()
    
    app = pg.mkQApp()
    traceviewer = PeelerWaveformViewer(controller)
    traceviewer.show()
    traceviewer.resize(800,600)
    
    if __name__ == '__main__':
        app.exec_()

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_ISIViewer():
    controller = get_controller()
    for k in controller.cluster_labels:
        controller.cluster_visible[k] = False
    for k in controller.cluster_labels[3:6]:
        controller.cluster_visible[k] = True
    #~ print(controller.cluster_visible)
    
    app = pg.mkQApp()
    isiviewer = ISIViewer(controller)
    isiviewer.show()
    isiviewer.refresh()
    if __name__ == '__main__':
        app.exec_()    

@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_CrossCorrelogramViewer():
    controller = get_controller()
    for k in controller.cluster_labels:
        controller.cluster_visible[k] = False
    for k in controller.cluster_labels[3:6]:
        controller.cluster_visible[k] = True
    #~ print(controller.cluster_visible)
    
    app = pg.mkQApp()
    ccgviewer = CrossCorrelogramViewer(controller)
    ccgviewer.show()
    ccgviewer.refresh()
    if __name__ == '__main__':
        app.exec_()    

    
@pytest.mark.skipif(ON_CI_CLOUD, reason='ON_CI_CLOUD')
def test_PeelerWindow():
    dataio = DataIO(dirname='test_peelerwindow')
    initial_catalogue = dataio.load_catalogue(chan_grp=0)

    app = pg.mkQApp()
    win = PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    if __name__ == '__main__':
        app.exec_()

    
    
if __name__ == '__main__':
    setup_module()
    
    #~ test_Peelercontroller()
    
    #~ test_PeelerTraceViewer()
    #~ test_SpikeList()
    #~ test_ClusterSpikeList()
    #~ test_PeelerWaveformViewer()
    #~ test_ISIViewer()
    #~ test_CrossCorrelogramViewer()
    
    test_PeelerWindow()


