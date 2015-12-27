from tridesclous import *
import  pyqtgraph as pg
from matplotlib import pyplot

def get_spikesorter():
    #~ spikesorter = SpikeSorter(dirname = '../../tests/datatest')
    spikesorter = SpikeSorter(dirname = '../../tests/datatest_neo')
    #~ print(spikesorter.summary(level=1))
    spikesorter.detect_peaks_extract_waveforms(seg_nums = 'all',  threshold=-5.,
                            peak_sign = '-', n_span = 2,  n_left=-30, n_right=50)
    print(spikesorter.summary(level=1))
    spikesorter.project(method = 'pca', n_components = 5)
    spikesorter.find_clusters(12)
    spikesorter.refresh_colors(reset=True, palette = 'husl')
    #~ print(spikesorter.cluster_labels)
    #~ print(spikesorter.cluster_count)
    #~ print(spikesorter.summary(level=1))

    spikesorter.clustering.construct_catalogue()
    spikesorter.clustering.plot_catalogue()
    pyplot.show()    
    
    return spikesorter


def test_traceviewer():
    app = pg.mkQApp()
    
    spikesorter = get_spikesorter()
    
    traceviewer = TraceViewer(spikesorter=spikesorter, mode = 'memory', signal_type = 'filtered')
    traceviewer.show()
    traceviewer.resize(800,600)
    
    app.exec_()
    

def test_traceviewer_linked():
    app = pg.mkQApp()
    spikesorter = get_spikesorter()
    
    traceviewer0 = TraceViewer(spikesorter=spikesorter, mode = 'memory', signal_type = 'filtered')
    traceviewer0.show()
    traceviewer0.resize(800,600)

    traceviewer1 = TraceViewer(spikesorter=spikesorter, shared_view_with = [traceviewer0], signal_type = 'unfiltered')
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

def test_clusterlist():
    app = pg.mkQApp()
    spikesorter = get_spikesorter()
    
    clusterlist = ClusterList(spikesorter = spikesorter)
    clusterlist.show()
    clusterlist.resize(800,400)

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


def test_from_classes():
    app = pg.mkQApp()
    
    dataio = DataIO(dirname = '../../tests/datatest')
    sigs = dataio.get_signals(seg_num=0)
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 5)
    waveformextractor = WaveformExtractor(peakdetector, n_left=-30, n_right=50)
    limit_left, limit_right = waveformextractor.find_good_limits(mad_threshold = 1.1)
    short_wf = waveformextractor.get_ajusted_waveforms(margin=2)
    clustering = Clustering(short_wf)
    features = clustering.project(method = 'pca', n_components = 5)
    clustering.find_clusters(7)
    catalogue = clustering.construct_catalogue()
    
    
    win = SpikeSortingWindow.from_classes(dataio, peakdetector, waveformextractor, clustering)
    win.show()
    
    app.exec_()    
    
    
if __name__ == '__main__':
    #~ test_traceviewer()
    #~ test_traceviewer_linked()
    #~ test_peaklist()
    #~ test_clusterlist()
    #~ test_ndviewer()
    
    test_mainwindow()
    #~ test_from_classes()
