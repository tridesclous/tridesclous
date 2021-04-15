import PyQt5 # this force pyqtgraph to deal with Qt5

# For matplotlib to Qt5 : 
#   * this avoid tinker problem when not installed
#   * work better with GUI
#   * trigger a warning on notebook
import matplotlib
import warnings
with warnings.catch_warnings():
    try:                                                                                                                                                                                                                                    
        warnings.simplefilter("ignore")
        matplotlib.use('Qt5Agg')                                                                                                                                                                                                            
    except:
        # on server without screen this is not possible.
        pass

from .myqt import QT,mkQApp

#for catalogue window
from .cataloguecontroller import CatalogueController
from .traceviewer import CatalogueTraceViewer
from .peaklists import PeakList, ClusterPeakList
from .ndscatter import NDScatter
from .waveformviewer import WaveformViewer
from .similarity import SpikeSimilarityView, ClusterSimilarityView, ClusterRatioSimilarityView
from .pairlist import PairList
from .silhouette import Silhouette
from .waveformhistviewer import WaveformHistViewer
from .featuretimeviewer import FeatureTimeViewer

from .cataloguewindow import CatalogueWindow

#for peeler window
from .peelercontroller import PeelerController
from .traceviewer import PeelerTraceViewer
from .spikelists import SpikeList, ClusterSpikeList
from .waveformviewer import PeelerWaveformViewer
from .isiviewer import ISIViewer
from .crosscorrelogramviewer import CrossCorrelogramViewer

from .peelerwindow import PeelerWindow


#main window
from .mainwindow import MainWindow
from .initializedatasetwindow import InitializeDatasetWindow, ChannelGroupWidget


from .probegeometryview import ProbeGeometryView
from .gpuselector import GpuSelector