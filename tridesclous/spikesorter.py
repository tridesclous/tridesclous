import numpy as np
import pandas as pd
import seaborn as sns

from .dataio import DataIO
from .filter import SignalFilter
from .peakdetector import PeakDetector
from .waveformextractor import WaveformExtractor
from .clustering import Clustering

from .tools import median_mad

from collections import OrderedDict

try:
    from pyqtgraph.Qt import QtCore, QtGui
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

class SpikeSorter:
    def __init__(self, dataio = None, **kargs):
        """
        Main class for spike sorting that encaspulate all other class in the same place:
            * DataIO
            * PeakDetector
            * WaveformExtractor
            * Clustering

        SpikeSorter handle the multi segment the strategy is:
            * take care of PeakDetector one segment per segment
            * take care of WaveformExtractor one segment per segment
            * take care of Clustering all segment at the same time.
        
        Usage:
            spikesorter = SpikeSorter(dataio=DataIO(..))
            or
            spikesorter = SpikeSorter(dirname = 'test', complib = 'blosc', complevel= 9)
                
        
        
        Aruments
        --------------
        same as DataIO
        
        
        """
        if dataio is None:
            self.dataio = DataIO(**kargs)        
        else:
            self.dataio = dataio
        
        self.all_peaks = None
        self.colors = {}
        self.qcolors = {}
    
    def summary(self, level=1):
        t = self.dataio.summary(level=level)
        t += 'Peak Cluster\n'
        if hasattr(self, 'cluster_count'):
            for ind in self.cluster_count.index:
                t += '  #{}: {}\n'.format(ind, self.cluster_count[ind])
        return t

    def __repr__(self):
        return self.summary(level=0)
    
    def apply_filter(self, highpass_freq = 300., seg_nums = 'all'):

        if seg_nums == 'all':
            seg_nums = self.dataio.segments_range.index

        for seg_num in seg_nums:
            sigs = self.dataio.get_signals(seg_num=seg_num, signal_type = 'unfiltered')
            filter =  SignalFilter(sigs, highpass_freq = highpass_freq)
            filtered_sigs = filter.get_filtered_data()
            self.dataio.append_signals(filtered_sigs,  seg_num=seg_num, signal_type = 'filtered')
    
    
    def detect_peaks_extract_waveforms(self, seg_nums = 'all',  
                threshold=-4, peak_sign = '-', n_span = 2,
                n_left=-30, n_right=50):
        if seg_nums == 'all':
            seg_nums = self.dataio.segments_range.index
        
        self.all_waveforms = []
        for seg_num in seg_nums:
            sigs = self.dataio.get_signals(seg_num=seg_num, signal_type = 'filtered')
            
            #peak
            peakdetector = PeakDetector(sigs, seg_num=seg_num)
            peakdetector.detect_peaks(threshold=threshold, peak_sign = peak_sign, n_span = n_span)
            
            #waveform
            waveformextractor = WaveformExtractor(peakdetector, n_left=n_left, n_right=n_right)
            if seg_num==seg_nums[0]:
                #the first segment impose limits to othes
                # this shoudl be FIXED later on
                limit_left, limit_right = waveformextractor.find_good_limits(mad_threshold = 1.1)
            else:
                waveformextractor.limit_left = limit_left
                waveformextractor.limit_right = limit_right
            
            short_wf = waveformextractor.get_ajusted_waveforms()
            self.all_waveforms.append(short_wf)

        self.all_waveforms = pd.concat(self.all_waveforms, axis=0)

        #create a colum to handle selection on UI
        self.clustering = Clustering(self.all_waveforms)
        self.peak_selection = pd.Series(name = 'selected', index = self.all_waveforms.index, dtype = bool)
        self.peak_selection[:] = False
    
    @property
    def peak_labels(self):
        if hasattr(self, 'clustering'):
            return self.clustering.labels

    @property
    def cluster_labels(self):
        if hasattr(self, 'clustering'):
            return self.clustering.cluster_labels
    
    #~ def load_all_peaks(self):
        #~ all = []
        #~ for seg_num in self.dataio.segments.index:
            #~ peaks = self.dataio.get_peaks(seg_num)
            #~ if peaks is not None:
                #~ all.append(peaks)
        #~ if len(all) == 0:
            #~ self.all_peaks = None
        #~ else:
            #~ self.all_peaks = pd.concat(all, axis=0)
            #~ #create a colum to handle selection on UI
            #~ self.all_peaks['selected'] = False

    def project(self, *args, **kargs):
        self.clustering.project(*args, **kargs)
    
    def find_clusters(self, *args, **kargs):
        self.clustering.find_clusters(*args, **kargs)
        self.on_new_cluster()
    
    def order_cluster(self):
        self.clustering.order_cluster()
        self.on_new_cluster()
    
    def on_new_cluster(self):
        if self.peak_labels is None: return
        
        self.clustering.reset()
        self.cluster_count = self.peak_labels.groupby(self.peak_labels).count()
        
        if not hasattr(self, 'cluster_visible'):
            self.cluster_visible = pd.Series(index = self.cluster_labels, name = 'visible')
            self.cluster_visible[:] = True
        for k in self.cluster_labels:
            if k not in self.cluster_visible:
                self.cluster_visible.loc[k] = True
    
    def refresh_colors(self, reset = True, palette = 'husl'):
        if self.peak_labels is None: return
        
        if reset:
            self.colors = {}
        
        self.on_new_cluster()
        
        n = self.cluster_labels.size
        color_table = sns.color_palette(palette, n)
        for i, k in enumerate(self.cluster_labels):
            if k not in self.colors:
                self.colors[k] = color_table[i]
        
        self.colors[-1] = (.4, .4, .4)
        
        if HAVE_QT:
            self.qcolors = {}
            for k, color in self.colors.items():
                r, g, b = color
                self.qcolors[k] = QtGui.QColor(r*255, g*255, b*255)
    
    def construct_catalogue(self):
        self.clustering.construct_catalogue()
    
    
    
    
