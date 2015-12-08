import numpy as np
import pandas as pd

from .dataio import DataIO
from .peakdetector import PeakDetector

from collections import OrderedDict

class SpikeSorter:
    def __init__(self, dataio = None, **kargs):
        """
        Main class for spike sorting that encaspulate all other class in the same place:
            * DataIO
            * PeakDetector
            * WaveformExtractor
            * Clustering
        
        usage:
        spikesorter = SpikeSorter(dataio=DataIO(..))
        or
        spikesorter = SpikeSorter(dirname = 'test', complib = 'blosc', complevel= 9)
        
        SpikeSorter handle the multi segment the strategy is:
            * take care of PeakDetector one segment per segment
            * take care of WaveformExtractor one segment per segment
            * take care of Clustering all segment at the same time.
        
        Aruments
        --------------
        same as DataIO
        
        
        """
        if dataio is None:
            self.dataio = DataIO(**kargs)        
        else:
            self.dataio = dataio
        
        self.all_peaks = None

    def summary(self, level=1):
        t = self.dataio.summary(level=level)
        return t

    def __repr__(self):
        return self.summary(level=0)
    
    def detect_peaks(self, seg_nums = 'all',  threshold=-4, peak_sign = '-', n_span = 2):
        if seg_nums == 'all':
            seg_nums = self.dataio.segments.index
        
        for seg_num in seg_nums:
            sigs = self.dataio.get_signals(seg_num=seg_num)
            peakdetector = PeakDetector(sigs, seg_num=seg_num)
            peakdetector.detect_peaks(threshold=threshold, peak_sign = peak_sign, n_span = n_span)
            
            peaks = pd.DataFrame(np.zeros(peakdetector.peak_index.size, dtype = 'int32'), columns = ['label'], index = peakdetector.peak_index)
            self.dataio.append_peaks(peaks,seg_num = seg_num, append = False)
    
    def load_all_peaks(self):
        self.all_peaks = []
        for seg_num in self.dataio.segments.index:
            peaks = self.dataio.get_peaks(seg_num)
            if peaks is not None:
                self.all_peaks.append(peaks)
        self.all_peaks = pd.concat(self.all_peaks, axis=0)
        
        #create a colum to handle selection on UI
        self.all_peaks['selected'] = False

