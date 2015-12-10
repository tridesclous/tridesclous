import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns

from tridesclous import DataIO, PeakDetector, WaveformExtractor, Clustering, Peeler




def test_peeler():
    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    #peak
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 2)
    
    #waveforms
    waveformextractor = WaveformExtractor(peakdetector, n_left=-30, n_right=50)
    limit_left, limit_right = waveformextractor.find_good_limits(mad_threshold = 1.1)
    short_wf = waveformextractor.get_ajusted_waveforms()
    print(short_wf.shape)
    
    #clustering
    clustering = Clustering(short_wf)
    features = clustering.project(method = 'pca', n_components = 5)
    clustering.find_clusters(7)
    catalogue = clustering.construct_catalogue()
    #~ print(catalogue)
    #peeler
    peeler = Peeler()
    
    
    

    
if __name__=='__main__':

    test_peeler()
    
    pyplot.show()