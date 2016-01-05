import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns

from tridesclous import DataIO, PeakDetector, WaveformExtractor, get_good_events

from tridesclous import Clustering




def test_clustering():
    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    #peak
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 2)
    
    #waveforms
    waveformextractor = WaveformExtractor(peakdetector, n_left=-30, n_right=50)
    limit_left, limit_right = waveformextractor.find_good_limits(mad_threshold = 1.1)
    short_wf = waveformextractor.get_ajusted_waveforms()
    good_events = get_good_events(short_wf, upper_thr=6.,lower_thr=-8.)
    print(short_wf.shape)
    
    #clustering
    clustering = Clustering(short_wf, good_events = good_events)
    
    #PCA
    features = clustering.project(method = 'pca', n_components = 5)
    clustering.plot_explained_variance_ratio()
    clustering.plot_waveform_variance()
    
    #~ clustering.plot_projection(plot_density = True)
    
    #Kmean
    labels = clustering.find_clusters(7)
    
    #~ clustering.plot_projection(plot_density = True)
    
    #make catalogue
    catalogue = clustering.construct_catalogue()
    clustering.plot_derivatives()
    clustering.plot_catalogue()


    clustering.merge_cluster(1,2)
    clustering.split_cluster(1, 2)
    
    dataio.save_catalogue(catalogue, limit_left, limit_right)
    clustering.catalogue, limit_left, limit_right = dataio.get_catalogue()
    clustering.plot_catalogue()
    
    
    

    
    
if __name__=='__main__':

    test_clustering()
    
    pyplot.show()

