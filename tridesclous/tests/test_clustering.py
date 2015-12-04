import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns

from tridesclous import DataIO, PeakDetector, WaveformExtractor

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
    print(short_wf.shape)
    
    #clustering
    clustering = Clustering(short_wf)
    
    #PCA
    features = clustering.project(method = 'pca', n_components = 5)
    #~ print(features)

    sns.set(style="white")
    g = sns.PairGrid(features, diag_sharey=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_upper(pyplot.scatter)
    g.map_diag(sns.kdeplot, lw=3)
    
    #Kmean
    labels = clustering.find_clusters(7)
    df = pd.concat([features, labels], axis=1)
    
    #~ hue_kws = {'cmap':['Blues_d','Greens_d']}
    g = sns.PairGrid(df, diag_sharey=False,hue='label', vars = features.columns)
    g.map_lower(sns.kdeplot)
    g.map_upper(pyplot.scatter)
    g.map_diag(sns.kdeplot, lw=3)

    #~ print(labels)
    
    
    
    
if __name__=='__main__':

    test_clustering()
    
    pyplot.show()

