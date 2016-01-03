import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns

from tridesclous import DataIO, PeakDetector, WaveformExtractor, Clustering, Peeler



def plot_interpolation():
    dataio = DataIO(dirname = 'datatest')
    sigs = dataio.get_signals(seg_num=0)
    
    #peak
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 5)
    
    #waveforms
    waveformextractor = WaveformExtractor(peakdetector, n_left=-30, n_right=50)
    limit_left, limit_right = waveformextractor.find_good_limits(mad_threshold = 1.1)
    #~ print(limit_left, limit_right)
    short_wf = waveformextractor.get_ajusted_waveforms(margin=2)
    #~ print(short_wf.shape)
    
    #clustering
    clustering = Clustering(short_wf)
    features = clustering.project(method = 'pca', n_components = 5)
    clustering.find_clusters(7)
    catalogue = clustering.construct_catalogue()
    
    k = list(catalogue.keys())[1]
    w0 = catalogue[k]['center']
    w1 = catalogue[k]['centerD']
    w2 = catalogue[k]['centerDD']
    
    fig, ax = pyplot.subplots()
    t = np.arange(w0.size)
    
    colors = sns.color_palette('husl', 12)
    
    all = []
    jitters = np.arange(-.5,.5,.1)
    for i, jitter in enumerate(jitters):
        pred = w0 + jitter*w1 + jitter**2/2.*w2
        all.append(pred)
        ax.plot(t+jitter, pred, marker = 'o', label = str(jitter), color = colors[i], linestyle = 'None')
    ax.plot(t, w0, marker = '*', markersize = 4, label = 'w0', lw = 1, color = 'k')     

    all = np.array(all)
    interpolated = all.transpose().flatten()
    t2 = np.arange(interpolated.size)/all.shape[0] + jitters[0]
    ax.plot(t2, interpolated, label = 'interp', lw = 1, color = 'm')

        


def test_peeler():
    dataio = DataIO(dirname = 'datatest')
    #~ dataio = DataIO(dirname = 'datatest_neo')
    
    sigs = dataio.get_signals(seg_num=0)
    
    #peak
    peakdetector = PeakDetector(sigs)
    peak_pos = peakdetector.detect_peaks(threshold=-4, peak_sign = '-', n_span = 5)
    
    #waveforms
    waveformextractor = WaveformExtractor(peakdetector, n_left=-30, n_right=50)
    limit_left, limit_right = waveformextractor.find_good_limits(mad_threshold = 1.1)
    #~ print(limit_left, limit_right)
    short_wf = waveformextractor.get_ajusted_waveforms(margin=2)
    #~ print(short_wf.shape)
    
    #clustering
    clustering = Clustering(short_wf)
    features = clustering.project(method = 'pca', n_components = 4)
    clustering.find_clusters(8, order_clusters = True)
    catalogue = clustering.construct_catalogue()
    #~ clustering.plot_catalogue(sameax = False)
    #~ clustering.plot_catalogue(sameax = True)
    
    #~ clustering.merge_cluster(1, 2)
    catalogue = clustering.construct_catalogue()
    clustering.plot_catalogue(sameax = False)
    #~ clustering.plot_catalogue(sameax = True)
    
    
    #peeler
    signals = peakdetector.normed_sigs
    peeler = Peeler(signals, catalogue,  limit_left, limit_right,
                            threshold=-5., peak_sign = '-', n_span = 5)
    
    prediction0, residuals0 = peeler.peel()
    prediction1, residuals1 = peeler.peel()
    
    fig, axs = pyplot.subplots(nrows = 6, sharex = True)#, sharey = True)
    axs[0].plot(signals)
    axs[1].plot(prediction0) 
    axs[2].plot(residuals0)
    axs[3].plot(prediction1)
    axs[4].plot(residuals1)
    
    for i in range(5):
        axs[i].set_ylim(-25, 10)
    
    colors = sns.color_palette('husl', len(catalogue))
    spiketrains = peeler.get_spiketrains()
    print(spiketrains)
    i = 0
    for k  in peeler.cluster_labels:
        pos = spiketrains[spiketrains['label']==k].index
        axs[5].plot(pos, np.ones(pos.size)*k, ls = 'None', marker = '|',  markeredgecolor = colors[i], markersize = 10, markeredgewidth = 2)
        i += 1
    axs[5].set_ylim(0, len(catalogue))
    #markerfacecolor = colors[i],
    
if __name__=='__main__':
    
    #~ plot_interpolation()
    
    test_peeler()
    
    pyplot.show()
