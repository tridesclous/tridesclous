import numpy as np
import os
import time

import sklearn
import sklearn.cluster
import sklearn.mixture
import sklearn.metrics

import scipy.signal
import scipy.stats
from sklearn.neighbors import KernelDensity

from . import labelcodes
from .tools import median_mad


import matplotlib.pyplot as plt
from .dip import diptest

from .sawchaincut import SawChainCut
from .pruningshears import PruningShears

#~ import pyclustering

import hdbscan

try:
    import isosplit5
    HAVE_ISOSPLIT5 = True
except:
    HAVE_ISOSPLIT5 = False



def find_clusters(catalogueconstructor, method='kmeans', selection=None, **kargs):
    """
    selection is mask bool size all_peaks
    """
    
    cc = catalogueconstructor
    
    if selection is None:
        # this include trash but not allien
        sel = (cc.all_peaks['cluster_label']>=-1)[cc.some_peaks_index]
        if np.all(sel):
            features = cc.some_features[:]
            waveforms = cc.some_waveforms[:]
            
        else:
            # this can be very long because copy!!!!!
            # TODO fix this for high channel count
            features = cc.some_features[sel]
            waveforms = cc.some_waveforms[sel]
        
        peaks = cc.all_peaks[cc.some_peaks_index]
        peaks = peaks[peaks['cluster_label']>=-1]
        detection_channel_indexes = peaks['channel'].copy()
        
    else:
        sel = selection[cc.some_peaks_index]
        features = cc.some_features[sel]
        waveforms = cc.some_waveforms[sel]
        detection_channel_indexes = cc.all_peaks['channel'][selection]
    
    if waveforms.shape[0] == 0:
        #print('oupas waveforms vide')
        return
    
    if method == 'kmeans':
        km = sklearn.cluster.KMeans(n_clusters=kargs.pop('n_clusters'), **kargs)
        labels = km.fit_predict(features)
        
    elif method == 'onecluster':
        labels = np.zeros(features.shape[0], dtype='int64')
        
    elif method == 'gmm':
        gmm = sklearn.mixture.GaussianMixture(n_components=kargs.pop('n_clusters'), **kargs)
        #~ labels =gmm.fit_predict(features)
        gmm.fit(features)
        labels =gmm.predict(features)
    #~ elif method == 'kmedois':
        #~ import pyclust
        #~ km = pyclust.KMedoids(n_clusters=kargs.pop('n_clusters'))
        #~ labels = km.fit_predict(features)
    elif method == 'agglomerative':
        agg = sklearn.cluster.AgglomerativeClustering(n_clusters=kargs.pop('n_clusters'), **kargs)
        labels = agg.fit_predict(features)
    elif method == 'dbscan':
        if 'eps' not in kargs:
            kargs['eps'] = 3
        if 'metric' not in kargs:
            kargs['metric'] = 'euclidean'
        if 'algorithm' not in kargs:
            kargs['algorithm'] = 'brute'
        #~ print('DBSCAN', kargs)
        dbscan = sklearn.cluster.DBSCAN(**kargs)
        labels = dbscan.fit_predict(features)
    elif method == 'hdbscan':
        if 'min_cluster_size' not in kargs:
            kargs['min_cluster_size'] = 20
        clusterer = hdbscan.HDBSCAN(**kargs)
        labels = clusterer.fit_predict(features)
        
    elif  method == 'optics':
        optic = sklearn.cluster.OPTICS(**kargs)
        labels = optic.fit_predict(features)
        
    elif  method == 'meanshift':
        ms = sklearn.cluster.MeanShift()
        labels = ms.fit_predict(features)
        
    elif method == 'sawchaincut':
        n_left = cc.info['waveform_extractor_params']['n_left']
        n_right = cc.info['waveform_extractor_params']['n_right']
        peak_sign = cc.info['peak_detector_params']['peak_sign']
        relative_threshold = cc.info['peak_detector_params']['relative_threshold']
        sawchaincut = SawChainCut(waveforms, n_left, n_right, peak_sign, relative_threshold, **kargs)
        labels = sawchaincut.do_the_job()
        
    elif method == 'pruningshears':
        n_left = cc.info['waveform_extractor_params']['n_left']
        n_right = cc.info['waveform_extractor_params']['n_right']
        peak_sign = cc.info['peak_detector_params']['peak_sign']
        relative_threshold = cc.info['peak_detector_params']['relative_threshold']
        dense_mode = cc.info['mode'] == 'dense'
        adjacency_radius_um = cc.adjacency_radius_um * 0.5 # TODO wokr on this
        channel_adjacency = cc.dataio.get_channel_adjacency(chan_grp=cc.chan_grp, adjacency_radius_um=adjacency_radius_um)
        channel_distances = cc.dataio.get_channel_distances(chan_grp=cc.chan_grp)
        
        noise_features = cc.some_noise_features
        
        
        #~ adjacency_radius_um = 200
        pruningshears = PruningShears(waveforms, features, detection_channel_indexes, noise_features, n_left, n_right, peak_sign, relative_threshold,
                                adjacency_radius_um, channel_adjacency, channel_distances, dense_mode, **kargs)
        
        #~ from .pruningshears import PruningShears_1_4_1
        #~ pruningshears = PruningShears_1_4_1(waveforms, features, noise_features, n_left, n_right, peak_sign, relative_threshold,
                                #~ adjacency_radius_um, channel_adjacency, channel_distances, dense_mode, **kargs)
        
        
        
        
        labels = pruningshears.do_the_job()
        
    elif method =='isosplit5':
        assert HAVE_ISOSPLIT5, 'isosplit5 is not installed'
        labels = isosplit5.isosplit5(features.T)
    else:
        raise(ValueError, 'find_clusters method unknown')
    
    if selection is None:
        #~ cc.all_peaks['cluster_label'][:] = labelcodes.LABEL_NO_WAVEFORM
        #~ cc.all_peaks['cluster_label'][cc.some_peaks_index] = labels
        cc.all_peaks['cluster_label'][cc.some_peaks_index[sel]] = labels
        
    else:
        labels[labels>=0] += max(max(cc.cluster_labels), -1) + 1
        cc.all_peaks['cluster_label'][cc.some_peaks_index[sel]] = labels
    
    
    return labels












