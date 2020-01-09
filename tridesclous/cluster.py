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

#~ import pyclustering

import hdbscan

try:
    import isosplit5
    HAVE_ISOSPLIT5 = True
except:
    HAVE_ISOSPLIT5 = False



def find_clusters(catalogueconstructor, method='kmeans', selection=None, **kargs):
    
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
    else:
        sel = selection[cc.some_peaks_index]
        features = cc.some_features[sel]
        waveforms = cc.some_waveforms[sel]
    
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
        
        pruningshears = PruningShears(waveforms, features, noise_features, n_left, n_right, peak_sign, relative_threshold,
                                adjacency_radius_um, channel_adjacency, channel_distances, dense_mode, **kargs)
        
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




class SawChainCut:
    def __init__(self, waveforms, n_left, n_right, peak_sign, threshold,
                    
                    
                    nb_min=20,
                    max_loop=1000,
                    break_nb_remain=30,
                    kde_bandwith=1.0,
                    auto_merge_threshold=2.,
                    print_debug=False):
        self.waveforms = waveforms
        self.n_left = n_left
        self.n_right = n_right
        self.width = n_right - n_left
        self.peak_sign = peak_sign
        self.threshold = threshold
        
        #user params
        self.print_debug = print_debug
        self.nb_min = nb_min
        self.max_loop = max_loop
        self.break_nb_remain = break_nb_remain
        self.kde_bandwith = kde_bandwith
        self.auto_merge_threshold = auto_merge_threshold

        # magic params that could be set to user
        self.binsize = 0.1
        self.minima_rejection_factor = .3
        self.threshold_margin = 1.
        self.margin_first_max = 1.
    
    def log(self, *args, **kargs):
        if self.print_debug:
            print(*args, **kargs)
    
    def do_the_job(self):
        
        cluster_labels = self.split_loop()
        cluster_labels = self.auto_merge_loop(cluster_labels)
        
        return cluster_labels
    
    def one_cut(self, x):

        #~ x = x[x>(thresh-threshold_margin)]
        #~ kde = scipy.stats.gaussian_kde(x, bw_method=kde_bandwith)
        #~ d = kde(bins)
        #~ d /= np.sum(d)
        
        kde = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwith)
        d = kde.fit(x[:, np.newaxis]).score_samples(self.bins[:, np.newaxis])
        d = np.exp(d)



        #local max
        d0, d1, d2 = d[:-2], d[1:-1], d[2:]
        #~ ind_max,  = np.nonzero((d0<d1) & (d2<d1))
        ind_max,  = np.nonzero((d0<d1) & (d2<=d1))
        ind_max += 1
        #~ ind_min,  = np.nonzero((d0>d1) & (d2>d1))
        ind_min,  = np.nonzero((d0>d1) & (d2>=d1))
        ind_min += 1
        
        #~ print('ind_max', ind_max)
        #~ print('ind_min', ind_min)
        #~ fig, ax = plt.subplots()
        #~ ax.plot(d)
        #~ ax.plot(ind_max, d[ind_max], ls='None', marker='o', color='r')
        #~ ax.plot(ind_min, d[ind_min], ls='None', marker='o', color='g')
        #~ plt.show()

        if ind_max.size>0:
            if ind_min.size==0:
                assert ind_max.size==1, 'Super louche pas de min mais plusieur max'
                ind_min = np.array([0, self.bins.size-1], dtype='int64')
            else:
                ind_min = ind_min.tolist()
                if ind_max[0]<ind_min[0]:
                    ind_min = [0] + ind_min
                if ind_max[-1]>ind_min[-1]:
                    ind_min = ind_min + [ self.bins.size-1]
                ind_min = np.array(ind_min, dtype='int64')
        
        
        #Loop reject small rebounce minimam/maxima
        #~ print('loop1')
        ind_max_cleaned = ind_max.tolist()
        ind_min_cleaned = ind_min.tolist()
        while True:
            rejected_minima = None
            rejected_maxima = None
            #~ print('ind_min_cleaned', ind_min_cleaned, self.bins[ind_min_cleaned])
            #~ print('ind_max_cleaned', ind_max_cleaned, self.bins[ind_max_cleaned])
            for i, ind in enumerate(ind_min_cleaned[1:-1]):
                prev_max = ind_max_cleaned[i]
                next_max = ind_max_cleaned[i+1]
                
                delta_density_prev = d[prev_max] - d[ind]
                delta_density_next = d[next_max] - d[ind]
                
                if min(delta_density_prev, delta_density_next)<d[ind]*self.minima_rejection_factor:
                    rejected_minima = ind
                    if delta_density_prev<delta_density_next:
                        rejected_maxima = prev_max
                    else:
                        rejected_maxima = next_max
                    break
            
            if rejected_minima is None:
                break
            
            ind_max_cleaned.remove(rejected_maxima)
            ind_min_cleaned.remove(rejected_minima)
        
        #~ print('loop2')
        #loop reject density with too few spikes
        while True:
            rejected_minima = None
            rejected_maxima = None
            
            #~ print('ind_min_cleaned', ind_min_cleaned, self.bins[ind_min_cleaned])
            #~ print('ind_max_cleaned', ind_max_cleaned, self.bins[ind_max_cleaned])
            
            for i, ind in enumerate(ind_min_cleaned[:-1]):
                next_min = ind_min_cleaned[i+1]
                n = np.sum(d[ind:next_min]*self.binsize) * x.size
                #~ print('n', n, self.bins[ind], self.bins[next_min], np.sum(d))
                if n<self.nb_min:
                    rejected_maxima = ind_max_cleaned[i]
                    if d[ind]<d[next_min]:
                        rejected_minima = next_min
                    else:
                        rejected_minima = ind
                    break
            
            if rejected_minima is None:
                break
            
            ind_max_cleaned.remove(rejected_maxima)
            ind_min_cleaned.remove(rejected_minima)
            

        #~ print('loop3')
        #TODO eliminate first avec meme critere loop 1
        if len(ind_min_cleaned)>=2:
            den_min0 = d[ind_min_cleaned[0]]
            den_max0 = d[ind_max_cleaned[0]]
            if (den_max0-den_min0)<den_min0*self.minima_rejection_factor:
                ind_min_cleaned = ind_min_cleaned[1:]
                ind_max_cleaned = ind_max_cleaned[1:]
        
        #~ print('loop4')
        if len(ind_min_cleaned)>=2:
            if self.bins[ind_max_cleaned[0]]<self.threshold+self.margin_first_max:
                ind_min_cleaned = ind_min_cleaned[1:]
                ind_max_cleaned = ind_max_cleaned[1:]
        
        
        if len(ind_min_cleaned)>=2:
            #TODO here criterium for best
            return self.bins[ind_min_cleaned[-2]], self.bins[ind_min_cleaned[-1]], d
        else:
            return None, None, d
        
    
    def split_loop(self):
        ind_peak = -self.n_left
        all_peak_max = self.waveforms[:, ind_peak, : ].copy()
        if self.peak_sign == '-' :
            all_peak_max = -all_peak_max
        
        nb_channel = self.waveforms.shape[2]
        self.bins = np.arange(self.threshold, np.max(all_peak_max),  self.binsize)
        
        cluster_labels = np.zeros(self.waveforms.shape[0], dtype='int64')
        k = 0
        chan_visited = []
        #~ for iloop in range(self.max_loop):
        iloop = -1
        while True:
            iloop += 1
            
            nb_remain = np.sum(cluster_labels>=k)
            sel = cluster_labels == k
            nb_working = np.sum(sel)
            self.log()
            self.log('iloop', iloop, 'k', k, 'nb_remain', nb_remain, 'nb_working', nb_working, {True:'full', False:'partial'}[nb_remain==nb_working])
            
            if iloop>=self.max_loop:
                cluster_labels[sel] = -1
                self.log('BREAK iloop', iloop)
                print('Warning SawChainCut reach max_loop limit there are maybe more cluster')
                break
            
            if iloop!=0 and nb_remain<self.break_nb_remain:
                cluster_labels[sel] = -1
                self.log('BREAK nb_remain', nb_remain, '<', self.break_nb_remain)
                break
            
            #~ self.log(all_peak_max.shape)
            peak_max = all_peak_max[sel, :]
            
            
            if nb_working<self.nb_min:
                self.log('TRASH: too few')
                cluster_labels[sel] = -1
                k += 1
                chan_visited = []
                continue
            
            percentiles = np.zeros(nb_channel)
            for c in range(nb_channel):
                x = peak_max[:, c]
                x = x[x>self.threshold]
                
                if x.size>self.nb_min:
                    per = np.nanpercentile(x, 90)
                else:
                    per = 0
                percentiles[c] = per
            order_visit = np.argsort(percentiles)[::-1]
            order_visit = order_visit[percentiles[order_visit]>0]
            
            #~ self.log(order_visit)
            #~ self.log(percentiles[order_visit])
            
            
            order_visit = order_visit[~np.in1d(order_visit, chan_visited)]
            
            if len(order_visit)==0:
                self.log('len(order_visit)==0')
                if np.sum(cluster_labels>k)>0:
                    k+=1
                    chan_visited = []
                    continue
                else:
                    cluster_labels[sel] = -1
                    self.log('BREAK no  more channel')
                    break
            
            actual_chan = order_visit[0]
            x = peak_max[:, actual_chan]
            x = x[x>(self.threshold-self.threshold_margin)]
            lim0, lim1, density_ = self.one_cut(x)
            #~ self.log('lim0, lim1', lim0, lim1)
            
            #~ if True:
            if False:
                if not os.path.exists('debug_sawchaincut'):
                    os.mkdir('debug_sawchaincut')
                
                if hasattr(self, 'n_cut'):
                    self.n_cut += 1
                else:
                    self.n_cut = 0
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(np.arange(self.smooth_kernel.size)*self.binsize, self.smooth_kernel)
                    #~ fig.savefig('debug_sawchaincut/smooth_kernel.png')
                
                count, _ = np.histogram(x, bins=self.bins)
                count = count.astype(float)/np.sum(count)
                
                filename = 'debug_sawchaincut/one_cut {}.png'.format(self.n_cut)
                fig, ax = plt.subplots()
                
                ax.set_title('nb_remain {}, nb_working {},  chan {} , x.size {}, k {} {}'.format(nb_remain, nb_working, actual_chan, x.size, k, {True:'full', False:'partial'}[nb_remain==nb_working]))
                ax.plot(self.bins[:-1], count, color='b')
                ax.plot(self.bins, density_, color='k')
                
                ax.set_xlim(0, self.bins[-1]+.5)
                ax.set_ylim(0,max(density_)*1.2)
                
                if lim0 is not None:
                    ax.axvline(lim0, color='k')
                    ax.axvline(lim1, color='k')
                
                fig.savefig(filename)
                
                #~ plt.show()
            
            #~ self.log(feat[labels==0].size, feat[labels==1].size)
            
            
            if lim0 is None:
                chan_visited.append(actual_chan)
                self.log('EXPLORE NEW DIM lim0 is None ',  len(chan_visited))
                continue
            
            
            ind, = np.nonzero(sel)
            #~ self.log(ind.shape)
            not_in = ~((peak_max[:, actual_chan]>lim0) & (peak_max[:, actual_chan]<lim1))
            #~ self.log(not_in.shape)
            cluster_labels[cluster_labels>k] += 1#TODO reflechir la dessus!!!
            cluster_labels[ind[not_in]] += 1

            if np.sum(not_in)==0:
                self.log('ACCEPT: not_in.sum()==0')
                k+=1
                chan_visited = []
                continue
        
        self.log('END loop', np.sum(cluster_labels==-1))
        return cluster_labels

    def auto_merge_loop(self, cluster_labels):
        cluster_labels2 = cluster_labels.copy()
        
        while True:
            self.log('')
            self.log('new loop')
            labels = np.unique(cluster_labels2)
            labels = labels[labels>=0]
            n = labels.size
            self.log(labels)
            centroids = np.zeros((labels.size, self.waveforms.shape[1], self.waveforms.shape[2]))
            
            
            max_per_cluster = 300
            for ind, k in enumerate(labels):
                ind_keep,  = np.nonzero(cluster_labels2 == k)
                if ind_keep.size > max_per_cluster:
                    sub_sel = np.random.choice(ind_keep.size, max_per_cluster, replace=False)
                    ind_keep = ind_keep[sub_sel]
                centroids[ind,:,:] = np.median(self.waveforms[ind_keep, :, :], axis=0)
            
            nb_merge = 0
            for i in range(n):
                for j in range(i+1, n):
                    k1, k2 = labels[i], labels[j]
                    wf1, wf2 = centroids[i, :, :], centroids[j, :, :]
                    d = np.max(np.abs(wf1-wf2))
                    if d<self.auto_merge_threshold:
                        self.log('d', d)
                        self.log('merge', k1, k2)
                        
                        cluster_labels2[cluster_labels2==k2] = k1
                        
                        nb_merge += 1
            
            self.log('nb_merge', nb_merge)
            if nb_merge == 0:
                break
        
        return cluster_labels2


class PruningShears:
    def __init__(self, waveforms, 
                        features,
                        noise_features,
                        n_left, n_right,
                        peak_sign, threshold,
                        adjacency_radius_um,
                        channel_adjacency,
                        channel_distances,
                        dense_mode,
                        
                        min_cluster_size=20,
                        max_loop=1000,
                        break_nb_remain=30,
                        auto_merge_threshold=2.,
                        max_per_cluster_for_median = 300,
                        
                        print_debug=False):
        
        self.waveforms = waveforms
        self.features = features
        self.noise_features = noise_features
        self.n_left = n_left
        self.n_right = n_right
        self.width = n_right - n_left
        self.peak_sign = peak_sign
        self.threshold = threshold
        self.adjacency_radius_um = adjacency_radius_um
        self.channel_adjacency = channel_adjacency
        self.channel_distances = channel_distances
        self.dense_mode =  dense_mode
        
        #user params
        self.min_cluster_size = min_cluster_size
        self.max_loop = max_loop
        self.break_nb_remain = break_nb_remain
        self.auto_merge_threshold = auto_merge_threshold
        self.print_debug = print_debug
        self.max_per_cluster_for_median = max_per_cluster_for_median
        
        #~ self.debug_plot = False
        self.debug_plot = True

    def log(self, *args, **kargs):
        if self.print_debug:
            print(*args, **kargs)
    
    def do_the_job(self):
        cluster_labels = self.explore_split_loop()
        cluster_labels = self.merge_and_clean(cluster_labels)
        return cluster_labels
    
    def next_channel(self, peak_max, chan_visited):
        self.log('next_channel percentiles', 'peak_max.size', peak_max.shape)
        self.log('chan_visited', chan_visited)
        nb_channel = self.waveforms.shape[2]
        percentiles = np.zeros(nb_channel)
        for c in range(nb_channel):
            x = peak_max[:, c]
            x = x[x>self.threshold]
            
            if x.size>self.min_cluster_size:
                #~ per = np.nanpercentile(x, 99)
                per = np.nanpercentile(x, 95)
            else:
                per = 0
            percentiles[c] = per
        order_visit = np.argsort(percentiles)[::-1]
        #~ print('percentiles', percentiles)
        order_visit = order_visit[percentiles[order_visit]>0]
        self.log('order_visit :5', order_visit[:5])
        self.log('percentiles :5', percentiles[order_visit[:5]])
        
        
        
        
        #~ self.log(order_visit)
        #~ self.log(percentiles[order_visit])
        order_visit = order_visit[~np.in1d(order_visit, chan_visited)]
        
        if len(order_visit)==0:
            self.log('len(order_visit)==0')
            return None
            #~ if np.sum(cluster_labels>k)>0:
                #~ k+=1
                #~ chan_visited = []
                #~ continue
            #~ else:
                #~ cluster_labels[mask_loop] = -1
                #~ self.log('BREAK no  more channel')
                #~ break
        
        actual_chan = order_visit[0]        
        
        return actual_chan
    
    def one_sub_cluster(self, local_data):
        # clustering on one channel or one adjacency
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        local_labels = clusterer.fit_predict(local_data)
        
        # try isosplit here not stable enought on windows
        #~ local_labels = isosplit5.isosplit5(local_data.T)
        
        return local_labels
    
    def select_candidate_labels(self, local_ind, local_labels, local_channels):
        # keep only cluster when:
        #   * template peak is center on self.n_left
        #   * cluster size is bug enoutgh

        candidate_labels = []
        possible_peak_values =  []
        possible_chan_peak =  []
        local_centroids = {}
        
        for l, label in enumerate(np.unique(local_labels)):

            if label<0: 
                continue

            ind = local_ind[label == local_labels]
            cluster_size = ind.size
            if ind.size > self.max_per_cluster_for_median:
                sub_sel = np.random.choice(ind.size, self.max_per_cluster_for_median, replace=False)
                ind = ind[sub_sel]
            
            if self.dense_mode:
                centroid = np.median(self.waveforms[ind, :, :], axis=0)
            else:
                centroid = np.median(self.waveforms[ind, :, :][:, :, local_channels], axis=0)
            
            local_centroids[label] = centroid
            
            if self.peak_sign == '-':
                chan_peak_local = np.argmin(np.min(centroid, axis=0))
                pos_peak = np.argmin(centroid[:, chan_peak_local])
            elif self.peak_sign == '+':
                chan_peak_local = np.argmax(np.max(centroid, axis=0))
                pos_peak = np.argmax(centroid[:, chan_peak_local])
            
            if self.dense_mode:
                chan_peak = chan_peak_local
            else:
                chan_peak = local_channels[chan_peak_local]
            
            peak_is_aligned = np.abs(-self.n_left - pos_peak) <= 1
            self.log('-self.n_left', -self.n_left, 'pos_peak', pos_peak, 'peak_is_aligned', peak_is_aligned)

            possible_peak_values.append(np.abs(centroid[pos_peak, chan_peak_local]))
            possible_chan_peak.append(chan_peak)
            
            self.log('label', label, 'chan_peak', chan_peak, 'chan_peak_local', chan_peak_local, 'peak_is_aligned', peak_is_aligned, 'peak values', np.abs(centroid[-self.n_left, chan_peak_local]), 'cluster_size', cluster_size)
            if peak_is_aligned and cluster_size>self.min_cluster_size:
                candidate_labels.append(label)
        
        return candidate_labels, possible_peak_values, possible_chan_peak, local_centroids
    
    
    def explore_split_loop(self):
        #~ print('all_peak_max')
        ind_peak = -self.n_left
        all_peak_max = self.waveforms[:, ind_peak, : ].copy()
        if self.peak_sign == '-' :
            all_peak_max = -all_peak_max
        
        #~ print('all_peak_max done')
        
        nb_channel = self.waveforms.shape[2]
        
        if self.dense_mode:
            # global pca so no sparse used
            pass
            
        else:
            # general case 
            n_components_by_channel = self.features.shape[1] // nb_channel
            
            weights_feat_chans = {}
            mask_feat_per_chan = {}
            mask_feat_per_adj = {}
            self.mask_feat_per_adj = mask_feat_per_adj
            for chan in range(nb_channel):
                adjacency = self.channel_adjacency[chan]
                
                # mask feature per channel
                mask_feat = np.zeros(self.features.shape[1], dtype='bool')
                mask_feat[chan*n_components_by_channel:(chan+1)*n_components_by_channel] = True
                mask_feat_per_chan[chan] = mask_feat
                
                # mask_feat per channel adjacency
                mask_feat = np.zeros(self.features.shape[1], dtype='bool')
                for i in range(n_components_by_channel):
                    mask_feat[adjacency*n_components_by_channel+i] = True
                mask_feat_per_adj[chan] = mask_feat
                
                # weights
                weights = []
                for adj_chan in adjacency:
                    d = self.channel_distances[chan, adj_chan]
                    w = np.exp(-d/self.adjacency_radius_um * 2) # TODO fix this factor 2
                    #~ print('chan', chan, 'adj_chan', adj_chan, 'd', d, 'w', w)
                    weights += [ w ] * n_components_by_channel
                weights_feat_chans[chan] = np.array(weights).reshape(1, -1)
        
        force_next_chan = None
        
        cluster_labels = np.zeros(self.waveforms.shape[0], dtype='int64')
        k = 0
        chan_visited = []
        #~ for iloop in range(self.max_loop):
        iloop = -1
        while True:
            iloop += 1
            
            nb_remain = np.sum(cluster_labels>=k)
            mask_loop = cluster_labels == k
            nb_working = np.sum(mask_loop)
            self.log()
            self.log('iloop', iloop, 'k', k, 'nb_remain', nb_remain, 'nb_working', nb_working, {True:'full', False:'partial'}[nb_remain==nb_working])
            
            if iloop>=self.max_loop:
                cluster_labels[mask_loop] = -1
                self.log('BREAK iloop', iloop)
                print('Warning PruningShears reach max_loop limit there are maybe more cluster')
                break
            
            if iloop!=0 and nb_remain<self.break_nb_remain:
                cluster_labels[mask_loop] = -1
                self.log('BREAK nb_remain', nb_remain, '<', self.break_nb_remain)
                break
            
            #~ self.log(all_peak_max.shape)
            peak_max = all_peak_max[mask_loop, :]
            
            if nb_working<self.min_cluster_size:
                self.log('TRASH: too few')
                cluster_labels[mask_loop] = -1
                k += 1
                #~ chan_visited = []
                force_next_chan = None
                continue
            
            if force_next_chan is None:
                actual_chan = self.next_channel(peak_max, chan_visited)
            elif force_next_chan is not None and force_next_chan in chan_visited:
                # special case when after a force channel no cluster is found
                self.log('impossible force', force_next_chan, 'chan_visited', chan_visited)
                #~ raise()
                actual_chan = self.next_channel(peak_max, chan_visited)
            else:
                actual_chan = force_next_chan
                self.log('force', actual_chan)
                
            
            if actual_chan is None:
                if np.sum(cluster_labels>k)>0:
                    k+=1
                    chan_visited = []
                    force_next_chan = None
                    continue
                else:
                    cluster_labels[mask_loop] = -1
                    self.log('BREAK actual_chan None')
                    break
            
            
            
            self.log('actual_chan', actual_chan)
            adjacency = self.channel_adjacency[actual_chan]
            self.log('adjacency', adjacency)
            #~ exit()
            
            
            #~ chan_features = self.features[:, actual_chan*n_components_by_channel:(actual_chan+1)*n_components_by_channel]
            
            mask_thresh = np.zeros(mask_loop.size, dtype='bool') # TODO before loop
            mask_thresh[mask_loop] = peak_max[:, actual_chan] > self.threshold
            
            ind_keep,  = np.nonzero(mask_loop & mask_thresh)
            # TODO avoid infinite loop
            #~ if 
            
            self.log('mask_thresh.size', mask_thresh.size, 'keep.sum', mask_thresh.sum())

            #~ mask_feat = np.zeros(self.features.shape[1], dtype='bool')
            #~ for i in range(n_components_by_channel):
                #~ mask_feat[adjacency*n_components_by_channel+i] = True
            
            
            # Step2: cluster on the channel only
            if self.dense_mode:
                channel_wf_features = self.features.take(ind_keep, axis=0)
                channel_noise_features = self.noise_features
            else:
                mask_feat = mask_feat_per_chan[actual_chan]
                channel_wf_features = self.features.take(ind_keep, axis=0).compress(mask_feat, axis=1)
                channel_noise_features = self.noise_features.compress(mask_feat, axis=1)
            
            self.log('channel_wf_features.shape', channel_wf_features.shape)
            
            channel_features = np.concatenate([channel_noise_features, channel_wf_features, ], axis=0)
            channel_labels = self.one_sub_cluster(channel_features)
            # remove noise label
            channel_labels = channel_labels[self.noise_features.shape[0]:]
            possible_labels = np.unique(channel_labels)
            possible_labels = possible_labels[possible_labels>=0]
            
            self.log('possible_labels', possible_labels)
            
            local_ind = ind_keep
            
            if len(possible_labels) == 0:
                self.log('len(possible_labels) == 0, explore new chan ')
                chan_visited.append(actual_chan)
                force_next_chan = None
                
                candidate_labels = None
                local_labels = None
                candidate_labels_l2 = None
                ind_keep_l2 = None
                self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                continue

            # explore candidate
            
            if self.dense_mode:
                local_channels = np.arange(nb_channel, dtype='int64')
            else:
                local_channels = [actual_chan]
            candidate_labels, possible_peak_values, possible_chan_peak, local_centroids= self.select_candidate_labels(ind_keep, channel_labels, local_channels)
            
            self.log('candidate_labels one channel', candidate_labels)

            # for not dense mode check if the best label is on anotehr channel
            if not self.dense_mode:
                ind_best = np.argmax(possible_peak_values)
                if possible_chan_peak[ind_best] != actual_chan:
                    
                    if  force_next_chan is not None:
                        self.log('2 consecutive force chan')
                        chan_visited.append(actual_chan)
                        force_next_chan = None
                        self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                        continue
                        
                    if len(chan_visited) and possible_chan_peak[ind_best] in chan_visited:
                        # prevent ping pong
                        if possible_peak_values[ind_best] not in candidate_labels:
                            self.log('prevent ping pong force chan')
                            chan_visited.append(actual_chan)
                            force_next_chan = None
                            self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                            continue
                    
                    self.log('Force channel exploration to channel', possible_chan_peak[ind_best] ,'actual_chan', actual_chan )
                    self.log('peak val', possible_peak_values[ind_best])
                    chan_visited.append(actual_chan)
                    force_next_chan = possible_chan_peak[ind_best]
                    
                    local_labels = None
                    candidate_labels_l2 = None
                    ind_keep_l2 = None
                    self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                    continue

            if len(candidate_labels) == 0:
                # TODO : do something else dense_mode=True
                self.log('len(candidate_labels) == 0')
                chan_visited.append(actual_chan)
                force_next_chan = None
                
                local_labels = None
                candidate_labels_l2 = None
                ind_keep_l2 = None
                self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                continue
            
            
            if self.dense_mode:
                raise(NotIMplementedError)
                
            else:
                # Step 3: do it on channel adjacency
                candidate_peak_values = [v for k, v in zip(possible_labels, possible_peak_values) if k in candidate_labels]
                candidate_chan_peak = [v for k, v in zip(possible_labels, possible_chan_peak) if k in candidate_labels]

                #~ ind_best = np.argmax(possible_peak_values)
                #~ best_label = possible_labels[ind_best]
                best_label = candidate_labels[np.argmax(candidate_peak_values)]
                self.log('best_label', best_label, 'peak_val', candidate_peak_values[np.argmax(candidate_peak_values)])
                
                
                ind_keep_l2 = ind_keep[best_label == channel_labels]

                mask_feat = mask_feat_per_adj[actual_chan]
                local_wf_features = self.features.take(ind_keep_l2, axis=0).compress(mask_feat, axis=1)
                local_noise_features = self.noise_features.compress(mask_feat, axis=1)
                self.log('local_wf_features.shape', local_wf_features.shape)
                
                # test weithed PCA
                # TODO put n_components to parameters
                
                
                # weihted
                #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=False)
                #~ local_wf_features_w = local_wf_features * weights_feat_chans[actual_chan]
                #~ m = local_wf_features_w.mean(axis=0)
                #~ local_wf_features_w -= m
                #~ local_noise_features_w = local_noise_features * weights_feat_chans[actual_chan]
                #~ local_noise_features_w -= m
                #~ pca.fit(local_wf_features_w)
                #~ reduced_features = np.concatenate([pca.transform(local_noise_features_w), pca.transform(local_wf_features_w)], axis=0)
                
                # not weighted
                #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=True)
                #~ local_wf_features_w = local_wf_features.copy()
                #~ local_noise_features_w = local_noise_features.copy()
                #~ reduced_features = pca.fit_transform(np.concatenate([local_noise_features_w, local_wf_features_w], axis=0))

                # not weihted not whiten
                #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=False)
                #~ local_wf_features_w = local_wf_features.copy()
                #~ m = local_wf_features_w.mean(axis=0)
                #~ local_wf_features_w -= m
                #~ local_noise_features_w = local_noise_features.copy()
                #~ local_noise_features_w -= m
                #~ pca.fit(local_wf_features_w)
                #~ reduced_features = np.concatenate([pca.transform(local_noise_features_w), pca.transform(local_wf_features_w)], axis=0)
                
                
                pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=True)
                reduced_features = pca.fit_transform(local_wf_features)
                
                
                self.log('reduced_features.shape', reduced_features.shape)
                
                local_labels = self.one_sub_cluster(reduced_features)
                #~ local_labels = local_labels[self.noise_features.shape[0]:]
                possible_labels_l2 = np.unique(local_labels)
                if possible_labels_l2.size == 1 and possible_labels_l2[0] == -1:
                    local_labels[:] = 0
                    possible_labels_l2 = np.unique(local_labels)
                    # TODO : put to trash every wf meadian +/- 5 MAD
                    
                possible_labels_l2 = possible_labels_l2[possible_labels_l2>=0]
                
                #~ if len(possible_labels_l2) == 0:
                    # Impossible because the trash is transform into one label
                    #~ self.log('len(possible_labels_l2) == 0 ')
                    #~ chan_visited.append(actual_chan)
                    #~ force_next_chan = None
                    #~ candidate_labels_l2 = None
                    #~ self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                    #~ raise(NotImplementedError)
                    #~ continue
                
                
                self.log('possible_labels_l2 adjacency channels', possible_labels_l2)

                candidate_labels_l2, possible_peak_values, possible_chan_peak, local_centroids= self.select_candidate_labels(ind_keep_l2, local_labels, adjacency)
                
                self.log('candidate_labels_l2', candidate_labels_l2)

                ind_best_l2 = np.argmax(possible_peak_values)
                if possible_chan_peak[ind_best_l2] != actual_chan:
                    
                    if force_next_chan is not None:
                        chan_visited.append(actual_chan)
                        force_next_chan = None
                        
                        self.log('2 consecutive force chan l2')
                        self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                        continue

                    if len(chan_visited) and possible_chan_peak[ind_best_l2] in chan_visited:
                        # prevent ping pong
                        if possible_chan_peak[ind_best_l2] not in candidate_labels_l2:
                            chan_visited.append(actual_chan)
                            force_next_chan = None
                            
                            self.log('prevent ping pong force chan l2')
                            self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                            continue

                    
                    self.log('Force channel exploration to channel', possible_chan_peak[ind_best_l2] ,'actual_chan', actual_chan )
                    self.log('peak val', possible_peak_values[ind_best_l2])
                    chan_visited.append(actual_chan)
                    force_next_chan = possible_chan_peak[ind_best_l2]
                    
                    self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                    continue
                
                if len(candidate_labels_l2) == 0:
                    #~ self.log('EXPLORE NEW DIM lim0 is None ',  len(chan_visited))
                    self.log('len(candidate_labels_l2) == 0 ')
                    chan_visited.append(actual_chan)
                    force_next_chan = None
                    
                    self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                    continue
                
                candidate_peak_values = [v for k, v in zip(possible_labels_l2, possible_peak_values) if k in candidate_labels_l2]
                candidate_chan_peak = [v for k, v in zip(possible_labels_l2, possible_chan_peak) if k in candidate_labels_l2]
                
                ind_best = np.argmax(candidate_peak_values)
                if candidate_chan_peak[ind_best] != actual_chan:
                    self.log('Force channel exploration adjacency', candidate_chan_peak[ind_best] , actual_chan )
                    self.log('peak val', candidate_peak_values[ind_best])
                    chan_visited.append(actual_chan)
                    force_next_chan = candidate_chan_peak[ind_best]
                    self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                    continue
                
                self.log('Best channel OK',  candidate_chan_peak[ind_best] , actual_chan )
                
                final_label = candidate_labels_l2[np.argmax(candidate_peak_values)]
                self.log('final_label', final_label)
                
                
                # remove trash
                ind_trash_label = ind_keep_l2[local_labels == -1]
                self.log('ind_trash_label.shape', ind_trash_label.shape)
                cluster_labels[ind_trash_label] = -1
                
                ind_new_label = ind_keep_l2[final_label == local_labels]
                
                self.log('ind_new_label.shape', ind_new_label.shape)

                cluster_labels[cluster_labels>=k] += 1
                cluster_labels[ind_new_label] -= 1
                
                k += 1
                chan_visited = []
                force_next_chan = None
                
                self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                continue



                

            
            
            
            
            """
            #######################
            
            
            if self.dense_mode:
                local_features = self.features.take(ind_keep, axis=0)
                self.log('local_features.shape', local_features.shape)
                pca =  sklearn.decomposition.IncrementalPCA(n_components=local_features.shape[1], whiten=False)
                #~ reduced_features = pca.fit_transform(local_features)
                
                
                local_noise_features = self.noise_features
                
                m = local_features.mean(axis=0)
                local_noise_features = self.noise_features - m
                pca.fit(local_features)
                reduced_features = np.concatenate([pca.transform(local_noise_features), pca.transform(local_features)], axis=0)
                
                
                self.log('reduced_features.shape', reduced_features.shape)
                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(reduced_features.T, color='k', alpha=0.3)
                #~ plt.show()
                
            else:
                mask_feat = mask_feat_per_adj[actual_chan]
                #~ local_features = self.features[ind_keep, :][:, mask_feat]
                local_features = self.features.take(ind_keep, axis=0).compress(mask_feat, axis=1)
                local_noise_features = self.noise_features.compress(mask_feat, axis=1)
                #~ print('local_noise_features.shape', local_noise_features.shape)
                #~ print(local_noise_features)

                self.log('local_features.shape', local_features.shape)
                
                # TODO put n_components to parameters
                #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=True)
                #~ reduced_features = pca.fit_transform(local_features)
                #~ self.log('reduced_features.shape',reduced_features.shape)
                
                # test weithed PCA
                # TODO put n_components to parameters
                pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=False)
                #~ print(weights_feat_chans[actual_chan])
                local_features_w = local_features * weights_feat_chans[actual_chan]
                m = local_features_w.mean(axis=0)
                local_features_w -= m
                
                local_noise_features_w = local_noise_features* weights_feat_chans[actual_chan]
                #~ local_noise_features_w -= local_noise_features_w.mean(axis=0)
                local_noise_features_w -= m
                
                #~ reduced_features = pca.fit_transform(local_features_w)
                
                pca.fit(local_features_w)
                reduced_features = np.concatenate([pca.transform(local_noise_features_w), pca.transform(local_features_w)], axis=0)
                #~ reduced_features = pca.fit_transform(np.concatenate([local_noise_features_w, local_features_w], axis=0))
                
                
                
                self.log('reduced_features.shape', reduced_features.shape)
                
                #~ fig, ax = plt.subplots()
                #~ ax.plot(local_features_w.T, color='k', alpha=0.3)
                #~ fig, ax = plt.subplots()
                #~ ax.plot(reduced_features.T, color='k', alpha=0.3)
                #~ plt.show()
            
            
            local_labels = self.one_sub_cluster(reduced_features)
            # remove noise label
            local_labels = local_labels[self.noise_features.shape[0]:]
            
            
            
            possible_labels = np.unique(local_labels)
            
            candidate_labels = []
            candidate_peak_values =  []
            candidate_chan_peak =  []
            # keep only cluster when:
            #   * template peak is center on self.n_left
            #   * template extremum channel is same as actual chan
            #~ fig, ax = plt.subplots()
            local_centroids = {}
            max_per_cluster_for_median = 300
            
            for l, label in enumerate(possible_labels):

                if label<0: 
                    continue

                ind = ind_keep[label == local_labels]
                cluster_size = ind.size
                if ind.size > max_per_cluster_for_median:
                    sub_sel = np.random.choice(ind.size, max_per_cluster_for_median, replace=False)
                    ind = ind[sub_sel]
                
                if self.dense_mode:
                    centroid_adj = np.median(self.waveforms[ind, :, :], axis=0)
                else:
                    centroid_adj = np.median(self.waveforms[ind, :, :][:, :, adjacency], axis=0)
                
                local_centroids[label] = centroid_adj
                
                    
                if self.peak_sign == '-':
                    chan_peak_local = np.argmin(np.min(centroid_adj, axis=0))
                    pos_peak = np.argmin(centroid_adj[:, chan_peak_local])
                elif self.peak_sign == '+':
                    chan_peak_local = np.argmax(np.max(centroid, axis=0))
                    pos_peak = np.argmax(centroid_adj[:, chan_peak_local])
                
                if self.dense_mode:
                    chan_peak = chan_peak_local
                else:
                    chan_peak = adjacency[chan_peak_local]
                #~ print('actual_chan', actual_chan, 'chan_peak', chan_peak)
                #~ print('adjacency', len(adjacency), adjacency)
                #~ print('keep it', np.abs(-self.n_left - pos_peak)<=1)
                
                #~ ax.plot(centroid_adj.T.flatten())
                
                peak_is_aligned = np.abs(-self.n_left - pos_peak) <= 1
                self.log('-self.n_left', -self.n_left, 'pos_peak', pos_peak, 'peak_is_aligned', peak_is_aligned)
                
                self.log('label', label, 'chan_peak', chan_peak, 'chan_peak_local', chan_peak_local, 'peak_is_aligned', peak_is_aligned, 'peak values', np.abs(centroid_adj[-self.n_left, chan_peak_local]), 'cluster_size', cluster_size)
                if peak_is_aligned and cluster_size>self.min_cluster_size:
                    candidate_labels.append(label)
                    candidate_peak_values.append(np.abs(centroid_adj[-self.n_left, chan_peak_local]))
                    candidate_chan_peak.append(chan_peak)
            #~ plt.show()


            
            #~ print(candidate_labels)
            #~ print(candidate_peak_values)
            
            
            # 2 piste:
            #  * garder seulement tres grand peak avant DBSCAN
            #  * garder seulement les template dont le peak est sur n_left  DONE
            
            
            #~ print(np.unique(local_labels))
            
            #~ if iloop >42:
                #~ self.debug_plot = True
            
            if self.debug_plot:
                self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)

            
            
            if force_next_chan is not None and len(candidate_labels) ==0 and\
                            len(possible_labels) ==1 and possible_labels[0] == -1:
            #~ if len(candidate_labels) ==0 and\
                            #~ len(possible_labels) ==1 and possible_labels[0] == -1:
                # sometimes even after a force channel exploration
                # the trash have one good cluster but fail into trash because
                # of hdbscan
                # in that case
                # we keep it with condition of peak value and the diptest
                pval = diptest(np.sort(reduced_features[:, 0]), numt=200)
                
                #~ centroid_adj = local_centroids[-1]
                
                ind = ind_keep
                if ind.size > max_per_cluster_for_median:
                    sub_sel = np.random.choice(ind.size, max_per_cluster_for_median, replace=False)
                    ind = ind[sub_sel]
                
                if self.dense_mode:
                    centroid_adj = np.median(self.waveforms[ind, :, :], axis=0)
                else:
                    centroid_adj = np.median(self.waveforms[ind, :, :][:, :, adjacency], axis=0)

                
                
                if self.peak_sign == '-':
                    chan_peak_local = np.argmin(np.min(centroid_adj, axis=0))
                    pos_peak = np.argmin(centroid_adj[:, chan_peak_local])
                elif self.peak_sign == '+':
                    chan_peak_local = np.argmax(np.max(centroid, axis=0))
                    pos_peak = np.argmax(centroid_adj[:, chan_peak_local])
                
                if self.dense_mode:
                    chan_peak = chan_peak_local
                else:
                    chan_peak = adjacency[chan_peak_local]
                
                peak_is_aligned = np.abs(-self.n_left - pos_peak) <= 1
                
                abs_peak_val = np.abs(centroid_adj[-self.n_left, chan_peak_local])
                
                self.log('Try save trash', 'peak_is_aligned', peak_is_aligned, 'pval', pval, 'chan_peak', chan_peak, 'actual_chan', actual_chan)
                if peak_is_aligned and actual_chan==chan_peak and abs_peak_val>self.threshold and pval>0.2 and len(local_labels)>self.min_cluster_size:
                    self.log('Trash saved !!!!!!!')
                    
                    candidate_labels = [0]
                    local_labels[:] = 0
                    candidate_peak_values = [abs_peak_val]
                    candidate_chan_peak = [chan_peak]
            

            self.log('possible_labels', possible_labels)
            self.log('candidate_labels', candidate_labels)
            
            #~ if len(candidate_labels) == 0 and force_next_chan is not None:
                #~ print('len(candidate_labels) == 0 and force_next_chan is not None')
                #~ raise()
                
            
            if len(candidate_labels) == 0:
                
                #~ self.log('EXPLORE NEW DIM lim0 is None ',  len(chan_visited))
                self.log('EXPLORE NEW DIM no label ')
                if self.debug_plot:
                    plt.show()
                
                chan_visited.append(actual_chan)
                force_next_chan = None
                continue
            
            ind_best = np.argmax(candidate_peak_values)
            if candidate_chan_peak[ind_best] != actual_chan:
                self.log('Force channel exploration', candidate_chan_peak[ind_best] , actual_chan )
                self.log('peak val', candidate_peak_values[ind_best])
                if self.debug_plot:
                    plt.show()
                chan_visited.append(actual_chan)
                force_next_chan = candidate_chan_peak[ind_best]
                continue
            
            self.log('Best channel OK',  candidate_chan_peak[ind_best] , actual_chan )
            
            best_label = candidate_labels[ind_best]
            self.log('best_label', best_label)
            
            
            # remove trash
            ind_trash_label = ind_keep[local_labels == -1]
            self.log('ind_trash_label.shape', ind_trash_label.shape)
            cluster_labels[ind_trash_label] = -1
            
            
            # TODO merge when HDSCAN FAIL!!!!!!!!
            # sometime hdbscan split a  clear cluster in many very small
            #~ d = np.max(np.abs(wf1-wf2))
            #~ if d<self.auto_merge_threshold:
            
            ind_new_label = ind_keep[best_label == local_labels]

            self.log('ind_new_label.shape', ind_new_label.shape)

            if self.debug_plot:
                self._plot_debug(actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2)
                
            cluster_labels[cluster_labels>=k] += 1
            cluster_labels[ind_new_label] -= 1
            
            k += 1
            chan_visited = []
            force_next_chan = None
            
            """
            
        self.log('END loop', np.sum(cluster_labels==-1))
        return cluster_labels


    def _plot_debug(self, actual_chan, channel_wf_features, channel_noise_features, channel_labels, possible_labels, candidate_labels, ind_keep, reduced_features, local_labels, candidate_labels_l2, ind_keep_l2):
        if not self.debug_plot:
            return
        
        from .matplotlibplot import plot_waveforms_density
        fig, axs = plt.subplots(ncols=3, nrows=2)
        #~ plt.show()
        
        colors = plt.cm.get_cmap('jet', len(np.unique(channel_labels)))
        colors2 = plt.cm.get_cmap('tab10', len(np.unique(local_labels)))
        
        # channel feature
        ax = axs[0, 0]
        
        feat0, feat1 = channel_wf_features[:, 0], channel_wf_features[:, 1]
        for l, label in enumerate(possible_labels):
            #~ if label not in candidate_labels:
                #~ continue
                
            sel = label == channel_labels
            if label<0: 
                color = 'k'
            else:
                color=colors(l)
            
            if label in candidate_labels:
                s = 3
            else:
                s = 1
            
            ax.scatter(feat0[sel], feat1[sel], s=s, color=color)
        ax.scatter(channel_noise_features[:, 0], channel_noise_features[:, 1], s=1, color='r')        
        
        if local_labels is not None:
            ax = axs[0, 1]
            
            
            noise_size = self.noise_features.shape[0]
            noise_size = 0
            ax.scatter(reduced_features[noise_size:, 0], reduced_features[noise_size:, 1], s=1, color='black')
            ax.scatter(reduced_features[:noise_size, 0], reduced_features[:noise_size, 1], s=1, color='r')

            for l, label in enumerate(np.unique(local_labels)):
                sel = label == local_labels
                if label is None:
                    # THIS IS A DEBUG
                    print('label', label)
                    print('local_labels', local_labels)
                    
                if label<0: 
                    color = 'k'
                else:
                    color=colors2(l)
                if candidate_labels_l2 is not None and label in candidate_labels_l2:
                    s = 3
                else:
                    s = 1
                ax.scatter(reduced_features[noise_size:, :][sel, :][:, 0], reduced_features[noise_size:, :][sel, :][:, 1], s=s, color=color)

        
        
        ax = axs[1, 0]
        wf_chan = self.waveforms[:,:, [actual_chan]][ind_keep, :, :]
        
        #~ bin_min, bin_max, bin_size =np.min(wf_chan)-3, np.max(wf_chan)+3, 0.2
        #~ im = plot_waveforms_density(wf_chan, bin_min, bin_max, bin_size, ax=ax)
        #~ im.set_clim(0, 50)
        #~ fig.colorbar(im)
        
        if wf_chan.shape[0] > 400:
            wf_chan = wf_chan[:400, :, :]
        ax.plot(wf_chan.swapaxes(1,2).reshape(wf_chan.shape[0], -1).T, color='k', alpha=0.1)
        
        for l, label in enumerate(possible_labels):
        
            sel = label == channel_labels
            ind = ind_keep[sel]
            wf = self.waveforms[:,:, actual_chan][ind, :].mean(axis=0)
            
            if label<0: 
                color = 'k'
            else:
                color=colors(l)
            
            if candidate_labels is not None and label in candidate_labels:
                ls = '-'
            else:
                ls = '--'
            
            ax.plot(wf, color=color, ls=ls)
        
        
        #~ ax.axvline(-self.n_left, color='w', lw=2)        
        ax.axvline(-self.n_left, color='m', lw=2)
        
        
        
        ax = axs[1, 1]
        if ind_keep_l2 is not None:
            mask_feat = self.mask_feat_per_adj[actual_chan]
            local_wf_features = self.features.take(ind_keep_l2, axis=0).compress(mask_feat, axis=1)
            if local_wf_features.shape[0] > 400:
                local_wf_features = local_wf_features[:400, :]
            ax.plot(local_wf_features.T, color='k', alpha=0.1)
            
            local_wf_features = self.features.take(ind_keep_l2, axis=0).compress(mask_feat, axis=1)
            
            if local_labels is not None:
                for l, label in enumerate(np.unique(local_labels)):
                    if label<0: 
                        continue
                    sel = label == local_labels
                    m = local_wf_features[sel].mean(axis=0)

                    color=colors2(l)
                    ax.plot(m, color=color, alpha=1)
        
        
        ax = axs[1, 2]
        adjacency = self.channel_adjacency[actual_chan]
        wf_adj = self.waveforms[:,:, adjacency][ind_keep_l2, :, :]
        if wf_adj.shape[0] > 400:
            wf_adj = wf_adj[:400, :, :]
        ax.plot(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).T, color='k', alpha=0.1)
        
        if local_labels is not None:
            for l, label in enumerate(np.unique(local_labels)):
                if label<0: 
                    continue
                sel = label == local_labels
                m = self.waveforms[:,:, adjacency][ind_keep_l2, :, :][sel].mean(axis=0)

                color=colors2(l)
                ax.plot(m.T.flatten(), color=color, alpha=1)
            
        
        
        

        plt.show()
        
        return
        

    def merge_and_clean(self, cluster_labels):
        # merge : 
        #   * max distance < auto_merge_threshold
        #   * 2 cluster have a shift
        # delete:
        #   * peak is not aligneds
        
        cluster_labels2 = cluster_labels.copy()
        
        while True:
            self.log('')
            self.log('new loop')
            labels = np.unique(cluster_labels2)
            labels = labels[labels>=0]
            n = labels.size
            self.log(labels)
            
            centroids = np.zeros((labels.size, self.waveforms.shape[1], self.waveforms.shape[2]))
            max_per_cluster = 300
            for ind, k in enumerate(labels):
                ind_keep,  = np.nonzero(cluster_labels2 == k)
                if ind_keep.size > max_per_cluster:
                    sub_sel = np.random.choice(ind_keep.size, max_per_cluster, replace=False)
                    ind_keep = ind_keep[sub_sel]
                centroids[ind,:,:] = np.median(self.waveforms[ind_keep, :, :], axis=0)
            
            #eliminate when best peak not aligned
            # TODO move this in the main loop!!!!!
            for ind, k in enumerate(labels):
                centroid = centroids[ind,:,:]
                if self.peak_sign == '-':
                    chan_peak = np.argmin(np.min(centroid, axis=0))
                    pos_peak = np.argmin(centroid[:, chan_peak])
                elif self.peak_sign == '+':
                    chan_peak = np.argmax(np.max(centroid, axis=0))
                    pos_peak = np.argmax(centroid[:, chan_peak])
                
                if np.abs(-self.n_left - pos_peak)>2:
                    #delete
                    cluster_labels2[cluster_labels2 == k] = -1
                    labels[ind] = -1
                    
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(centroid.T.flatten())
                    #~ for i in range(centroids.shape[2]):
                        #~ ax.axvline(i*self.width-self.n_left)
                    #~ ax.set_title('delete')
                    #~ plt.show()
                
            
            n_shift = 2
            nb_merge = 0
            for i in range(n):
                k1 = labels[i]
                if k1 == -1:
                    continue
                wf1 = centroids[i, n_shift:-n_shift, :]
                for j in range(i+1, n):
                    k2 = labels[j]
                    if k2 == -1:
                        continue
                        
                    for shift in range(n_shift*2+1):
                        wf2 = centroids[j, shift:wf1.shape[0]+shift, :]
                        d = np.max(np.abs(wf1-wf2))
                        if d<self.auto_merge_threshold:
                            self.log('d', d)
                            self.log('merge', k1, k2)
                            cluster_labels2[cluster_labels2==k2] = k1
                            nb_merge += 1
                            #~ fig, ax = plt.subplots()
                            #~ ax.plot(wf1.T.flatten())
                            #~ ax.plot(wf2.T.flatten())
                            #~ ax.set_title('merge')
                            #~ plt.show()
                            break
                    
                    
            
            self.log('nb_merge', nb_merge)
            if nb_merge == 0:
                break
        
        return cluster_labels2

# TODO : 
# pour aller vite prendre seulement le dernier percentile quand beaucoup de spike et grand percentile.
# adjency à 2 niveau pour eviter les flat sur les bort  >>>>>> adjency_radius * 0.5
# nettoyage quand le max n'est pas sur n_left
# over merge

# quand chan_max diffrent actual_chan alors on recommence sur chan_max

# nouvelle idee : idem avant mais garde le max meme si pas aligné!!!


# nouvelle idée global:
# hdbscan sur PCA channel + bruit
# garde meilleur peak
# si template max_chan==actual_chan alors pca sur voisinage uniquement sel + bruit

# TODO
# faire un dip test avant le level2 local









