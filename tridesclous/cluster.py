import numpy as np
import os

import sklearn
import sklearn.cluster
import sklearn.mixture
import sklearn.metrics

import scipy.signal
import scipy.stats

from . import labelcodes
from .tools import median_mad




def find_clusters(catalogueconstructor, method='kmeans', selection=None, n_clusters=1, **kargs):
    cc = catalogueconstructor
    
    if selection is None:
        features = cc.some_features
        waveforms = cc.some_waveforms
    else:
        sel = selection[cc.some_peaks_index]
        features = cc.some_features[sel]
        waveforms = cc.some_waveforms[sel]
    
    if method == 'kmeans':
        km = sklearn.cluster.KMeans(n_clusters=n_clusters,**kargs)
        labels = km.fit_predict(features)
    elif method == 'gmm':
        gmm = sklearn.mixture.GaussianMixture(n_components=n_clusters,**kargs)
        #~ labels =gmm.fit_predict(features)
        gmm.fit(features)
        labels =gmm.predict(features)
    elif method == 'agglomerative':
        agg = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, **kargs)
        labels = agg.fit_predict(features)
    elif method == 'dbscan':
        dbscan = sklearn.cluster.DBSCAN(**kargs)
        labels = dbscan.fit_predict(features)
    elif method == 'dirtycut':
        n_left = cc.info['params_waveformextractor']['n_left']
        n_right = cc.info['params_waveformextractor']['n_right']
        peak_sign = cc.info['params_peakdetector']['peak_sign']
        relative_threshold = cc.info['params_peakdetector']['relative_threshold']

        dirtycut = DirtyCut(waveforms, n_left, n_right, peak_sign, relative_threshold)
        labels = dirtycut.do_the_job()
    else:
        raise(ValueError, 'find_clusters method unknown')
    
    if selection is None:
        cc.all_peaks['label'][:] = labelcodes.LABEL_UNSLASSIFIED
        cc.all_peaks['label'][cc.some_peaks_index] = labels
    else:
        labels += max(cc.cluster_labels) + 1
        cc.all_peaks['label'][cc.some_peaks_index[sel]] = labels
    
    
    return labels






class DirtyCut:
    def __init__(self, waveforms, n_left, n_right, peak_sign, threshold):
        self.waveforms = waveforms
        self.n_left = n_left
        self.n_right = n_right
        self.width = n_right - n_left
        self.peak_sign = peak_sign
        self.threshold = threshold
        
        self.binsize = 0.1
        
        #~ self.smooth_kernel = scipy.signal.gaussian(51, 5)
        self.smooth_kernel = scipy.signal.gaussian(51, 15)
        self.smooth_kernel /= np.sum(self.smooth_kernel)
        
        self.nb_min = 10
        
        self.max_loop = 1000
        
        self.break_left_over = 30
        
        self.threshold_similarity = 0.9
        
        self.debug = True
        #~ self.debug = False

    
    def one_cut(self, x):
        labels = np.zeros(x.size, dtype='int64')
        
        # test with histogram
        #~ count, bins = np.histogram(x, bins=self.bins)
        #~ count_smooth = np.convolve(count, self.smooth_kernel, mode='same')
        #~ bins = bins[:1]
        
        # test with kernel density
        #~ kernel = stats.gaussian_kde(values)
        kern = scipy.stats.gaussian_kde(x)
        count = kern(self.bins)
        count_smooth = count
        bins = self.bins
        count_smooth[np.abs(bins)<self.threshold] = 0
        
        
        local_min_indexes, = np.nonzero((count_smooth[1:-1]<count_smooth[:-2])& (count_smooth[1:-1]<=count_smooth[2:]))
        
        #TODO work on this: accept cut unerthrehold ????
        keep = np.abs(bins[local_min_indexes])>(self.threshold + self.binsize)
        local_min_indexes = local_min_indexes[keep]
        
        nb_over_thresh = np.sum(np.abs(x)>self.threshold)
        
        if local_min_indexes.size==0:
            lim = 0
        else:
            #several cut possible
            n_on_left = []
            for ind in local_min_indexes:
                lim = bins[ind]
                n = np.sum(count[bins<=lim])
                n_on_left.append(n)

            n_on_left = np.array(n_on_left)
            #~ print('n_on_left', n_on_left, 'local_min_indexes', local_min_indexes,  x.size)
            #~ p = np.argmin(np.abs(n_on_left-x.size//2))
            p = np.argmin(np.abs(n_on_left-nb_over_thresh//2))
            
            #~ print('p', p)
            lim = bins[local_min_indexes[p]]

        
        if self.peak_sign == '-':
            labels[x>lim] = 1
        elif self.peak_sign == '+':
            labels[x<lim] = 1
        
        return labels, lim, bins, count_smooth
    
    def split_loop(self):
        cluster_labels = np.zeros(self.waveforms.shape[0], dtype='int64')

        if self.peak_sign == '-':
            m = np.min(self.waveforms[:, -self.n_left, :])
            self.bins=np.arange(m,0, self.binsize)
        elif self.peak_sign == '+':
            m = np.min(self.waveforms[:, -self.n_left, :])
            self.bins=np.arange(0, m, self.binsize)
        print('bins', self.bins[0], '...', self.bins[-1])
        
        
        flat_wf = self.waveforms.swapaxes(1,2).reshape(self.waveforms.shape[0], -1)
        
        k = 0
        dim_visited = []
        for i in range(self.max_loop):
                    
            left_over = np.sum(cluster_labels>=k)
            print()
            print('i', i, 'k', k, 'left_over', left_over)
            
            sel = cluster_labels == k
            
            if i!=0 and left_over<self.break_left_over:# or k==40:
                #~ cluster_labels[sel] = -k
                cluster_labels[sel] = -1
                print('BREAK left_over<', self.break_left_over)
                break
            
            wf_sel = flat_wf[sel]
            n_with_label = wf_sel .shape[0]
            print('n_with_label', n_with_label)
            
            if wf_sel .shape[0]<30:
                print('too few')
                cluster_labels[sel] = -1
                k+=1
                dim_visited = []
                continue
            
            med, mad = median_mad(wf_sel)
            #~ med = np.mean(wf_sel, axis=0)
            #~ mad = np.std(wf_sel, axis=0)
            
            
            if np.all(mad<1.6):
                print('mad<1.6')
                k+=1
                dim_visited = []
                continue

            if np.all(wf_sel.shape[0]<100):
                #TODO remove this!!!!!
                print('Too small cluster')
                k+=1
                dim_visited = []
                continue
            
            
            #~ weight = mad-1
            #~ feat = wf_sel * weight
            #~ feat = wf_sel[:, np.argmax(mad), None]
            #~ print(feat.shape)
            #~ print(feat)
            
            
            #~ while True:
            
            if self.peak_sign == '-':
                possible_dim, = np.nonzero(med<0)
            elif self.peak_sign == '+':
                possible_dim, = np.nonzero(med>0)
            
            possible_dim = possible_dim[~np.in1d(possible_dim, dim_visited)]
            if len(possible_dim)==0:
                print('BREAK len(possible_dim)==0')
                break
                
            #strategy1: take max mad
            #~ dim = possible_dim[np.argmax(mad[possible_dim])]
            
            #strategy2: take where bigest nb of devian
            #~ factor = 6
            factor = self.threshold
            if self.peak_sign == '-':
                nb_outer = np.sum(wf_sel[:, possible_dim]<(med[possible_dim]-factor*mad[possible_dim]), axis=0)
            elif self.peak_sign == '+':
                nb_outer = np.sum(wf_sel[:, possible_dim]>(med[possible_dim]+factor*mad[possible_dim]), axis=0)
            print('nb_outer', nb_outer.shape, nb_outer, wf_sel.shape, len(possible_dim))
            dim = possible_dim[np.argmax(nb_outer)]
            
            
            print('dim', dim)
            
            feat = wf_sel[:, dim]
            
            labels, lim, bins, count_smooth = self.one_cut(feat)

            if self.debug:
            #~ if False:
                import matplotlib.pyplot as plt
            #~ if False:
                if not os.path.exists('debug_dirtycut'):
                    os.mkdir('debug_dirtycut')
                
                if hasattr(self, 'n_cut'):
                    self.n_cut += 1
                else:
                    self.n_cut = 0
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(np.arange(self.smooth_kernel.size)*self.binsize, self.smooth_kernel)
                    #~ fig.savefig('debug_dirtycut/smooth_kernel.png')
                
                count, _ = np.histogram(feat, bins=self.bins)
                count = count.astype(float)/np.sum(count)
                
                filename = 'debug_dirtycut/one_cut {}.png'.format(self.n_cut)
                fig, axs = plt.subplots(nrows=3)

                axs[0].fill_between(np.arange(med.size), med-mad, med+mad, alpha=.5)
                axs[0].plot(np.arange(med.size), med)
                axs[0].set_title(str(wf_sel.shape[0]))
                axs[0].axvline(dim, color='r')
                axs[0].set_ylim(-10, 5)
                
                axs[1].plot(mad)
                axs[1].axvline(dim, color='r')
                axs[1].set_ylim(0,5)
                
                axs[2].plot(self.bins[:-1], count, color='b')
                axs[2].plot(bins, count_smooth, color='k')
                axs[2].axvline(lim, color='k')
                axs[2].set_xlim(-80,0.1)
                
                fig.savefig(filename)
                
                #~ plt.show()
            
            print(feat[labels==0].size, feat[labels==1].size)
            
            
            
            print('nb1', np.sum(labels==1), 'nb0', np.sum(labels==0))
            
            if np.sum(labels==0)==0:
                channel_index = dim // self.width
                #~ dim_visited.append(dim)
                dim_visited.extend(range(channel_index*width, (channel_index+1)*width))
                print('nb0==0', dim_visited)
                
                continue
            
            ind, = np.nonzero(sel)
            #~ cluster_labels[cluster_labels>k] += 1#TODO reflechir la dessus!!!
            cluster_labels[ind[labels==1]] += 1
            
            if np.sum(labels==1)==0:
                k+=1
                dim_visited = []
                    
        
        
        return cluster_labels
    
    
    def merge_loop(self, unmerged_labels):
        cluster_labels = unmerged_labels
        
        labels = np.unique(cluster_labels)
        labels = labels[labels>=0]
        
        #trash too small
        for k in labels:
            nb = np.sum(cluster_labels==k)
            if nb<self.nb_min:
                cluster_labels[cluster_labels==k] = -1
                print('trash', k)
        
        # relabel
        labels = np.unique(cluster_labels)
        labels = labels[labels>=0]
        for l, k in enumerate(labels):
            cluster_labels[cluster_labels==k] = l
        
        labels = np.unique(cluster_labels)
        labels = labels[labels>=0]
        
        #regroup if very high similarity
        flat_wf = self.waveforms.swapaxes(1,2).reshape(self.waveforms.shape[0], -1)
        centroids = []
        for k in labels:
            sel = cluster_labels==k
            med = np.median(flat_wf[sel], axis=0)
            centroids.append(med)
        centroids = np.array(centroids)
        similarity = sklearn.metrics.pairwise.cosine_similarity(centroids)
        similarity = np.triu(similarity)

        ind0, ind1 = np.nonzero(similarity>self.threshold_similarity)
        keep = ind0!=ind1
        ind0 = ind0[keep]
        ind1 = ind1[keep]
        pairs = list(zip(labels[ind0], labels[ind1]))
        #~ print(pairs)
        for k1, k2 in pairs:
            #~ print(k1, k2, similarity[k1, k2])
            cluster_labels[cluster_labels==k2] = k1
        
        #~ fig, ax = plt.subplots()
        #~ im  = ax.matshow(centroids, cmap='viridis', aspect='auto')
        #~ fig.colorbar(im)
        #~ plt.show()
        #~ fig, ax = plt.subplots()
        #~ im  = ax.matshow(similarity, cmap='viridis')
        #~ fig.colorbar(im)
        #~ plt.show()
        
        # relabel
        labels = np.unique(cluster_labels)
        labels = labels[labels>=0]
        for l, k in enumerate(labels):
            cluster_labels[cluster_labels==k] = l
        
        return cluster_labels
    
    def do_the_job(self):
        
        cluster_labels = self.split_loop()
        cluster_labels = self.merge_loop(cluster_labels)
        
        return cluster_labels

    