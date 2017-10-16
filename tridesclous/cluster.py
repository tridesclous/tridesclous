import numpy as np
import os

import sklearn
import sklearn.cluster
import sklearn.mixture

import scipy.signal
import scipy.stats

from . import labelcodes
from .tools import median_mad

import matplotlib.pyplot as plt


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
        labels = dirtycut.split_loop()
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
        self.peak_sign = peak_sign
        self.threshold = threshold
        
        self.binsize = 0.1
        
        #~ self.smooth_kernel = scipy.signal.gaussian(51, 5)
        self.smooth_kernel = scipy.signal.gaussian(51, 15)
        self.smooth_kernel /= np.sum(self.smooth_kernel)
        
        self.max_loop = 1000
        
        self.break_left_over = 30
        
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
        
        local_min_indexes, = np.nonzero((count_smooth[1:-1]<count_smooth[:-2])& (count_smooth[1:-1]<=count_smooth[2:]))
        
        #TODO work on this: accept cut unerthrehold ????
        #~ keep = np.abs(bins[local_min_indexes])>self.threshold
        #~ local_min_indexes = local_min_indexes[keep]
        
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
            p = np.argmin(np.abs(n_on_left-x.size//2))
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
                cluster_labels[sel] = -k
                print('BREAK left_over<', self.break_left_over)
                break
            
            wf_sel = flat_wf[sel]
            n_with_label = wf_sel .shape[0]
            print('n_with_label', n_with_label)
            
            if wf_sel .shape[0]<30:
                print('too few')
                cluster_labels[sel] = -k
                k+=1
                dim_visited = []
                continue
            
            med, mad = median_mad(wf_sel)
            
            if np.all(mad<1.6):
                print('mad<1.6')
                k+=1
                dim_visited = []
                continue

            if np.all(wf_sel.shape[0]<100):
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
                #~ dim = None
                break
            dim = possible_dim[np.argmax(mad[possible_dim])]
            print('dim', dim)
            
            #~ fig, axs = plt.subplots(nrows=2, sharex=True)
            #~ axs[0].fill_between(np.arange(med.size), med-mad, med+mad, alpha=.5)
            #~ axs[0].plot(np.arange(med.size), med)
            #~ axs[0].set_title(str(wf_sel.shape[0]))
            #~ axs[1].plot(mad)
            #~ axs[1].axvline(dim, color='r')
            #~ plt.show()


            
            feat = wf_sel[:, dim]
            
            labels, lim, bins, count_smooth = self.one_cut(feat)

            if self.debug:
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
                fig, ax = plt.subplots()
                ax.plot(self.bins[:-1], count, color='b')
                ax.plot(bins, count_smooth, color='k')
                ax.axvline(lim, color='k')
                ax.set_xlim(-40,0)
                fig.savefig(filename)
                #~ plt.show()            
            
            
            #~ if labels is None:
                #~ channel_index = dim // width
                #~ dim_visited.append(dim)
                #~ dim_visited.extend(range(channel_index*width, (channel_index+1)*width))
                #~ print('loop', dim_visited)
                #~ continue
            
            
                
            
            print(feat[labels==0].size, feat[labels==1].size)
            
            
            
            print('nb1', np.sum(labels==1), 'nb0', np.sum(labels==0))
            
            if np.sum(labels==0)==0:
                channel_index = dim // width
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
        
    