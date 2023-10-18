"""
This is an old home made method for clustering in TDC.
This is kept for comparison and compatibility.
prunningshears should give better results.

"""

import numpy as np
import os
import time


import scipy.signal
import scipy.stats
from sklearn.neighbors import KernelDensity






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
        
        #~ import matplotlib.pyplot as plt
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
            
            
            order_visit = order_visit[~np.isin(order_visit, chan_visited)]
            
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
                import matplotlib.pyplot as plt
                if not os.path.exists('debug_sawchaincut'):
                    os.mkdir('debug_sawchaincut')
                
                if hasattr(self, 'n_cut'):
                    self.n_cut += 1
                else:
                    self.n_cut = 0
                    
                    #~ import matplotlib.pyplot as plt
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
            
            
            n_spike_for_centroid = 300
            for ind, k in enumerate(labels):
                ind_keep,  = np.nonzero(cluster_labels2 == k)
                if ind_keep.size > n_spike_for_centroid:
                    sub_sel = np.random.choice(ind_keep.size, n_spike_for_centroid, replace=False)
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