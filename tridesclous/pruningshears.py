"""
This is automatic home made method for clustering in TDC.

It is based on hdbscan algo.


"""


import numpy as np
import os
import time

import sklearn
import sklearn.cluster
import sklearn.mixture
import sklearn.metrics




import matplotlib.pyplot as plt
from .dip import diptest


import hdbscan

try:
    import isosplit5
    HAVE_ISOSPLIT5 = True
except:
    HAVE_ISOSPLIT5 = False


class PruningShears:
    def __init__(self, waveforms, 
                        features,
                        detection_channel_indexes,
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
                        
                        print_debug=False,
                        debug_plot=False
                        ):
        
        self.waveforms = waveforms
        self.features = features
        self.detection_channel_indexes = detection_channel_indexes
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
        #~ self.debug_plot = True
        self.debug_plot = debug_plot

    def log(self, *args, **kargs):
        if self.print_debug:
            print(*args, **kargs)
    
    def do_the_job(self):
        cluster_labels = self.explore_split_loop()
        #~ t0 = time.perf_counter()
        cluster_labels = self.try_oversplit(cluster_labels)
        #~ t1 = time.perf_counter()
        #~ print('try_oversplit', t1-t0)
        cluster_labels = self.merge_and_clean(cluster_labels)
        return cluster_labels
    
    def next_channel(self, mask_loop, chan_visited):
        self.log('next_channel percentiles', 'peak_max.size', np.sum(mask_loop))
        self.log('chan_visited', chan_visited)
        
        peak_max = self.all_peak_max[mask_loop, :]
        
        #~ self.detection_channel_indexes[mask_loop]
        
        
        nb_channel = self.waveforms.shape[2]
        percentiles = np.zeros(nb_channel)
        for c in range(nb_channel):
            if c in chan_visited:
                continue
            
            #~ x = peak_max[:, c]
            #~ x = x[x>self.threshold]
            
            #~ if x.size>self.min_cluster_size * 4:
                #~ per = np.nanpercentile(x, 99)
                #~ per = np.nanpercentile(x, 95)
                #~ per = np.nanpercentile(x, 90)
                #~ per = np.nanpercentile(x, 80)
            
            #~ elif x.size>self.min_cluster_size:
                #~ per = np.nanpercentile(x, 60)
            #~ else:
                #~ per = 0
            #~ percentiles[c] = per
            
            mask = self.detection_channel_indexes[mask_loop] == c
            #~ print(mask.size, peak_max.shape)
            x = peak_max[mask, :][:, c]
            if x.size >self.min_cluster_size:
                per = np.nanpercentile(x, 90)
            else:
                per = 0
            percentiles[c] = per
            
            
        
        #~ mask = percentiles > 0
        #~ print('mask.size', mask.size, 'mask.sum', mask.sum())
        
        order_visit = np.argsort(percentiles)[::-1]
        percentiles = percentiles[order_visit]
        
        mask = (percentiles > 0) & ~np.in1d(order_visit, chan_visited)
        
        order_visit = order_visit[mask]
        percentiles = percentiles[mask]
        
        #~ print('percentiles', percentiles)
        self.log('order_visit :10', order_visit[:10])
        self.log('percentiles :10', percentiles[:10])
        
        if len(order_visit)==0:
            self.log('len(order_visit)==0')
            return None
        
        actual_chan = order_visit[0]
        #~ if len(order_visit)>3:
            #~ # add some randomness
            #~ order_visit = np.random.permutation(order_visit[:order_visit.size//3])
            #~ actual_chan = order_visit[0]
        #~ else:
            #~ actual_chan = order_visit[0]
        
        return actual_chan
    
    def one_sub_cluster(self, local_data, allow_single_cluster):
        # clustering on one channel or one adjacency
        
        
        n_components = min(local_data.shape[1], 6)
        
        pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components, whiten=True)
        local_features = pca.fit_transform(local_data)
        
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, allow_single_cluster=allow_single_cluster, metric='l2')
        #~ clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, allow_single_cluster=True)
        #~ clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20, allow_single_cluster=True)
        
        #~ t0 = time.perf_counter()
        #~ local_labels = clusterer.fit_predict(local_data)
        #~ t1 = time.perf_counter()
        #~ print('fit_predict wf', t1-t0)

        
        #~ t0 = time.perf_counter()
        local_labels = clusterer.fit_predict(local_features)
        #~ t1 = time.perf_counter()
        #~ print('fit_predict pca', t1-t0)
        
        
        
        # try isosplit here not stable enought on windows
        #~ local_labels = isosplit5.isosplit5(local_data.T)
        
        return local_labels
    
    def check_candidate_labels(self, local_ind, local_labels, actual_chan):
        # keep only cluster when:
        #   * template peak is center on self.n_left
        #   * cluster size is bug enoutgh
        
        local_channels = self.channel_adjacency[actual_chan]
        
        #~ candidate_labels = []
        peak_is_on_chan = []
        peak_is_aligned = []
        best_chan_peak_values = []
        best_chan = []
        cluster_sizes = []
        #~ actual_chan_peak_values =  []
        
        
        local_centroids = {}
        
        unique_labels = np.unique(local_labels)
        unique_labels = unique_labels[unique_labels>=0]
        
        for l, label in enumerate(unique_labels):


            ind = local_ind[label == local_labels]
            cluster_size = ind.size
            cluster_sizes.append(cluster_size)
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
                # for dense we don't care the channel
                # and alignement
                chan_peak = chan_peak_local
                peak_is_on_chan.append(True)
                aligned = True
            else:
                chan_peak = local_channels[chan_peak_local]
                peak_is_on_chan.append(chan_peak == actual_chan)
                best_chan.append(chan_peak)
                aligned = np.abs(-self.n_left - pos_peak) <= 1
                
            self.log('-self.n_left', -self.n_left, 'pos_peak', pos_peak, 'aligned', aligned)
            peak_is_aligned.append(aligned)
            
            best_chan_peak_value = np.abs(centroid[pos_peak, chan_peak_local])
            best_chan_peak_values.append(best_chan_peak_value)
            
            self.log('label', label, 'chan_peak', chan_peak, 'chan_peak_local', chan_peak_local, 'aligned', aligned, 'best_chan_peak_value', best_chan_peak_value, 'cluster_size', cluster_size, 'peak_is_on_chan', peak_is_on_chan[-1])
            #~ if peak_is_aligned and cluster_size>self.min_cluster_size:
                #~ candidate_labels.append(label)
        
        cluster_sizes = np.array(cluster_sizes)
        peak_is_aligned = np.array(peak_is_aligned)
        peak_is_on_chan = np.array(peak_is_on_chan)
        best_chan_peak_values = np.array(best_chan_peak_values)
        best_chan = np.array(best_chan)
        
        if unique_labels.size>0:
            #~ print(cluster_sizes.shape, peak_is_aligned.shape, peak_is_on_chan.shape)
            candidate_mask= (cluster_sizes>=self.min_cluster_size) & peak_is_aligned & peak_is_on_chan
            #~ candidate_labels = unique_labels[candidate_mask]
            
            elsewhere_mask= (cluster_sizes>=self.min_cluster_size) & peak_is_aligned & ~peak_is_on_chan
            #~ candidate_labels_elsewhere = unique_labels[elsewhere_mask]
            
        else:
            candidate_mask = np.array([], dtype='bool')
            elsewhere_mask = np.array([], dtype='bool')
        
        
        return unique_labels, candidate_mask, elsewhere_mask, best_chan_peak_values,\
                        best_chan, peak_is_aligned, peak_is_on_chan, local_centroids

    
    def explore_split_loop(self):
        #~ print('all_peak_max')
        ind_peak = -self.n_left
        self.all_peak_max = self.waveforms[:, ind_peak, : ].copy()
        if self.peak_sign == '-' :
            self.all_peak_max *= -1
        
        #~ print('all_peak_max done')
        
        nb_channel = self.waveforms.shape[2]
        
        if self.dense_mode:
            # global pca so no sparse used
            pass
            
        else:
            # general case 
            n_components_by_channel = self.features.shape[1] // nb_channel
            
            self.weights_feat_chans = {}
            self.mask_feat_per_chan = {}
            self.mask_feat_per_adj = {}
            for chan in range(nb_channel):
                adjacency = self.channel_adjacency[chan]
                
                # mask feature per channel
                mask_feat = np.zeros(self.features.shape[1], dtype='bool')
                mask_feat[chan*n_components_by_channel:(chan+1)*n_components_by_channel] = True
                self.mask_feat_per_chan[chan] = mask_feat
                
                # mask_feat per channel adjacency
                mask_feat = np.zeros(self.features.shape[1], dtype='bool')
                for i in range(n_components_by_channel):
                    mask_feat[adjacency*n_components_by_channel+i] = True
                self.mask_feat_per_adj[chan] = mask_feat
                
                # weights
                weights = []
                for adj_chan in adjacency:
                    d = self.channel_distances[chan, adj_chan]
                    w = np.exp(-d/self.adjacency_radius_um * 2) # TODO fix this factor 2
                    #~ print('chan', chan, 'adj_chan', adj_chan, 'd', d, 'w', w)
                    weights += [ w ] * n_components_by_channel
                self.weights_feat_chans[chan] = np.array(weights).reshape(1, -1)
        
        force_next_chan = None
        
        cluster_labels = np.zeros(self.waveforms.shape[0], dtype='int64')
        last_nb_remain = cluster_labels.size
        
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
            
            if nb_working<self.min_cluster_size:
                self.log('TRASH: too few. Explore remain')
                cluster_labels[mask_loop] = -1
                k += 1
                chan_visited = []
                force_next_chan = None
                continue
            
            if self.dense_mode:
                actual_chan = 0
            else:
                
                if force_next_chan is None:
                    actual_chan = self.next_channel(mask_loop,  chan_visited)
                elif force_next_chan is not None and force_next_chan in chan_visited:
                    # this is impossible normally
                    self.log('!!!!! impossible force', force_next_chan, 'chan_visited', chan_visited)
                    raise # TODO
                    #~ actual_chan = self.next_channel(peak_max, chan_visited)
                else:
                    actual_chan = force_next_chan
                    self.log('force', actual_chan)

            
            if actual_chan is None:
                # no more chan to explore
                #~ cluster_labels[mask_loop] = -1
                #~ self.log('BREAK actual_chan None')
                #~ self.log('BREAK actual_chan None')
                #~ break
                
                if self.dense_mode:
                    pass
                    # impossible because actual_chan always 0
                else:
                    # start again explore all channels
                    if nb_remain < last_nb_remain:
                        cluster_labels[mask_loop] += 1
                        k += 1
                        chan_visited = []
                        force_next_chan = None
                        last_nb_remain = nb_remain
                        self.log('Explore again all channel once')
                        continue
                    else:
                        self.log('Stop exploring again all channel once')
                        break
                    
            
            self.log('actual_chan', actual_chan)
            adjacency = self.channel_adjacency[actual_chan]
            self.log('adjacency', adjacency)
            
            
            mask_thresh = np.zeros(mask_loop.size, dtype='bool')
            if self.dense_mode:
                mask_thresh[:] = True
            else:
                peak_max = self.all_peak_max[mask_loop, :]
                mask_thresh[mask_loop] = peak_max[:, actual_chan] > self.threshold
                
            self.log('mask_loop.size', mask_loop.size, 'mask_loop.sum', mask_loop.sum(), 'mask_thresh.sum', mask_thresh.sum())
            
            ind_l0,  = np.nonzero(mask_loop & mask_thresh)
            self.log('ind_l0.size', ind_l0.size)
            
            if ind_l0.size < self.min_cluster_size:
                force_next_chan = None
                chan_visited.append(actual_chan)
                self.log('len(ind_l0.size) < self.min_cluster_size')
                final_label = None
                self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                continue
            
            
            # Step2: 
            if self.dense_mode:
                wf_features = self.features.take(ind_l0, axis=0)
            else:
                mask_feat = self.mask_feat_per_adj[actual_chan]
                wf_features = self.features.take(ind_l0, axis=0).compress(mask_feat, axis=1)
            
            self.log('wf_features.shape', wf_features.shape)
            
            features_l0 = wf_features
            self.log('features_l0.shape', features_l0.shape)
            
            labels_l0 = self.one_sub_cluster(features_l0, allow_single_cluster=False)
            
            possible_labels_l0 = np.unique(labels_l0)
            possible_labels_l0 = possible_labels_l0[possible_labels_l0>=0]
            
            if len(possible_labels_l0) == 0:
                # add noise to force one cluster to be discovered
                if self.dense_mode:
                    noise_features = self.noise_features
                else:
                    noise_features = self.noise_features.compress(mask_feat, axis=1)
                self.log('Add noise to force one cluster')
                features_with_noise = np.concatenate([noise_features, wf_features, ], axis=0)
                labels_l0 = self.one_sub_cluster(features_with_noise, allow_single_cluster=False)
                labels_l0 = labels_l0[self.noise_features.shape[0]:]
                possible_labels_l0 = np.unique(labels_l0)
                possible_labels_l0 = possible_labels_l0[possible_labels_l0>=0]
                
                # instead of experimental save trash
                #~ if len(possible_labels_l0) == 0:
                    #~ self.log('Add noise do not work, explore other channel')
                    #~ force_next_chan = None
                    #~ chan_visited.append(actual_chan)
                    #~ final_label = None
                    #~ self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                    #~ continue
            
            if len(possible_labels_l0) == 0:
                # experimental!!!!!!

                pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=True)
                reduced_features = pca.fit_transform(wf_features)
                
                pval = diptest(np.sort(reduced_features[:, 0]), numt=200)
                
                if pval is not None and pval>0.2:
                    self.log('len(possible_labels_l0) == 0 BUT diptest pval', pval)

                    if self.dense_mode:
                        wf_l0 = self.waveforms.take(ind_l0, axis=0)
                    else:
                        mask_feat = self.mask_feat_per_adj[actual_chan]
                        wf_l0 = self.waveforms.take(ind_l0, axis=0).take(adjacency, axis=2)
                    centroid = np.median(wf_l0, axis=0)
                    centroid = centroid[np.newaxis, :, :]
                    out_up = np.any(np.any(wf_l0 > centroid + 4, axis=2), axis=1)
                    out_dw = np.any(np.any(wf_l0 < centroid - 4, axis=2), axis=1)
                    ok = ~out_up & ~out_dw
                    if np.sum(ok) < self.min_cluster_size:
                        self.log('!!!!!!! Not save trash!!!!!! np.sum(ok)', np.sum(ok))
                        force_next_chan = None
                        chan_visited.append(actual_chan)
                        final_label = None
                        self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                        continue
                    labels_l0[ok] = 0
                    labels_l0[~ok] = -1
                    possible_labels_l0 = np.array([0], dtype='int64')
                else:
                    self.log('len(possible_labels_l0) == 0 AND diptest pval', pval, 'no cluster at all')
                    if self.dense_mode:
                        self.log('no more with dense mode')
                        break
                    else:
                        force_next_chan = None
                        chan_visited.append(actual_chan)
                        #TODO dip test ????
                        self.log('len(possible_labels_l0) == 0')
                        final_label = None
                        self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                        continue
            
            
            self.log('possible_labels_l0', possible_labels_l0)
            
            self.log('labels_l0 == -1 size', labels_l0[labels_l0==-1].size)
            

            # explore candidate
            possible_labels_l0, candidate_mask_l0, elsewhere_mask_l0, best_chan_peak_values, best_chan,\
                    peak_is_aligned, peak_is_on_chan, local_centroids = self.check_candidate_labels(ind_l0, labels_l0, actual_chan)
            
            
            self.log('possible_labels_l0', possible_labels_l0)
            candidate_labels_l0 = possible_labels_l0[candidate_mask_l0]
            self.log('candidate_labels_l0', candidate_labels_l0)
            candidate_labels_elsewhere_l0 = possible_labels_l0[elsewhere_mask_l0]
            self.log('candidate_labels_elsewhere_l0', candidate_labels_elsewhere_l0)
            
            
            if len(possible_labels_l0) == 0:
                self.log('len(possible_labels_l0) == 0 impossible')
                raise()
            
            if np.sum(elsewhere_mask_l0):
                ind_best_elsewhere = np.argmax(best_chan_peak_values[elsewhere_mask_l0])
                elsewhere_peak_val = best_chan_peak_values[elsewhere_mask_l0][ind_best_elsewhere]
            else:
                elsewhere_peak_val = None
            
            if np.sum(candidate_mask_l0):
                ind_best = np.argmax(best_chan_peak_values[candidate_mask_l0])
                peak_val = best_chan_peak_values[candidate_mask_l0][ind_best]
            else:
                peak_val = None
            
            self.log('peak_val', peak_val, 'elsewhere_peak_val', elsewhere_peak_val)
            
            if elsewhere_peak_val is not None:
                # TODO put in params
                ratio_peak_force_explore = 1.5
                if (peak_val is not None and elsewhere_peak_val > peak_val*ratio_peak_force_explore) or peak_val is None:
                    # there is a better option elsewhere
                    force_next_chan = best_chan[elsewhere_mask_l0][ind_best_elsewhere]
                    if force_next_chan in chan_visited:
                        if peak_val is None:
                            self.log('Impossible to force_next_chan!!!!!!!!!!!!!!!', force_next_chan, 'actual_chan', actual_chan)
                            
                            force_next_chan = None
                            chan_visited.append(actual_chan)
                            
                            
                            final_label = None
                            self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                            continue
                        else:
                            force_next_chan = None
                            # test peak_val
                    else:
                        chan_visited.append(actual_chan)
                        self.log('force_next_chan', force_next_chan, 'actual_chan', actual_chan)
                        
                        final_label = None
                        self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                        continue
            
            if peak_val is None:
                # no candidate and no elsewhere
                force_next_chan = None
                chan_visited.append(actual_chan)
                final_label = None
                self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                continue                
            
            
            final_label = possible_labels_l0[candidate_mask_l0][ind_best]
            
            self.log('final_label', final_label, 'peak_val', peak_val)
            
            # remove trash : TODO
            #~ if not self.dense_mode:
            ind_trash_label = ind_l0[labels_l0 == -1]
            self.log('ind_trash_label.shape', ind_trash_label.shape)
            cluster_labels[ind_trash_label] = -1


            ind_new_label = ind_l0[final_label == labels_l0]
            
            self.log('ind_new_label.shape', ind_new_label.shape)

            cluster_labels[cluster_labels>=k] += 1
            cluster_labels[ind_new_label] -= 1
            
            k += 1
            #~ chan_visited = []
            force_next_chan = None
            
            self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
            
            continue



            self._plot_debug(actual_chan, ind_l0, channel_features_l0, labels_l0, possible_labels_l0, ind_l2, labels_l2, possible_labels_l2, final_label_l2)
            
        self.log('END loop', np.sum(cluster_labels==-1))
        return cluster_labels

    def _plot_debug(self, actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label):
        if not self.debug_plot:
            return
        
        from .matplotlibplot import plot_waveforms_density
        fig, axs = plt.subplots(ncols=2, nrows=2)
        
        colors = plt.cm.get_cmap('jet', len(possible_labels_l0))
        colors = {possible_labels_l0[l]:colors(l) for l in range(len(possible_labels_l0))}
        colors[-1] = 'k'
        
        ax = axs[0, 0]
        
        ax.set_title('actual_chan '+str(actual_chan))
        
        if self.dense_mode:
            mask_feat = slice(None)
        else:
            mask_feat = self.mask_feat_per_adj[actual_chan]
        local_wf_features = features_l0
        
        
        local_wf_features = features_l0
        
        if possible_labels_l0 is not None:
            for l, label in enumerate([-1] + possible_labels_l0.tolist()):
                #~ if label<0: 
                    #~ continue
                sel = label == labels_l0
                if np.sum(sel) ==0:
                    continue
                m = local_wf_features[sel].mean(axis=0)

                ind, = np.nonzero(sel)

                if ind.size> 200:
                    ind = ind[:200]

                color=colors[label]
                
                ax.plot(local_wf_features[ind].T, color=color, alpha=0.05)
                
                if label>=0:
                    ax.plot(m, color=color, alpha=1)
        
        
        ax = axs[1, 0]
        adjacency = self.channel_adjacency[actual_chan]
        
        wf_adj = self.waveforms[:,:, adjacency][ind_l0, :, :]
        if possible_labels_l0 is not None:
            for l, label in enumerate([-1] + possible_labels_l0.tolist()):
                sel = label == labels_l0
                if np.sum(sel) ==0:
                    continue
                m = wf_adj[sel].mean(axis=0)
                ind, = np.nonzero(sel)

                if ind.size> 200:
                    ind = ind[:200]
                
                color=colors[label]
                
                ax.plot(wf_adj[ind].swapaxes(1,2).reshape(ind.size, -1).T, color=color, alpha=0.05)
                
                if label>=0:
                    ax.plot(m.T.flatten(), color=color, alpha=1, lw=2)
            
            
            for c, chan in enumerate(adjacency):
            #~ ind = adjacency.tolist().index(actual_chan)
                if self.dense_mode:
                    lw = 1
                else:
                    if chan == actual_chan:
                        lw = 2
                    else:
                        lw = 1
                ax.axvline(self.width * c - self.n_left, color='m', lw=lw)
        
        
        ax = axs[0, 1]
        if final_label is not None:
            
            adjacency = self.channel_adjacency[actual_chan]
            sel = final_label == labels_l0
            
            ax.set_title('keep size {}'.format(np.sum(sel)))
            
            wf_adj = self.waveforms[:,:, adjacency][ind_l0, :, :][sel]
            m = wf_adj.mean(axis=0)

            if wf_adj.shape[0] > 400:
                wf_adj = wf_adj[:400, :, :]
            
            color=colors[final_label]
            ax.plot(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).T, color=color, alpha=0.1)
            ax.plot(m.T.flatten(), color=color, alpha=1, lw=2)

        ax = axs[1, 1]
        if final_label is not None:
            sel = -1 == labels_l0
            
            ax.set_title('trash size {}'.format(np.sum(sel)))
            
            if np.sum(sel) > 0:
                wf_adj = self.waveforms[:,:, adjacency][ind_l0, :, :][sel]
                if wf_adj.shape[0] > 400:
                    wf_adj = wf_adj[:400, :, :]
                ax.plot(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).T, color='k', alpha=0.1)

            for c, chan in enumerate(adjacency):
            #~ ind = adjacency.tolist().index(actual_chan)
                if self.dense_mode:
                    lw = 1
                else:
                    if chan == actual_chan:
                        lw = 2
                    else:
                        lw = 1
                ax.axvline(self.width * c - self.n_left, color='m', lw=lw)

        

        plt.show()
        
        #~ self.debug_plot = False
        
        return
    
    
    def try_oversplit(self, cluster_labels):
        cluster_labels = cluster_labels.copy()
        
        labels = np.unique(cluster_labels)
        labels = labels[labels>=0]
        
        m = np.max(labels) + 1
        
        #~ max_per_cluster= 300 
        for label in np.unique(labels):
            #~ print()
            #~ print('label', label)

            ind_keep,  = np.nonzero(cluster_labels == label)
            if self.dense_mode:
                waveforms = self.waveforms.take(ind_keep, axis=0)
                extremum_channel = 0
            else:
                centroid = np.median(self.waveforms[ind_keep, :, :], axis=0)
                if self.peak_sign == '-':
                    extremum_channel = np.argmin(centroid[-self.n_left,:], axis=0)
                elif self.peak_sign == '+':
                    extremum_channel = np.argmax(centroid[-self.n_left,:], axis=0)
                # TODO by sparsity level threhold and not radius
                adjacency = self.channel_adjacency[extremum_channel]
                waveforms = self.waveforms.take(ind_keep, axis=0).take(adjacency, axis=2)
            wf_flat = waveforms.swapaxes(1,2).reshape(waveforms.shape[0], -1)


            

            #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=6, whiten=True)
            pca =  sklearn.decomposition.IncrementalPCA(n_components=6, whiten=True)
            
            feats = pca.fit_transform(wf_flat)
            pval = diptest(np.sort(feats[:, 0]), numt=200)
            #~ print('pval', pval)
            self.log('label', label,'pval', pval)
            if pval<0.2:
            
                clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, allow_single_cluster=False, metric='l2')
                sub_labels = clusterer.fit_predict(feats[:, :2])
                unique_sub_labels = np.unique(sub_labels)
                if unique_sub_labels.size ==  1 and unique_sub_labels[0] == -1:
                    pass
                else:
                    possible_labels, candidate_mask, elsewhere_mask, best_chan_peak_values, best_chan,\
                        peak_is_aligned, peak_is_on_chan, local_centroids = self.check_candidate_labels(ind_keep, sub_labels, extremum_channel)
                    
                    self.log('possible_labels',possible_labels)
                    
                    
                    
                    for sub_label in unique_sub_labels:
                        sub_mask = sub_labels == sub_label
                        
                        valid = sub_label in possible_labels[peak_is_aligned]
                        
                        #~ print('sub_label', 'valid', valid)
                        
                        
                        
                        if sub_label == -1:
                            cluster_labels[ind_keep[sub_mask]] = -1
                        else:
                            # TODO check if peak center and size OK
                            cluster_labels[ind_keep[sub_mask]] = sub_label + m 
                    
                    if np.max(unique_sub_labels) >=0:
                        m += np.max(unique_sub_labels) + 1
                
                if self.debug_plot:
                    fig, axs = plt.subplots(ncols=3)
                    colors = plt.cm.get_cmap('jet', len(unique_sub_labels))
                    colors = {unique_sub_labels[l]:colors(l) for l in range(len(unique_sub_labels))}
                    colors[-1] = 'k'
                    
                    ax = axs[0]
                    ax.plot(wf_flat.T, color='k', alpha=0.1)
                    for sub_label in unique_sub_labels:
                        valid = sub_label in possible_labels[peak_is_aligned]
                        
                        sub_mask = sub_labels == sub_label
                        color = colors[sub_label]
                        ax.plot(wf_flat.T, color=color, alpha=0.1)
                        if valid:
                            ls = '-'
                        else:
                            ls = '--'
                        if sub_label>=0:
                            ax.plot(np.median(wf_flat[sub_mask], axis=0), color=color, lw=2, ls=ls)
                    

                    ax = axs[1]
                    ax.plot(feats.T, color='k', alpha=0.1)
                    
                    ax = axs[2]
                    ax.scatter(feats[:, 0], feats[:, 1], color='k')
                
                    plt.show()
            
        return cluster_labels



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
                        
            for ind, k in enumerate(labels):
                ind_keep,  = np.nonzero(cluster_labels2 == k)
                if ind_keep.size > self.max_per_cluster_for_median:
                    sub_sel = np.random.choice(ind_keep.size, self.max_per_cluster_for_median, replace=False)
                    ind_keep = ind_keep[sub_sel]
                centroids[ind,:,:] = np.median(self.waveforms[ind_keep, :, :], axis=0)
            
            #eliminate when best peak not aligned
            # eliminate when peak value is too small
            # TODO move this in the main loop!!!!!
            for ind, k in enumerate(labels):
                centroid = centroids[ind,:,:]
                if self.peak_sign == '-':
                    chan_peak = np.argmin(np.min(centroid, axis=0))
                    pos_peak = np.argmin(centroid[:, chan_peak])
                    #~ print(centroid.shape, -self.n_left, chan_peak)
                    peak_val = centroid[-self.n_left, chan_peak]
                elif self.peak_sign == '+':
                    chan_peak = np.argmax(np.max(centroid, axis=0))
                    pos_peak = np.argmax(centroid[:, chan_peak])
                    peak_val = centroid[-self.n_left, chan_peak]
                
                if np.abs(-self.n_left - pos_peak)>2:
                    self.log('remove not aligned peak', 'k', k)
                    #delete
                    cluster_labels2[cluster_labels2 == k] = -1
                    labels[ind] = -1
                if np.abs(peak_val) < self.threshold + 0.5:
                    self.log('remove small peak', 'k', k, 'peak_val', peak_val)
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


# TODO new
# si len(possible_labels_l0) == 0 alors on garde quand meme DONE
# si pas aligne plus que n_shift_merged et un seul cluster alors trash
# elsewhere candiate avec size




"""
# to compare with previous release
class PruningShears_1_4_1:
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
        
        self.debug_plot = False
        #~ self.debug_plot = True

    def log(self, *args, **kargs):
        if self.print_debug:
            print(*args, **kargs)
    
    def do_the_job(self):
        cluster_labels = self.explore_split_loop()
        cluster_labels = self.merge_and_clean(cluster_labels)
        return cluster_labels
    
    def next_channel(self, peak_max, chan_visited):
        self.log('next_channel percentiles')
        nb_channel = self.waveforms.shape[2]
        percentiles = np.zeros(nb_channel)
        for c in range(nb_channel):
            x = peak_max[:, c]
            x = x[x>self.threshold]
            
            if x.size>self.min_cluster_size:
                per = np.nanpercentile(x, 99)
            else:
                per = 0
            percentiles[c] = per
        order_visit = np.argsort(percentiles)[::-1]
        #~ print('percentiles', percentiles)
        order_visit = order_visit[percentiles[order_visit]>0]
        #~ print('order_visit', order_visit)
        
        
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
        # TODO try isosplit here
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        local_labels = clusterer.fit_predict(local_data)
        
        #~ clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, 
                    #~ cluster_selection_epsilon=0.1, cluster_selection_method = 'eom')
        #~ local_labels = clusterer.fit_predict(local_data)
        
        #~ local_labels = isosplit5.isosplit5(local_data.T)
        
        
        # unidip try
        #~ import unidip
        #~ data = np.sort(local_data[:, 0])
        #~ unidip.dip.diptst(data)
        #~ fig, ax = plt.subplots()
        #~ ax.hist(data, bins=500)
        #~ clusterer = unidip.UniDip(data)
        #~ intervals = clusterer.run()
        #~ print('intervals', intervals)
        #~ local_labels = -np.ones(local_data.shape[0], dtype='int64')
        #~ for label, (lim0, lim1) in enumerate(intervals):
            #~ mask = (local_data[:, 0] >=data[lim0]) & (local_data[:, 0] <=data[lim1])
            #~ local_labels[mask] = label
            
            #~ ax.axvline(data[lim0])
            #~ ax.axvline(data[lim1])
        #~ print(local_labels)
        #~ plt.show()
        
        return local_labels
    
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
            mask_feat_chans = {}
            for chan in range(nb_channel):
                adjacency = self.channel_adjacency[chan]
                
                # mask_feat
                mask_feat = np.zeros(self.features.shape[1], dtype='bool')
                for i in range(n_components_by_channel):
                    mask_feat[adjacency*n_components_by_channel+i] = True
                mask_feat_chans[chan] = mask_feat
                
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
            else:
                self.log('force', actual_chan)
                actual_chan = force_next_chan
                
            
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
                mask_feat = mask_feat_chans[actual_chan]
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
                self.log('label', label, 'chan peak values', chan_peak_local, 'peak_is_aligned', peak_is_aligned, 'peak values', np.abs(centroid_adj[-self.n_left, chan_peak_local]), 'cluster_size', cluster_size)
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
            
            if self.debug_plot:
            #~ if True:
            #~ if True and k>9:
            #~ if True and k==3:
            #~ if False:
                
                reduced_noise_features = reduced_features[:self.noise_features.shape[0]]
                reduced_features = reduced_features[self.noise_features.shape[0]:]
                
                from .matplotlibplot import plot_waveforms_density
                
                colors = plt.cm.get_cmap('jet', len(np.unique(local_labels)))
                
                fig, axs = plt.subplots(ncols=3)
                
                ax = axs[0]
                if self.dense_mode:
                    feat0, feat1 = local_features[:, 0], local_features[:, 1]
                else:
                    chan_features = self.features[:, actual_chan*n_components_by_channel:(actual_chan+1)*n_components_by_channel]
                    feat0, feat1 = chan_features[ind_keep, :][:, 0], chan_features[ind_keep, :][:, 1]
                for l, label in enumerate(np.unique(local_labels)):
                    #~ if label not in candidate_labels:
                        #~ continue
                    sel = label == local_labels
                    if label<0: 
                        color = 'k'
                    else:
                        color=colors(l)
                    if label in candidate_labels:
                        s = 3
                    else:
                        s = 1
                        
                    ax.scatter(feat0[sel], feat1[sel], s=s, color=color)
                    
                
                ax.scatter(local_noise_features[:, 0], local_noise_features[:, 1], s=1, color='r')
                
                
                ax = axs[1]
                ax.scatter(reduced_features[:, 0], reduced_features[:, 1], s=1, color='black')
                ax.scatter(reduced_noise_features[:, 0], reduced_noise_features[:, 1], s=1, color='r')
                
                
                
                #~ feat0, feat1 = chan_features[keep_wf, :][:, 0], chan_features[keep, :][:, 1]
                #~ ax.scatter(reduced_features[:, 0], reduced_features[:, 1], s=1)
                for l, label in enumerate(np.unique(local_labels)):

                    #~ if label not in candidate_labels:
                        #~ continue
                    
                    sel = label == local_labels
                    if label<0: 
                        color = 'k'
                    else:
                        color=colors(l)
                    
                    if label in candidate_labels:
                        s = 3
                    else:
                        s = 1
                    
                    ax.scatter(reduced_features[sel, :][:, 0], reduced_features[sel, :][:, 1], s=s, color=color)
                

                ax = axs[2]
                wf_chan = self.waveforms[:,:, [actual_chan]][mask_loop & mask_thresh, :, :]
                bin_min, bin_max, bin_size = -40, 20, 0.2
                #~ bin_min, bin_max, bin_size = -340, 180, 1
                im = plot_waveforms_density(wf_chan, bin_min, bin_max, bin_size, ax=ax)
                #~ im.set_clim(0, 50)
                #~ im.set_clim(0, 25)
                #~ im.set_clim(0, 150)
                #~ im.set_clim(0, 250)
                fig.colorbar(im)
                
                for l, label in enumerate(np.unique(local_labels)):
                
                    #~ if label not in candidate_labels:
                        #~ continue
                    #~ if label<0: 
                        #~ continue
                    sel = label == local_labels
                    ind = ind_keep[sel]
                    wf = self.waveforms[:,:, actual_chan][ind, :].mean(axis=0)
                    
                    if label<0: 
                        color = 'k'
                    else:
                        color=colors(l)
                    
                    if label in candidate_labels:
                        ls = '-'
                    else:
                        ls = '--'
                    
                    ax.plot(wf, color=color, ls=ls)
                
                
                ax.axvline(-self.n_left, color='w', lw=2)
            
            
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
            if len(candidate_labels) == 0:
                chan_visited.append(actual_chan)
                #~ self.log('EXPLORE NEW DIM lim0 is None ',  len(chan_visited))
                self.log('EXPLORE NEW DIM no label ')
                if self.debug_plot:
                    plt.show()
                
                force_next_chan = None
                continue
            
            ind_best = np.argmax(candidate_peak_values)
            if candidate_chan_peak[ind_best] != actual_chan:
                self.log('Force channel exploration', candidate_chan_peak[ind_best] , actual_chan )
                if self.debug_plot:
                    plt.show()
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
            #~ if True:
            #~ if True and k==3:
            #~ if False:
                
                from .matplotlibplot import plot_waveforms_density
                
                fig, axs = plt.subplots(ncols=2)

                wf_adj = self.waveforms[:,:, adjacency][ind_new_label, :, :]
                m = np.median(wf_adj, axis=0)

                #~ print(wf_adj.shape)
                #~ print(m.shape)
                #~ print(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).shape)
                
                ax = axs[0]
                if wf_adj.shape[0] > 150:
                    wf_adj = wf_adj[:150, :, :]
                ax.plot(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).T, color='k', alpha=0.1)
                ax.plot(m.T.flatten(), color='m')
                
                ax = axs[1]
                bin_min, bin_max, bin_size = -40, 20, 0.2
                #~ bin_min, bin_max, bin_size = -340, 180, 1
                im = plot_waveforms_density(wf_adj, bin_min, bin_max, bin_size, ax=ax)
                #~ im.set_clim(0, 150)
                #~ im.set_clim(0, 250)
                fig.colorbar(im)
                ax.plot(m.T.flatten(), color='m')
                
                txt = 'k={} chan={} adj={}'.format(k, actual_chan, adjacency)
                fig.suptitle(txt)
                

                
                plt.show()
                
            
            
            #~ self.log('ind_loop.size', ind_loop.size)
            
            #~ not_in = np.ones(ind_keep.size, dtype='bool')
            #~ not_in[ind_new_label] = False
            #~ cluster_labels[cluster_labels>k] += 1#TODO reflechir la dessus!!!
            #~ cluster_labels[ind_keep[not_in]] += 1
            cluster_labels[cluster_labels>=k] += 1
            cluster_labels[ind_new_label] -= 1
            
            k += 1
            chan_visited = []
            force_next_chan = None
            

            #~ if np.sum(not_in)==0:
            #~ if ind_new_label.size == ind_keep.size:
                #~ self.log('ACCEPT: not_in.sum()==0')
                #~ k+=1
                #~ chan_visited = []
                #~ continue
        
        self.log('END loop', np.sum(cluster_labels==-1))
        return cluster_labels

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
"""