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
        
    def make_local_features(self, ind, actual_chan):
        
        mask_feat = self.mask_feat_per_adj[actual_chan]
        
        local_wf_features = self.features.take(ind, axis=0).compress(mask_feat, axis=1)
        #~ local_noise_features = self.noise_features.compress(mask_feat, axis=1)
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
        
        return reduced_features
        
    
    def check_candidate_labels(self, local_ind, local_labels, actual_chan):
        # keep only cluster when:
        #   * template peak is center on self.n_left
        #   * cluster size is bug enoutgh
        
        local_channels = self.channel_adjacency[actual_chan]
        
        #~ candidate_labels = []
        peak_is_on_chan = []
        peak_is_aligned = []
        best_chan_peak_values = []
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
                chan_peak = chan_peak_local
                peak_is_on_chan.append(True)
            else:
                chan_peak = local_channels[chan_peak_local]
                peak_is_on_chan.append(chan_peak == actual_chan)
            
            
            
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
        
        if unique_labels.size>0:
            #~ print(cluster_sizes.shape, peak_is_aligned.shape, peak_is_on_chan.shape)
            selected_mask= (cluster_sizes>=self.min_cluster_size) & peak_is_aligned & peak_is_on_chan
            candidate_labels = unique_labels[selected_mask]
        else:
            candidate_labels = np.array([], dtype='int64')
        
        return unique_labels, candidate_labels, best_chan_peak_values, peak_is_aligned, peak_is_on_chan, local_centroids
    
    
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
                self.log('!!!!! impossible force', force_next_chan, 'chan_visited', chan_visited)
                #~ raise()
                actual_chan = self.next_channel(peak_max, chan_visited)
            else:
                actual_chan = force_next_chan
                self.log('force', actual_chan)
                
            
            if actual_chan is None:
                if np.sum(cluster_labels>k)>0:
                    cluster_labels[mask_loop] = -1
                    self.log('!!!!!!! BREAK actual_chan None but still some spikes')
                    break
                    
                    #~ k+=1
                    #~ chan_visited = []
                    #~ force_next_chan = None
                    #~ continue
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
            
            self.log('mask_thresh.size', mask_thresh.size, 'keep.sum', mask_thresh.sum())
            
            ind_l0,  = np.nonzero(mask_loop & mask_thresh)
            
            
            
            # TODO avoid infinite loop
            #~ if 
            
            

            #~ mask_feat = np.zeros(self.features.shape[1], dtype='bool')
            #~ for i in range(n_components_by_channel):
                #~ mask_feat[adjacency*n_components_by_channel+i] = True
            
            
            # Step2: cluster on the channel only
            if self.dense_mode:
                channel_wf_features = self.features.take(ind_l0, axis=0)
                channel_noise_features = self.noise_features
            else:
                mask_feat = self.mask_feat_per_chan[actual_chan]
                channel_wf_features = self.features.take(ind_l0, axis=0).compress(mask_feat, axis=1)
                channel_noise_features = self.noise_features.compress(mask_feat, axis=1)
            
            self.log('channel_wf_features.shape', channel_wf_features.shape)
            
            channel_features_l0 = np.concatenate([channel_noise_features, channel_wf_features, ], axis=0)
            self.log('channel_features_l0.shape', channel_features_l0.shape)
            
            labels_l0 = self.one_sub_cluster(channel_features_l0)
            # remove noise label
            labels_l0 = labels_l0[self.noise_features.shape[0]:]
            possible_labels_l0 = np.unique(labels_l0)
            possible_labels_l0 = possible_labels_l0[possible_labels_l0>=0]
            
            self.log('possible_labels_l0', possible_labels_l0)
            
            if len(labels_l0) ==0 or len(labels_l0) ==1:
                labels_l0[:] = 0 
                possible_labels_l0 = np.unique(labels_l0)
                sleG.log('ATTTENTION pas de label')
                
                
            
            #~ if len(possible_labels_l0) == 0:
                #~ self.log('len(possible_labels_l0) == 0, explore new chan ')
                #~ chan_visited.append(actual_chan)
                #~ force_next_chan = None
                
                #~ final_label_l2 = None
                #~ self._plot_debug(actual_chan, ind_l0, channel_features_l0, labels_l0, possible_labels_l0, ind_l2, labels_l2, possible_labels_l2, final_label_l2)
                
                #~ continue
            
            # loop over subcluster only on channel
            # and try to subcluster it with new features
            labels_l2 = np.zeros(labels_l0.size, dtype='int64')
            labels_l2[:] = -1
            label_offset = 0
            for label in possible_labels_l0:
                
                mask = labels_l0 == label
                ind_l1 = ind_l0[mask]
                if ind_l1.size < self.min_cluster_size:
                    continue
                
                local_ind,  = np.nonzero(mask)
                
                # no noise in features
                # TODO : with noise ?
                local_features = self.make_local_features(ind_l1, actual_chan)
                
                labels_l1 = self.one_sub_cluster(local_features)
                self.log('level1: label', label, 'ind_l1.size', ind_l1.size, 'level1: labels_l1', np.unique(labels_l1))
                
                mask2 = labels_l1>=0
                labels_l2[local_ind[mask2]] = labels_l1[mask2] + label_offset
                
                label_offset += np.max(labels_l1) + 1
            
            mask = labels_l2>=0
            ind_l2 = ind_l0[mask]
            labels_l2 = labels_l2[mask]
            
            possible_labels_l2 = np.unique(labels_l2)
            
            
            
            # explore candidate
            possible_labels_l2, candidate_labels_l2, best_chan_peak_values, peak_is_aligned, peak_is_on_chan, local_centroids = self.check_candidate_labels(ind_l2, labels_l2, actual_chan)
            self.log('possible_labels_l2', possible_labels_l2)
            self.log('candidate_labels_l2', candidate_labels_l2)
            
            
            if len(candidate_labels_l2) == 0:
                # TODO : do something else dense_mode=True
                self.log('len(candidate_labels) == 0')
                chan_visited.append(actual_chan)
                force_next_chan = None


                final_label_l2 = None
                self._plot_debug(actual_chan, ind_l0, channel_features_l0, labels_l0, possible_labels_l0, ind_l2, labels_l2, possible_labels_l2, final_label_l2)
                continue
            
            #~ final_label_l2 = None
            #~ self._plot_debug(actual_chan, ind_l0, channel_features_l0, labels_l0, possible_labels_l0, ind_l2, labels_l2, possible_labels_l2, final_label_l2)
            
            
            mask = np.in1d(possible_labels_l2, candidate_labels_l2, assume_unique=True)
            best_ind = np.argmax(best_chan_peak_values[mask])
            best_label_l2 = candidate_labels_l2[best_ind ]
            
            #~ best_peak_value = best_chan_peak_values[mask][best_ind]
            
            self.log('best_ind', best_ind, 'best_label_l2', best_label_l2, 'best_peak_value', best_chan_peak_values[mask][best_ind])
            
            final_label_l2 = best_label_l2

            # remove trash : TODO
            #~ ind_trash_label = ind_keep_l2[local_labels == -1]
            #~ self.log('ind_trash_label.shape', ind_trash_label.shape)
            #~ cluster_labels[ind_trash_label] = -1
            
            ind_new_label = ind_l2[final_label_l2 == labels_l2]
            
            self.log('ind_new_label.shape', ind_new_label.shape)

            cluster_labels[cluster_labels>=k] += 1
            cluster_labels[ind_new_label] -= 1
            
            k += 1
            chan_visited = []
            force_next_chan = None
            
            self._plot_debug(actual_chan, ind_l0, channel_features_l0, labels_l0, possible_labels_l0, ind_l2, labels_l2, possible_labels_l2, final_label_l2)
            
            continue



            self._plot_debug(actual_chan, ind_l0, channel_features_l0, labels_l0, possible_labels_l0, ind_l2, labels_l2, possible_labels_l2, final_label_l2)
            
            
            """
            exit()
            

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

                candidate_labels_l2, possible_peak_values, possible_chan_peak, local_centroids= self.check_candidate_labels(ind_keep_l2, local_labels, adjacency)
                
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


            
        self.log('END loop', np.sum(cluster_labels==-1))
        return cluster_labels

    def _plot_debug(self, actual_chan, ind_l0, channel_features_l0, labels_l0, possible_labels_l0, ind_l2, labels_l2, possible_labels_l2, final_label_l2):
        if not self.debug_plot:
            return
        
        from .matplotlibplot import plot_waveforms_density
        fig, axs = plt.subplots(ncols=3, nrows=2)
        #~ plt.show()
        
        colors = plt.cm.get_cmap('jet', len(possible_labels_l0))
        if possible_labels_l2 is not None:
            colors2 = plt.cm.get_cmap('tab10', len(possible_labels_l2))
        
        noise_size = self.noise_features.shape[0]
        
        print(channel_features_l0.shape)
        print(labels_l0.shape)
        # channel feature
        ax = axs[0, 0]
        #~ ax.set_title('actual_chan {}'.format(actual_chan))
        
        feat0, feat1 = channel_features_l0[noise_size:, 0], channel_features_l0[noise_size:, 1]
        
        ax.scatter(feat0, feat1, s=1, color='k')
        ax.scatter(channel_features_l0[:noise_size, 0], channel_features_l0[:noise_size, 1], s=1, color='r')        
        
        for l, label in enumerate(possible_labels_l0):
            #~ if label not in candidate_labels:
                #~ continue
                
            sel = label == labels_l0
            if label<0: 
                color = 'k'
            else:
                color=colors(l)
            
            #~ if label in candidate_labels:
                #~ s = 3
            #~ else:
                #~ s = 1
            s = 1
            
            ax.scatter(feat0[sel], feat1[sel], s=s, color=color)


        ax = axs[1, 0]
        wf_chan = self.waveforms[:,:, [actual_chan]][ind_l0, :, :]
        
        #~ bin_min, bin_max, bin_size =np.min(wf_chan)-3, np.max(wf_chan)+3, 0.2
        #~ im = plot_waveforms_density(wf_chan, bin_min, bin_max, bin_size, ax=ax)
        #~ im.set_clim(0, 50)
        #~ fig.colorbar(im)
        
        if wf_chan.shape[0] > 400:
            wf_chan = wf_chan[:400, :, :]
        ax.plot(wf_chan.swapaxes(1,2).reshape(wf_chan.shape[0], -1).T, color='k', alpha=0.1)
        
        for l, label in enumerate(possible_labels_l0):

            sel = label == labels_l0
            if label<0: 
                color = 'k'
            else:
                color=colors(l)
        
            ind = ind_l0[sel]
            
            wf = self.waveforms[:,:, actual_chan][ind, :].mean(axis=0)
            
            #~ if candidate_labels is not None and label in candidate_labels:
                #~ ls = '-'
            #~ else:
                #~ ls = '--'
            
            ax.plot(wf, color=color, ls='-')
            
        #~ plt.show()
        
        
        #~ if local_labels is not None:
            #~ ax = axs[0, 1]
            
            
            #~ noise_size = self.noise_features.shape[0]
            #~ noise_size = 0
            #~ ax.scatter(reduced_features[noise_size:, 0], reduced_features[noise_size:, 1], s=1, color='black')
            #~ ax.scatter(reduced_features[:noise_size, 0], reduced_features[:noise_size, 1], s=1, color='r')

            #~ for l, label in enumerate(np.unique(local_labels)):
                #~ sel = label == local_labels
                #~ if label is None:
                    #~ # THIS IS A DEBUG
                    #~ print('label', label)
                    #~ print('local_labels', local_labels)
                    
                #~ if label<0: 
                    #~ color = 'k'
                #~ else:
                    #~ color=colors2(l)
                #~ if candidate_labels_l2 is not None and label in candidate_labels_l2:
                    #~ s = 3
                #~ else:
                    #~ s = 1
                #~ ax.scatter(reduced_features[noise_size:, :][sel, :][:, 0], reduced_features[noise_size:, :][sel, :][:, 1], s=s, color=color)

        #~ plt.show()
        #~ ax.axvline(-self.n_left, color='w', lw=2)        
        #~ ax.axvline(-self.n_left, color='m', lw=2)
        
        
        
        
        ax = axs[1, 1]
        
        if ind_l2.size > 0:
            mask_feat = self.mask_feat_per_adj[actual_chan]
            local_wf_features = self.features.take(ind_l2, axis=0).compress(mask_feat, axis=1)
            if local_wf_features.shape[0] > 400:
                local_wf_features = local_wf_features[:400, :]
            
            ax.plot(local_wf_features.T, color='k', alpha=0.1)
            
            local_wf_features = self.features.take(ind_l2, axis=0).compress(mask_feat, axis=1)
            
            if possible_labels_l2 is not None:
                for l, label in enumerate(np.unique(possible_labels_l2)):
                    #~ if label<0: 
                        #~ continue
                    sel = label == labels_l2
                    
                    m = local_wf_features[sel].mean(axis=0)

                    color=colors2(l)
                    ax.plot(m, color=color, alpha=1)
        
                
        
        
        
        ax = axs[0, 2]
        if ind_l2.size > 0:
            adjacency = self.channel_adjacency[actual_chan]
            wf_adj = self.waveforms[:,:, adjacency][ind_l2, :, :]
            if wf_adj.shape[0] > 400:
                wf_adj = wf_adj[:400, :, :]
            ax.plot(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).T, color='k', alpha=0.1)
            
            wf_adj = self.waveforms[:,:, adjacency][ind_l2, :, :]
            
            if possible_labels_l2 is not None:
                for l, label in enumerate(np.unique(possible_labels_l2)):
                    if label<0: 
                        continue
                    sel = label == labels_l2
                    m = wf_adj[sel].mean(axis=0)

                    color=colors2(l)
                    ax.plot(m.T.flatten(), color=color, alpha=1)
        
        
        ax = axs[1, 2]
        if ind_l2.size > 0:
            if final_label_l2 is not None:
                adjacency = self.channel_adjacency[actual_chan]
                sel = final_label_l2 == labels_l2
                wf_adj = self.waveforms[:,:, adjacency][ind_l2, :, :][sel]
                m = wf_adj.mean(axis=0)

                if wf_adj.shape[0] > 400:
                    wf_adj = wf_adj[:400, :, :]
                
                ax.plot(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).T, color='k', alpha=0.1)
                
                #~ l = possible_labels_l2.tolist().index(final_label_l2)
                #~ color=colors2(l)
                
                color='m'
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