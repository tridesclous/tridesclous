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






from .dip import diptest


import hdbscan

try:
    import isosplit5
    HAVE_ISOSPLIT5 = True
except:
    HAVE_ISOSPLIT5 = False


class PruningShears:
    def __init__(self,  
                        features,
                        channel_to_features,
                        peaks,
                        peak_index, # in catalogue constructor size
                        noise_features,
                        n_left, n_right,
                        peak_sign, threshold,
                        geometry,
                        dense_mode,
                        catalogueconstructor, # for get_some_waveform TODO find better
                        
                        
                        adjacency_radius_um=50,
                        high_adjacency_radius_um = 30,
                        
                        min_cluster_size=20,
                        max_loop=1000,
                        break_nb_remain=30,
                        auto_merge_threshold=2.,
                        max_per_cluster_for_median = 300,
                        n_components_local_pca=3,
                        
                        print_debug=False,
                        debug_plot=False
                        ):
        
   
        self.features = features
        self.channel_to_features = channel_to_features
        self.peaks = peaks
        self.peak_index = peak_index
        self.noise_features = noise_features
        self.n_left = n_left
        self.n_right = n_right
        self.width = n_right - n_left
        self.peak_sign = peak_sign
        self.threshold = threshold
        
        self.geometry = geometry
        self.dense_mode =  dense_mode
        self.catalogueconstructor = catalogueconstructor
        self.cc = catalogueconstructor
        
        
        
        #user params
        self.adjacency_radius_um = adjacency_radius_um
        self.high_adjacency_radius_um = high_adjacency_radius_um
        
        self.min_cluster_size = min_cluster_size
        self.max_loop = max_loop
        self.break_nb_remain = break_nb_remain
        self.auto_merge_threshold = auto_merge_threshold
        self.print_debug = print_debug
        self.max_per_cluster_for_median = max_per_cluster_for_median
        self.n_components_local_pca = n_components_local_pca
        
        self.nb_channel = self.channel_to_features.shape[0]
        
        self.channel_distances = sklearn.metrics.pairwise.euclidean_distances(geometry)
        #~ print('self.channel_distances', self.channel_distances)
        
        if self.dense_mode:
            self.channel_adjacency = {c: np.arange(self.nb_channel) for c in range(self.nb_channel)}
            self.channel_high_adjacency = {c: np.arange(self.nb_channel) for c in range(self.nb_channel)}
            
        else:
            self.channel_adjacency = {}
            for c in range(self.nb_channel):
                nearest, = np.nonzero(self.channel_distances[c, :] < self.adjacency_radius_um)
                self.channel_adjacency[c] = nearest
            
            self.channel_high_adjacency = {}
            for c in range(self.nb_channel):
                nearest, = np.nonzero(self.channel_distances[c, :] < self.high_adjacency_radius_um)
                self.channel_high_adjacency[c] = nearest
        
        #~ print('self.channel_adjacency', self.channel_adjacency)
        #~ print('adjacency_radius_um', adjacency_radius_um)
        


        
        #~ self.debug_plot = False
        #~ self.debug_plot = True
        self.debug_plot = debug_plot
        
        

    def log(self, *args, **kargs):
        if self.print_debug:
            print(*args, **kargs)
    
    def do_the_job(self):
        self.centroids = {}
        
        t0 = time.perf_counter()
        cluster_labels = self.explore_split_loop()
        t1 = time.perf_counter()
        self.log('explore_split_loop', t1-t0)
        

        
        return cluster_labels
    
    def next_channel(self, mask_loop, chan_visited):
        self.log('next_channel percentiles', 'peak_max.size', np.sum(mask_loop))
        self.log('chan_visited', chan_visited)
        
        #~ peak_max = self.all_peak_max[mask_loop, :]
        
        #~ self.peaks['channel'][mask_loop]
        
        
        percentiles = np.zeros(self.nb_channel)
        for c in range(self.nb_channel):
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
            
            mask = self.peaks['channel'][mask_loop] == c
            #~ print(mask.size, peak_max.shape)
            #~ x = peak_max[mask, :][:, c]
            x = np.abs(self.peaks['extremum_amplitude'][mask_loop][mask])
            if x.size >self.min_cluster_size:
                per = np.nanpercentile(x, 90)
            else:
                per = 0
            percentiles[c] = per
            
            
        
        #~ mask = percentiles > 0
        #~ print('mask.size', mask.size, 'mask.sum', mask.sum())
        
        order_visit = np.argsort(percentiles)[::-1]
        percentiles = percentiles[order_visit]
        
        mask = (percentiles > 0) & ~np.isin(order_visit, chan_visited)
        
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
        
        
        #~ n_components = min(local_data.shape[1], self.n_components_local_pca)
        #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components, whiten=True)
        
        n_components = min(local_data.shape[1]-1, self.n_components_local_pca)
        pca =  sklearn.decomposition.TruncatedSVD(n_components=n_components)
        
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
                #~ centroid = np.median(self.waveforms[ind, :, :], axis=0)
                wfs = self.cc.get_some_waveforms( peaks_index=self.peak_index[ind], channel_indexes=None)
                centroid = np.median(wfs, axis=0)
            else:
                #~ centroid = np.median(self.waveforms[ind, :, :][:, :, local_channels], axis=0)
                local_channels = self.channel_adjacency[actual_chan]
                wfs = self.cc.get_some_waveforms( peaks_index=self.peak_index[ind], channel_indexes=local_channels)
                centroid = np.median(wfs, axis=0)
            
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
        #~ ind_peak = -self.n_left
        #~ self.all_peak_max = self.waveforms[:, ind_peak, : ].copy()
        #~ if self.peak_sign == '-' :
            #~ self.all_peak_max *= -1
        
        #~ print('all_peak_max done')
        
        #~ nb_channel = self.waveforms.shape[2]
        
        if self.dense_mode:
            # global pca so no sparse used
            pass
            
        else:
            # general case 
            n_components_by_channel = self.features.shape[1] // self.nb_channel
            
            #~ self.weights_feat_chans = {}
            self.mask_feat_per_chan = {}
            self.mask_feat_per_adj = {}
            for chan in range(self.nb_channel):
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
                #~ weights = []
                #~ for adj_chan in adjacency:
                    #~ d = self.channel_distances[chan, adj_chan]
                    #~ w = np.exp(-d/self.adjacency_radius_um * 2) # TODO fix this factor 2
                    #~ print('chan', chan, 'adj_chan', adj_chan, 'd', d, 'w', w)
                    #~ weights += [ w ] * n_components_by_channel
                #~ self.weights_feat_chans[chan] = np.array(weights).reshape(1, -1)
        
        force_next_chan = None
        
        cluster_labels = np.zeros(self.features.shape[0], dtype='int64')
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
            
            if not self.dense_mode:
                adjacency = self.channel_adjacency[actual_chan]
                high_adjacency = self.channel_high_adjacency[actual_chan]
                self.log('adjacency', adjacency, 'high_adjacency', high_adjacency)
            
            
            mask_thresh = np.zeros(mask_loop.size, dtype='bool')
            if self.dense_mode:
                mask_thresh[:] = True
            else:
                #~ peak_max = self.all_peak_max[mask_loop, :]
                #~ mask_thresh[mask_loop] = peak_max[:, actual_chan] > self.threshold
                mask_thresh[mask_loop] = np.in1d(self.peaks['channel'][mask_loop], high_adjacency)
                
            self.log('mask_loop.size', mask_loop.size, 'mask_loop.sum', mask_loop.sum(), 'mask_thresh.sum', mask_thresh.sum())
            
            ind_l0,  = np.nonzero(mask_loop & mask_thresh)
            self.log('ind_l0.size', ind_l0.size)
            
            if ind_l0.size < self.min_cluster_size:
                force_next_chan = None
                chan_visited.append(actual_chan)
                self.log('len(ind_l0.size) < self.min_cluster_size')
                features_l0 = None
                labels_l0 = None
                possible_labels_l0 = None
                candidate_labels_l0 = None
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
                if len(possible_labels_l0) == 0:
                    if self.dense_mode:
                        self.log('no more with dense mode')
                        break
                    else:
                
                        self.log('Add noise do not work, explore other channel')
                        force_next_chan = None
                        chan_visited.append(actual_chan)
                        final_label = None
                        candidate_labels_l0 = None
                        final_label = None
                        self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                        continue
            
            #~ if len(possible_labels_l0) == 0:
                #~ # experimental!!!!!!
                #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=2, whiten=True)
                #~ reduced_features = pca.fit_transform(wf_features)
                
                #~ pval = diptest(np.sort(reduced_features[:, 0]), numt=200)
                
                #~ if pval is not None and pval>0.2:
                    #~ self.log('len(possible_labels_l0) == 0 BUT diptest pval', pval)

                    #~ if self.dense_mode:
                        #~ wf_l0 = self.waveforms.take(ind_l0, axis=0)
                    #~ else:
                        #~ mask_feat = self.mask_feat_per_adj[actual_chan]
                        #~ wf_l0 = self.waveforms.take(ind_l0, axis=0).take(adjacency, axis=2)
                    #~ centroid = np.median(wf_l0, axis=0)
                    #~ centroid = centroid[np.newaxis, :, :]
                    #~ out_up = np.any(np.any(wf_l0 > centroid + 4, axis=2), axis=1)
                    #~ out_dw = np.any(np.any(wf_l0 < centroid - 4, axis=2), axis=1)
                    #~ ok = ~out_up & ~out_dw
                    #~ if np.sum(ok) < self.min_cluster_size:
                        #~ self.log('!!!!!!! Not save trash!!!!!! np.sum(ok)', np.sum(ok))
                        #~ force_next_chan = None
                        #~ chan_visited.append(actual_chan)
                        #~ final_label = None
                        #~ candidate_labels_l0 = None
                        #~ self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                        #~ continue
                    #~ labels_l0[ok] = 0
                    #~ labels_l0[~ok] = -1
                    #~ possible_labels_l0 = np.array([0], dtype='int64')
                #~ else:
                    #~ self.log('len(possible_labels_l0) == 0 AND diptest pval', pval, 'no cluster at all')
                    #~ if self.dense_mode:
                        #~ self.log('no more with dense mode')
                        #~ break
                    #~ else:
                        #~ force_next_chan = None
                        #~ chan_visited.append(actual_chan)
                        #~ #TODO dip test ????
                        #~ self.log('len(possible_labels_l0) == 0')
                        #~ final_label = None
                        #~ self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                        #~ continue



            #~ self.debug_plot = iloop>20
            
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
            
            final_label_elsewhere = None
            if elsewhere_peak_val is not None:
                # TODO put in params
                ratio_peak_force_explore = 1.5
                if (peak_val is not None and elsewhere_peak_val > peak_val*ratio_peak_force_explore) or peak_val is None:
                    # there is a better option elsewhere
                    force_next_chan = best_chan[elsewhere_mask_l0][ind_best_elsewhere]
                    if len(chan_visited)>0 and force_next_chan == chan_visited[-1]:
                        # last vist is the next one so lets avoid ping pong
                        if peak_val is None:
                            # keep actual chan anyway
                            final_label_elsewhere =possible_labels_l0[elsewhere_mask_l0][ind_best_elsewhere]
                            self.log('Best label other channel', force_next_chan, 'label', final_label_elsewhere)
                        else:
                            force_next_chan = None
                    
                    elif force_next_chan in chan_visited:
                        # but not last one
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
                        # elsewhere have better option and not visisted yet
                        chan_visited.append(actual_chan)
                        self.log('force_next_chan', force_next_chan, 'actual_chan', actual_chan)
                        
                        final_label = None
                        self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                        continue
            
            if peak_val is None and final_label_elsewhere is None:
                # no candidate and no elsewhere
                force_next_chan = None
                chan_visited.append(actual_chan)
                final_label = None
                self._plot_debug(actual_chan, ind_l0, features_l0, labels_l0, possible_labels_l0, candidate_labels_l0, final_label)
                continue                
            
            if final_label_elsewhere is not None:
                final_label = final_label_elsewhere
            else:
                final_label = possible_labels_l0[candidate_mask_l0][ind_best]
            
            self.log('final_label', final_label, 'peak_val', peak_val)
            
            # remove trash : TODO
            #~ if not self.dense_mode:
            #~ if len(possible_labels_l0) > 2:
            if True:
                # put -1 labels to trash
                ind_trash_label = ind_l0[labels_l0 == -1]
                cluster_labels[ind_trash_label] = -1
                self.log('trash -1 n=', ind_trash_label.shape)
                
                #~ # put peak_is_on_chan=True / aligned False
                #~ bad_labels = possible_labels_l0[~peak_is_aligned & peak_is_on_chan]
                # put aligned False
                bad_labels = possible_labels_l0[~peak_is_aligned]
                ind_trash_label = ind_l0[np.in1d(labels_l0, bad_labels)]
                self.log('trash bad_labels for trash', bad_labels, 'n=', ind_trash_label.size)
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
        
        import matplotlib.pyplot as plt
        from .matplotlibplot import plot_waveforms_density
        fig, axs = plt.subplots(ncols=3, nrows=2)
        
        colors = plt.cm.get_cmap('Set3', len(possible_labels_l0))
        colors = {possible_labels_l0[l]:colors(l) for l in range(len(possible_labels_l0))}
        colors[-1] = 'k'
        
        ax = axs[0, 0]
        
        ax.set_title('actual_chan '+str(actual_chan))
        
        #~ if self.dense_mode:
            #~ mask_feat = slice(None)
        #~ else:
            #~ mask_feat = self.mask_feat_per_adj[actual_chan]
        local_wf_features = features_l0
        #~ print('local_wf_features.shape', local_wf_features.shape)
        
        #~ local_wf_features = features_l0
        
        if possible_labels_l0 is not None:
            for l, label in enumerate([-1] + possible_labels_l0.tolist()):
                #~ if label<0: 
                    #~ continue
                sel = label == labels_l0
                if np.sum(sel) ==0:
                    continue
                m = local_wf_features[sel].mean(axis=0)

                ind, = np.nonzero(sel)

                #~ if ind.size> 200:
                    #~ ind = ind[:200]
                if ind.size> self.max_per_cluster_for_median:
                    ind = ind[:self.max_per_cluster_for_median]

                color=colors[label]
                
                ax.plot(local_wf_features[ind].T, color=color, alpha=0.3)
                
                if label>=0:
                    ax.plot(m, color=color, alpha=1)
        
        ax = axs[0, 1]
        #~ n_components = min(local_wf_features.shape[1], self.n_components_local_pca)
        #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components, whiten=True)
        n_components = min(local_wf_features.shape[1]-1, self.n_components_local_pca)
        pca =  sklearn.decomposition.TruncatedSVD(n_components=n_components)
        reduced_features_l0 = pca.fit_transform(local_wf_features)
        #~ print('reduced_features_l0.shape', reduced_features_l0.shape)
        if possible_labels_l0 is not None:
            for l, label in enumerate([-1] + possible_labels_l0.tolist()):
                sel = label == labels_l0
                if np.sum(sel) ==0:
                    continue
                color=colors[label]
                #~ print('label', label, 'color', color)
                ax.scatter(reduced_features_l0[:, 0][sel], reduced_features_l0[:, 1][sel], color=color, s=2)

        ax = axs[1, 1]
        if possible_labels_l0 is not None:
            for l, label in enumerate([-1] + possible_labels_l0.tolist()):
                sel = label == labels_l0
                if np.sum(sel) ==0:
                    continue
                ind, = np.nonzero(sel)
                #~ if ind.size> 200:
                    #~ ind = ind[:200]
                if ind.size> self.max_per_cluster_for_median:
                    ind = ind[:self.max_per_cluster_for_median]
                    
                color=colors[label]
                ax.plot(reduced_features_l0[ind].T, color=color, alpha=0.3, lw=1)


                
        
        
        ax = axs[1, 0]
        adjacency = self.channel_adjacency[actual_chan]
        #~ print('plot actual_chan', actual_chan, 'adjacency', adjacency, )
        #~ wf_adj = self.waveforms[:,:, adjacency][ind_l0, :, :]
        wf_adj = self.cc.get_some_waveforms(self.peak_index[ind_l0], channel_indexes=adjacency)
        
        
        if possible_labels_l0 is not None:
            for l, label in enumerate([-1] + possible_labels_l0.tolist()):
                sel = label == labels_l0
                if np.sum(sel) ==0:
                    continue
                m = wf_adj[sel].mean(axis=0)
                ind, = np.nonzero(sel)

                #~ if ind.size> 200:
                    #~ ind = ind[:200]
                
                if ind.size> self.max_per_cluster_for_median:
                    ind = ind[:self.max_per_cluster_for_median]
                
                color=colors[label]
                
                ax.plot(wf_adj[ind].swapaxes(1,2).reshape(ind.size, -1).T, color=color, alpha=0.3)
                
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

        ax = axs[0, 2]
        if final_label is not None:
            
            adjacency = self.channel_adjacency[actual_chan]
            sel = final_label == labels_l0
            
            ax.set_title('keep size {}'.format(np.sum(sel)))
            
            #~ wf_adj = self.waveforms[:,:, adjacency][ind_l0, :, :][sel]
            wf_adj = self.cc.get_some_waveforms(self.peak_index[ind_l0], channel_indexes=adjacency)[sel]
            
            m = wf_adj.mean(axis=0)

            if wf_adj.shape[0] > 400:
                wf_adj = wf_adj[:400, :, :]
            
            color=colors[final_label]
            ax.plot(wf_adj.swapaxes(1,2).reshape(wf_adj.shape[0], -1).T, color=color, alpha=0.1)
            ax.plot(m.T.flatten(), color=color, alpha=1, lw=2)

        ax = axs[1, 2]
        if final_label is not None:
            sel = -1 == labels_l0
            
            ax.set_title('trash size {}'.format(np.sum(sel)))
            
            if np.sum(sel) > 0:
                #~ wf_adj = self.waveforms[:,:, adjacency][ind_l0, :, :][sel]
                wf_adj = self.cc.get_some_waveforms(self.peak_index[ind_l0], channel_indexes=adjacency)[sel]
                
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

