"""
Some heuristic method to clean after clustering:
  * auto-split
  * auto-merge

"""


import numpy as np
import os
import time

import sklearn
import sklearn.cluster
import sklearn.mixture
import sklearn.metrics
import sklearn.decomposition




import matplotlib.pyplot as plt

from .dip import diptest
from .waveformtools import equal_template


import hdbscan




def auto_split(catalogueconstructor, 
                        max_per_cluster=None,
                        #~ adjacency_radius_um=50,
                        high_adjacency_radius_um = 30,
                        n_components_local_pca=3,
                        #~ pval_thresh=0.1,
                        pval_thresh=0.8,
                        min_cluster_size=20,
            ):
    #~ return
    cc = catalogueconstructor
    peak_sign = cc.info['peak_detector_params']['peak_sign']
    dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['waveform_extractor_params']['n_left']
    n_right = cc.info['waveform_extractor_params']['n_right']
    #~ relative_threshold = cc.info['peak_detector_params']['relative_threshold']
    
    nb_channel = cc.nb_channel

        
    #~ print(cc.channel_distances)
    
    
    if dense_mode:
        #~ channel_adjacency = {c: np.arange(nb_channel) for c in range(nb_channel)}
        channel_high_adjacency = {c: np.arange(nb_channel) for c in range(nb_channel)}
        
    else:
        #~ channel_adjacency = {}
        #~ for c in range(nb_channel):
            #~ nearest, = np.nonzero(cc.channel_distances[c, :] < adjacency_radius_um)
            #~ channel_adjacency[c] = nearest
        
        channel_high_adjacency = {}
        for c in range(nb_channel):
            nearest, = np.nonzero(cc.channel_distances[c, :] < high_adjacency_radius_um)
            channel_high_adjacency[c] = nearest
    
    #~ print(channel_high_adjacency)

    
    
    m = np.max(cc.positive_cluster_labels) + 1
    
    
    for label in cc.positive_cluster_labels:
        #~ print()
        #~ print('label', label)
        
        peak_index, = np.nonzero(cc.all_peaks['cluster_label'] == label)
        
        if max_per_cluster is not None and peak_index.size>max_per_cluster:
            keep = np.random.choice(peak_index.size, max_per_cluster, replace=False)
            peak_index = peak_index[keep]
        
        

        if dense_mode:
            #~ waveforms = waveforms.take(ind_keep, axis=0)
            waveforms = cc.get_some_waveforms(peak_index, channel_indexes=None)
            extremum_channel = 0
            centroid = np.median(waveforms, axis=0)
        else:
            #~ centroid = np.median(waveforms[ind_keep, :, :], axis=0)
            waveforms = cc.get_some_waveforms(peak_index, channel_indexes=None)
            centroid = np.median(waveforms, axis=0)
            
            if peak_sign == '-':
                extremum_channel = np.argmin(centroid[-n_left,:], axis=0)
            elif peak_sign == '+':
                extremum_channel = np.argmax(centroid[-n_left,:], axis=0)
            # TODO by sparsity level threhold and not radius
            #~ adjacency = channel_adjacency[extremum_channel]
            #~ waveforms = waveforms.take(ind_keep, axis=0).take(adjacency, axis=2)
            high_adjacency = channel_high_adjacency[extremum_channel]
            #~ waveforms = waveforms.take(ind_keep, axis=0).take(high_adjacency, axis=2)
            waveforms = waveforms.take(high_adjacency, axis=2)
            
            
        wf_flat = waveforms.swapaxes(1,2).reshape(waveforms.shape[0], -1)
        #~ print('label', label)
        #~ print(wf_flat.shape)
        
        #~ exit()

        #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components_local_pca, whiten=True)
        
        n_components = min(wf_flat.shape[1]-1, n_components_local_pca)
        pca =  sklearn.decomposition.TruncatedSVD(n_components=n_components)
        
        
        feats = pca.fit_transform(wf_flat)
        pval = diptest(np.sort(feats[:, 0]), numt=200)
        #~ print('pval', pval)
        print('label', label,'pval', pval, pval<pval_thresh)
        
        #~ continue
        
        #~ if pval<0.2:
        if pval<pval_thresh: # TODO global params
        
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=False, metric='l2')
            sub_labels = clusterer.fit_predict(feats[:, :2])
            unique_sub_labels = np.unique(sub_labels)
            print(unique_sub_labels)
            if unique_sub_labels.size ==  1 and unique_sub_labels[0] == -1:
                sub_labels[:] = 0
                unique_sub_labels = np.unique(sub_labels)
                
            if not dense_mode:
                peak_is_aligned = check_peak_all_aligned(sub_labels, waveforms, peak_sign, n_left)
            
            if unique_sub_labels.size >1:
                for i, sub_label in enumerate(unique_sub_labels):
                    sub_mask = sub_labels == sub_label
                    
                    if dense_mode:
                        valid=True
                    else:
                        valid = peak_is_aligned[i]
                    #~ print('sub_label', 'valid', valid)
                    
                    if sub_label == -1 or not valid:
                        #~ cluster_labels[ind_keep[sub_mask]] = -1
                        cc.all_peaks['cluster_label'][peak_index[sub_mask]] = -1
                    else:
                        #~ cluster_labels[ind_keep[sub_mask]] = sub_label + m 
                        new_label = label + m
                        print(label, m, new_label)
                        cc.all_peaks['cluster_label'][peak_index[sub_mask]] = new_label
                        cc.add_one_cluster(new_label)
                        
                        m += 1
                
                cc.pop_labels_from_cluster([label])
                
                #~ m += np.max(unique_sub_labels) + 1
            

                #~ if True:
                if False:
                    fig, axs = plt.subplots(ncols=4)
                    colors = plt.cm.get_cmap('Set3', len(unique_sub_labels))
                    colors = {unique_sub_labels[l]:colors(l) for l in range(len(unique_sub_labels))}
                    colors[-1] = 'k'
                    
                    for sub_label in unique_sub_labels:
                        #~ valid = sub_label in possible_labels[peak_is_aligned]
                        if dense_mode:
                            valid=True
                        else:
                            valid = peak_is_aligned[i]                    
                        
                        sub_mask = sub_labels == sub_label
                        if valid:
                            ls = '-'
                            color = colors[sub_label]
                        else:
                            ls = '--'
                            color = 'k'

                        ax = axs[0]
                        ax.plot(wf_flat[sub_mask].T, color=color, alpha=0.1)
                            
                        ax = axs[3]
                        if sub_label>=0:
                            ax.plot(np.median(wf_flat[sub_mask], axis=0), color=color, lw=2, ls=ls)
                    
                    for sub_label in unique_sub_labels:
                        if dense_mode:
                            valid=True
                        else:
                            valid = peak_is_aligned[i]                    
                        
                        sub_mask = sub_labels == sub_label
                        color = colors[sub_label]
                        if valid:
                            color = colors[sub_label]
                        else:
                            color = 'k'
                        ax = axs[1]
                        ax.plot(feats[sub_mask].T, color=color, alpha=0.1)
                    
                        ax = axs[2]
                        ax.scatter(feats[sub_mask][:, 0], feats[sub_mask][:, 1], color=color)
                    plt.show()


def check_peak_all_aligned(local_labels, waveforms, peak_sign, n_left):
    
    peak_is_aligned = []
    for k in np.unique(local_labels):
        wfs = waveforms[local_labels == k]
        centroid = np.median(wfs, axis=0)
        
        if peak_sign == '-':
            chan_peak_local = np.argmin(np.min(centroid, axis=0))
            pos_peak = np.argmin(centroid[:, chan_peak_local])
        elif peak_sign == '+':
            chan_peak_local = np.argmax(np.max(centroid, axis=0))
            pos_peak = np.argmax(centroid[:, chan_peak_local])    
        
        al = np.abs(-n_left - pos_peak) <= 1        
        peak_is_aligned.append(al)
    
    return np.array(peak_is_aligned)


#~ debug_plot = False
debug_plot = True

def auto_merge(catalogueconstructor,
                        max_per_cluster=None,
                        auto_merge_threshold=2.,
        ):
    cc = catalogueconstructor
    peak_sign = cc.info['peak_detector_params']['peak_sign']
    #~ dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['waveform_extractor_params']['n_left']
    n_right = cc.info['waveform_extractor_params']['n_right']
    peak_width = n_right - n_left
    threshold = cc.info['peak_detector_params']['relative_threshold']
    
    #~ cluster_labels2 = cluster_labels.copy()
    
    
    # centroids already computed before
    
    extremum_channel = {}
    extremum_index = {}
    extremum_amplitude = {}
    centroids = {}
    
    while True:
        #~ print('')
        #~ print('new loop')
        #~ labels = np.unique(cluster_labels2)
        #~ labels = labels[labels>=0]
        #~ n = labels.size
        #~ print(labels)
        
        #~ for ind, k in enumerate(labels):
        
        labels = cc.positive_cluster_labels.copy()
        print(labels)
        
        for ind, k in enumerate(labels):
        
        
            if k not in centroids:
                # TODO take from cc.centroids
                peak_index, = np.nonzero(cc.all_peaks['cluster_label'] == k)
                
                if max_per_cluster is not None and peak_index.size>max_per_cluster:
                    keep = np.random.choice(peak_index.size, max_per_cluster, replace=False)
                    peak_index = peak_index[keep]
                
                waveforms = cc.get_some_waveforms(peak_index, channel_indexes=None)
                centroid = np.median(waveforms, axis=0)
                centroids[k] = centroid
            
            if k not in extremum_index:
                centroid = centroids[k]
                
                if peak_sign == '-':
                    chan_peak = np.argmin(np.min(centroid, axis=0))
                    pos_peak = np.argmin(centroid[:, chan_peak])
                    peak_val = centroid[-n_left, chan_peak]
                elif peak_sign == '+':
                    chan_peak = np.argmax(np.max(centroid, axis=0))
                    pos_peak = np.argmax(centroid[:, chan_peak])
                    peak_val = centroid[-n_left, chan_peak]
                extremum_index[k] = pos_peak
                extremum_channel[k] = chan_peak
                extremum_amplitude[k] = np.abs(peak_val)
        
        
        #eliminate when best peak not aligned
        # eliminate when peak value is too small
        # TODO move this in the main loop!!!!!
        #~ for ind, k in enumerate(labels):
        labels = cc.positive_cluster_labels.copy()
        for ind, k in enumerate(labels):
            
            if np.abs(-n_left - extremum_index[k])>2:
                print('remove not aligned peak', 'k', k)
                #delete
                #~ cluster_labels2[cluster_labels2 == k] = -1
                mask = cc.all_peaks['cluster_label'] == k
                cc.all_peaks['cluster_label'][mask] = -1
                cc.pop_labels_from_cluster([k])
                #~ cc.add_one_cluster(new_label)
                
                labels[ind] = -1
                centroids.pop(k)
                #~ extremum_index.pop(k) # usefull ?
                #~ extremum_channel.pop(k) # usefull ?
                #~ extremum_amplitude.pop(k) # usefull ?
                continue
            
            # TODO check if this is really  relevant
            if np.abs(extremum_amplitude[k]) < threshold + 0.5:
                print('remove small peak', 'k', k, 'peak_val', extremum_amplitude[k])
                #~ cluster_labels2[cluster_labels2 == k] = -1

                mask = cc.all_peaks['cluster_label'] == k
                cc.all_peaks['cluster_label'][mask] = -1
                cc.pop_labels_from_cluster([k])
                
                #~ extremum_index.pop(k) # usefull ?
                #~ extremum_channel.pop(k) # usefull ?
                #~ extremum_amplitude.pop(k) # usefull ?

                if debug_plot:
                    fig, ax = plt.subplots()
                    centroid = centroids[k]
                    ax.plot(centroid.T.flatten(), color='g')
                    for i in range(centroid.shape[1]):
                        ax.axvline(i*peak_width-n_left, color='k')
                    ax.set_title('delete')
                    plt.show()
                
                labels[ind] = -1
                centroids.pop(k)
                    
                
                continue
            
        
        n_shift = 2
        nb_merge = 0
        amplitude_factor_thresh = 0.2
        
        n = labels.size
        
        pop_from_centroids = []
        for i in range(n):
            k1 = labels[i]
            if k1 == -1:
                # this can have been removed yet
                continue
            
            for j in range(i+1, n):
                k2 = labels[j]
                if k2 == -1:
                    # this can have been removed yet
                    continue
                
                #~ print(k1, k2)
                
                thresh = max(extremum_amplitude[k1], extremum_amplitude[k2]) * amplitude_factor_thresh
                thresh = max(thresh, auto_merge_threshold)
                #~ print('thresh', thresh)
                
                if equal_template(centroids[k1], centroids[k2], thresh=thresh, n_shift=n_shift):
                    #~ print('merge', k1, k2)
                    #~ cluster_labels2[cluster_labels2==k2] = k1

                    mask = cc.all_peaks['cluster_label'] == k2
                    cc.all_peaks['cluster_label'][mask] = k1
                    cc.pop_labels_from_cluster([k2])
                    
                    nb_merge += 1
                    
                    # remove from centroid doct to recompute it
                    pop_from_centroids.append(k1)
                    pop_from_centroids.append(k2)
                    
                    if debug_plot:
                    
                        fig, ax = plt.subplots()
                        ax.plot(centroids[k1].T.flatten())
                        ax.plot(centroids[k2].T.flatten())
                        ax.set_title('merge')
                        plt.show()
        
        for k in np.unique(pop_from_centroids):
            if k in centroids:
                centroids.pop(k)
        
        #~ print('nb_merge', nb_merge)
        if nb_merge == 0:
            break
    

