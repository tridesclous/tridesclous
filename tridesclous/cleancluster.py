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

from joblib import Parallel, delayed




import matplotlib.pyplot as plt



from .dip import diptest
from .waveformtools import equal_template


import hdbscan

debug_plot = False
#~ debug_plot = True


def _get_sparse_waveforms_flatten(cc, dense_mode, label, channel_adjacency, n_spike_for_centroid=None):
    peak_index, = np.nonzero(cc.all_peaks['cluster_label'] == label)
    if n_spike_for_centroid is not None and peak_index.size>n_spike_for_centroid:
        keep = np.random.choice(peak_index.size, n_spike_for_centroid, replace=False)
        peak_index = peak_index[keep]

    if dense_mode:
        waveforms = cc.get_some_waveforms(peak_index, channel_indexes=None)
        extremum_channel = 0
        centroid = np.median(waveforms, axis=0)
    else:
        waveforms = cc.get_some_waveforms(peak_index, channel_indexes=None)
        centroid = np.median(waveforms, axis=0)
        
        peak_sign = cc.info['peak_detector_params']['peak_sign']
        n_left = cc.info['waveform_extractor_params']['n_left']
        
        if peak_sign == '-':
            extremum_channel = np.argmin(centroid[-n_left,:], axis=0)
        elif peak_sign == '+':
            extremum_channel = np.argmax(centroid[-n_left,:], axis=0)
        # TODO by sparsity level threhold and not radius
        adjacency = channel_adjacency[extremum_channel]
        waveforms = waveforms.take(adjacency, axis=2)
        
    wf_flat = waveforms.swapaxes(1,2).reshape(waveforms.shape[0], -1)
    
    return waveforms, wf_flat, peak_index
    

def _compute_one_dip_test(cc, dirname, chan_grp, label, n_components_local_pca, adjacency_radius_um):
    # compute dip test to try to over split
    from .dataio import DataIO
    from .catalogueconstructor import CatalogueConstructor
    
    if cc is None:
        dataio = DataIO(dirname)
        cc = CatalogueConstructor(dataio=dataio, chan_grp=chan_grp)

    peak_sign = cc.info['peak_detector_params']['peak_sign']
    dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['waveform_extractor_params']['n_left']
    n_right = cc.info['waveform_extractor_params']['n_right']
    peak_width = n_right - n_left
    nb_channel = cc.nb_channel
    
    if dense_mode:
        channel_adjacency = {c: np.arange(nb_channel) for c in range(nb_channel)}
    else:
        channel_adjacency = {}
        for c in range(nb_channel):
            nearest, = np.nonzero(cc.channel_distances[c, :] < adjacency_radius_um)
            channel_adjacency[c] = nearest

    
    waveforms, wf_flat, peak_index = _get_sparse_waveforms_flatten(cc, dense_mode, label, channel_adjacency, n_spike_for_centroid=cc.n_spike_for_centroid)
    
    
    #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components_local_pca, whiten=True)
    
    n_components = min(wf_flat.shape[1]-1, n_components_local_pca)
    pca =  sklearn.decomposition.TruncatedSVD(n_components=n_components)
    
    feats = pca.fit_transform(wf_flat)
    pval = diptest(np.sort(feats[:, 0]), numt=200)
    
    return pval


    


def auto_split(catalogueconstructor, 
                        n_spike_for_centroid=None,
                        adjacency_radius_um = 30,
                        n_components_local_pca=3,
                        pval_thresh=0.1,
                        min_cluster_size=20,
                        maximum_shift=2,
                        n_jobs=-1,
                        #~ n_jobs=1,
                        joblib_backend='loky',
            ):
    cc = catalogueconstructor
    peak_sign = cc.info['peak_detector_params']['peak_sign']
    dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['waveform_extractor_params']['n_left']
    n_right = cc.info['waveform_extractor_params']['n_right']
    peak_width = n_right - n_left
    nb_channel = cc.nb_channel
    
    if dense_mode:
        channel_adjacency = {c: np.arange(nb_channel) for c in range(nb_channel)}
    else:
        channel_adjacency = {}
        for c in range(nb_channel):
            nearest, = np.nonzero(cc.channel_distances[c, :] < adjacency_radius_um)
            channel_adjacency[c] = nearest
    
    if len(cc.positive_cluster_labels) ==0:
        return
    
    m = np.max(cc.positive_cluster_labels) + 1
    
    # pvals = []
    # for label in cc.positive_cluster_labels:
    #     pval = _compute_one_dip_test(cc.dataio.dirname, cc.chan_grp, label, n_components_local_pca, adjacency_radius_um)
    #     print('label', label,'pval', pval, pval<pval_thresh)
    #     pvals.append(pval)
    
    if cc.memory_mode == 'ram':
        # prevent paralell because not persistent
        n_jobs = 1
        cc2 = cc
    else:
        cc2 = None
    
    pvals = Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                    delayed(_compute_one_dip_test)(cc2, cc.dataio.dirname, cc.chan_grp, label, n_components_local_pca, adjacency_radius_um)
                    for label in cc.positive_cluster_labels)
    
    pvals = np.array(pvals)
    inds, = np.nonzero(pvals<pval_thresh)
    splitable_labels = cc.positive_cluster_labels[inds]
    #~ print('splitable_labels', splitable_labels)
    
    for label in splitable_labels:
        
        waveforms, wf_flat, peak_index = _get_sparse_waveforms_flatten(cc, dense_mode, label, channel_adjacency, n_spike_for_centroid=None)
        
        #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components_local_pca, whiten=True)
        n_components = min(wf_flat.shape[1]-1, n_components_local_pca)
        pca =  sklearn.decomposition.TruncatedSVD(n_components=n_components)
        feats = pca.fit_transform(wf_flat)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=False, metric='l2')
        sub_labels = clusterer.fit_predict(feats[:, :2])
        unique_sub_labels = np.unique(sub_labels)
        #~ print(unique_sub_labels)
        if unique_sub_labels.size ==  1 and unique_sub_labels[0] == -1:
            sub_labels[:] = 0
            unique_sub_labels = np.unique(sub_labels)
            
        if not dense_mode:
            peak_is_aligned = check_peak_all_aligned(sub_labels, waveforms, peak_sign, n_left, maximum_shift)

        if debug_plot:
            fig, ax= plt.subplots()
            ax.plot(np.median(wf_flat, axis=0))
            ax.set_title('label '+str(label))
            for i in range(waveforms.shape[2]):
                ax.axvline(i*peak_width-n_left, color='k')
            ax.set_title('pval ' + str(pval))
            plt.show()
        
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
                    #~ print(label, m, new_label)
                    cc.all_peaks['cluster_label'][peak_index[sub_mask]] = new_label
                    cc.add_one_cluster(new_label)
                    
                    m += 1
            
            cc.pop_labels_from_cluster([label])
            
            #~ m += np.max(unique_sub_labels) + 1
        

            #~ if True:
            #~ if False:
            if debug_plot:
                print('label', label,'pval', pval, pval<pval_thresh)
                
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



def check_peak_all_aligned(local_labels, waveforms, peak_sign, n_left, maximum_shift):
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
        
        al = np.abs(-n_left - pos_peak) <= maximum_shift
        peak_is_aligned.append(al)
    
    return np.array(peak_is_aligned)



def trash_not_aligned(cc, maximum_shift=2):
    n_left = cc.info['waveform_extractor_params']['n_left']
    peak_sign = cc.info['peak_detector_params']['peak_sign']
    
    to_remove = []
    for k in list(cc.positive_cluster_labels):
        #~ print(k)

        centroid = cc.get_one_centroid(k)
        
        if peak_sign == '-':
            chan_peak = np.argmin(np.min(centroid, axis=0))
            extremum_index = np.argmin(centroid[:, chan_peak])
            peak_val = centroid[-n_left, chan_peak]
        elif peak_sign == '+':
            chan_peak = np.argmax(np.max(centroid, axis=0))
            extremum_index = np.argmax(centroid[:, chan_peak])
            peak_val = centroid[-n_left, chan_peak]

        if np.abs(-n_left - extremum_index)>maximum_shift:
            if debug_plot:
                n_left = cc.info['waveform_extractor_params']['n_left']
                n_right = cc.info['waveform_extractor_params']['n_right']
                peak_width = n_right - n_left
                
                print('remove not aligned peak', 'k', k)
                fig, ax = plt.subplots()
                #~ centroid = centroids[k]
                ax.plot(centroid.T.flatten())
                ax.set_title('not aligned peak')
                for i in range(centroid.shape[1]):
                    ax.axvline(i*peak_width-n_left, color='k')
                plt.show()
            
            mask = cc.all_peaks['cluster_label'] == k
            cc.all_peaks['cluster_label'][mask] = -1
            to_remove.append(k)
        
            
    cc.pop_labels_from_cluster(to_remove)


def auto_merge(catalogueconstructor,
                        auto_merge_threshold=2.3,
                        maximum_shift=2,
                        amplitude_factor_thresh = 0.2,
        ):
    cc = catalogueconstructor
    peak_sign = cc.info['peak_detector_params']['peak_sign']
    #~ dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['waveform_extractor_params']['n_left']
    n_right = cc.info['waveform_extractor_params']['n_right']
    peak_width = n_right - n_left
    threshold = cc.info['peak_detector_params']['relative_threshold']
    
    while True:
        
        labels = cc.positive_cluster_labels.copy()
        
        
        nb_merge = 0
        
        n = labels.size
        
        #~ pop_from_centroids = []
        new_centroids = []
        pop_from_cluster = []
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
                #~ print('  k2', k2)
                
                ind1 = cc.index_of_label(k1)
                extremum_amplitude1 = np.abs(cc.clusters[ind1]['extremum_amplitude'])
                centroid1 = cc.get_one_centroid(k1)

                ind2 = cc.index_of_label(k2)
                extremum_amplitude2 = np.abs(cc.clusters[ind2]['extremum_amplitude'])
                centroid2 = cc.get_one_centroid(k2)
        
                thresh = max(extremum_amplitude1, extremum_amplitude2) * amplitude_factor_thresh
                thresh = max(thresh, auto_merge_threshold)
                #~ print('thresh', thresh)
                
                #~ t1 = time.perf_counter()
                do_merge = equal_template(centroid1, centroid2, thresh=thresh, n_shift=maximum_shift)
                #~ t2 = time.perf_counter()
                #~ print('equal_template', t2-t1)
                
                #~ print('do_merge', do_merge)
                
                #~ if debug_plot:
                #~ print(k1, k2)
                #~ if k1==4  and k2==5:
                    #~ print(k1, k2, do_merge, thresh)
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(centroid1.T.flatten())
                    #~ ax.plot(centroid2.T.flatten())
                    #~ ax.set_title('merge ' + str(do_merge))
                    #~ plt.show()
                
                
                
                
                if do_merge:
                    #~ print('merge', k1, k2)
                    #~ cluster_labels2[cluster_labels2==k2] = k1

                    mask = cc.all_peaks['cluster_label'] == k2
                    cc.all_peaks['cluster_label'][mask] = k1
                    
                    #~ t1 = time.perf_counter()
                    #~ cc.compute_one_centroid(k1)
                    #~ t2 = time.perf_counter()
                    #~ print('cc.compute_one_centroid', t2-t1)
                    
                    new_centroids.append(k1)
                    pop_from_cluster.append(k2)
                    
                    labels[j] = -1
                    
                    nb_merge += 1
                    
                    if debug_plot:
                    
                        fig, ax = plt.subplots()
                        ax.plot(centroid1.T.flatten())
                        ax.plot(centroid2.T.flatten())
                        ax.set_title('merge '+str(k1)+' '+str(k2))
                        plt.show()
        
        #~ for k in np.unique(pop_from_cluster):
            #~ cc.pop_labels_from_cluster([k])
        pop_from_cluster = np.unique(pop_from_cluster)
        cc.pop_labels_from_cluster(pop_from_cluster)
        
        new_centroids = np.unique(new_centroids)
        new_centroids = [k for k in new_centroids if k not in pop_from_cluster]
        cc.compute_several_centroids(new_centroids)

        #~ cc.compute_one_centroid(k)
        
        
        
        #~ for k in np.unique(pop_from_centroids):
            #~ if k in centroids:
                #~ centroids.pop(k)
        
        #~ print('nb_merge', nb_merge)
        if nb_merge == 0:
            break


def trash_low_extremum(cc, min_extremum_amplitude=None):
    if min_extremum_amplitude is None:
        threshold = cc.info['peak_detector_params']['relative_threshold']
        min_extremum_amplitude = threshold + 0.5
    
    to_remove = []
    for k in list(cc.positive_cluster_labels):
        #~ print(k)
        ind = cc.index_of_label(k)
        assert k == cc.clusters[ind]['cluster_label'], 'this is a bug in trash_low_extremum'
        
        extremum_amplitude = np.abs(cc.clusters[ind]['extremum_amplitude'])
        #~ print('k', k , extremum_amplitude)
        if extremum_amplitude < min_extremum_amplitude:
            if debug_plot:
                print('k', k , extremum_amplitude, 'too small')
            
            mask = cc.all_peaks['cluster_label']==k
            cc.all_peaks['cluster_label'][mask] = -1
            to_remove.append(k)
    cc.pop_labels_from_cluster(to_remove)


def trash_small_cluster(cc, minimum_size=10):
    to_remove = []
    for k in list(cc.positive_cluster_labels):
        mask = cc.all_peaks['cluster_label']==k
        cluster_size = np.sum(mask)
        #~ print(k, cluster_size)
        if cluster_size <= minimum_size :
            cc.all_peaks['cluster_label'][mask] = -1
            to_remove.append(k)
    cc.pop_labels_from_cluster(to_remove)
