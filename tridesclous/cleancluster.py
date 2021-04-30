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
import joblib



from .dip import diptest
from .waveformtools import  compute_shared_channel_mask, equal_template_with_distrib_overlap, equal_template_with_distance


import hdbscan

debug_plot = False
#~ debug_plot = True


def _get_sparse_waveforms_flatten(cc, dense_mode, label, channel_adjacency):
    peak_index, = np.nonzero(cc.all_peaks['cluster_label'] == label)
    
    if dense_mode:
        waveforms = cc.get_some_waveforms(peak_index, channel_indexes=None)
        extremum_channel = 0
        centroid = np.median(waveforms, axis=0)
    else:
        waveforms = cc.get_some_waveforms(peak_index, channel_indexes=None)
        centroid = np.median(waveforms, axis=0)
        
        peak_sign = cc.info['peak_detector']['peak_sign']
        n_left = cc.info['extract_waveforms']['n_left']
        
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
        # reload because parralel jobs
        dataio = DataIO(dirname)
        cc = CatalogueConstructor(dataio=dataio, chan_grp=chan_grp)

    peak_sign = cc.info['peak_detector']['peak_sign']
    dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['extract_waveforms']['n_left']
    n_right = cc.info['extract_waveforms']['n_right']
    peak_width = n_right - n_left
    nb_channel = cc.nb_channel
    
    if dense_mode:
        channel_adjacency = {c: np.arange(nb_channel) for c in range(nb_channel)}
    else:
        channel_adjacency = {}
        for c in range(nb_channel):
            nearest, = np.nonzero(cc.channel_distances[c, :] < adjacency_radius_um)
            channel_adjacency[c] = nearest

    #~ waveforms, wf_flat, peak_index = _get_sparse_waveforms_flatten(cc, dense_mode, label, channel_adjacency)
    waveforms = cc.get_cached_waveforms(label)
    centroid = cc.get_one_centroid(label)
    #~ print('label', label, waveforms.shape, centroid.shape)
    if not dense_mode:
        # TODO by sparsity level threhold and not radius
        if peak_sign == '-':
            extremum_channel = np.argmin(centroid[-n_left,:], axis=0)
        elif peak_sign == '+':
            extremum_channel = np.argmax(centroid[-n_left,:], axis=0)
        adjacency = channel_adjacency[extremum_channel]
        waveforms = waveforms.take(adjacency, axis=2)
    wf_flat = waveforms.swapaxes(1,2).reshape(waveforms.shape[0], -1)
    
    
    
    
    #~ pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components_local_pca, whiten=True)
    
    n_components = min(wf_flat.shape[1]-1, n_components_local_pca)
    pca =  sklearn.decomposition.TruncatedSVD(n_components=n_components)
    
    try:
        feats = pca.fit_transform(wf_flat)
    except ValueError:
        print('Erreur in diptest TruncatedSVD for label {}'.format(label))
        return None
        
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
    
    assert cc.some_waveforms is not None, 'run cc.cache_some_waveforms() first'
    
    peak_sign = cc.info['peak_detector']['peak_sign']
    dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['extract_waveforms']['n_left']
    n_right = cc.info['extract_waveforms']['n_right']
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
    
    if n_jobs < 0:
        n_jobs = joblib.cpu_count() + 1 - n_jobs
    
    if n_jobs > 1:
        n_jobs = min(n_jobs, len( cc.positive_cluster_labels))
    
    pvals = Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                    delayed(_compute_one_dip_test)(cc2, cc.dataio.dirname, cc.chan_grp, label, n_components_local_pca, adjacency_radius_um)
                    for label in cc.positive_cluster_labels)
    
    pvals = np.array(pvals)
    inds, = np.nonzero(pvals<pval_thresh)
    splitable_labels = cc.positive_cluster_labels[inds]
    #~ print('splitable_labels', splitable_labels)
    
    #~ for label in splitable_labels:
    #~ new_labels = []
    for ind in inds:
        label = cc.positive_cluster_labels[ind]
        pval = pvals[ind]
        
        # do not use cache to get ALL waveform
        waveforms, wf_flat, peak_index = _get_sparse_waveforms_flatten(cc, dense_mode, label, channel_adjacency)
        
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
            import matplotlib.pyplot as plt
            fig, ax= plt.subplots()
            
            ax.plot(wf_flat.T, color='m', alpha=0.5)
            
            ax.plot(np.median(wf_flat, axis=0), color='k')
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
                    #~ new_labels.append(new_label)
                    
                    m += 1
            
            cc.pop_labels_from_cluster([label])
            
            #~ m += np.max(unique_sub_labels) + 1
        

            #~ if True:
            #~ if False:
            if debug_plot:
                #~ print('label', label,'pval', pval, pval<pval_thresh)
                #~ print('label', label,'pval', pval, pval<pval_thresh)
                import matplotlib.pyplot as plt
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
    n_left = cc.info['extract_waveforms']['n_left']
    peak_sign = cc.info['peak_detector']['peak_sign']
    
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
                import matplotlib.pyplot as plt
                n_left = cc.info['extract_waveforms']['n_left']
                n_right = cc.info['extract_waveforms']['n_right']
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
                        #~ maximum_shift=2,
                        amplitude_factor_thresh = 0.3,
                        n_shift=2,
                        #~ sparse_thresh=1.5,
                        sparse_thresh=3,
                        quantile_limit=0.9,
                        recursive_loop=False,
                        
        ):
    cc = catalogueconstructor
    assert cc.some_waveforms is not None, 'run cc.cache_some_waveforms() first'
    
    peak_sign = cc.info['peak_detector']['peak_sign']
    #~ dense_mode = cc.info['mode'] == 'dense'
    n_left = cc.info['extract_waveforms']['n_left']
    n_right = cc.info['extract_waveforms']['n_right']
    peak_width = n_right - n_left
    threshold = cc.info['peak_detector']['relative_threshold']
    
    while True:
        
        #~ labels = cc.positive_cluster_labels.copy()
        
        keep = cc.cluster_labels>=0
        labels = cc.clusters[keep]['cluster_label'].copy()
        
        centroids = cc.centroids_median[keep, :, :].copy()
        share_channel_mask = compute_shared_channel_mask(centroids, cc.mode,  sparse_thresh)
        
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

            #~ t1 = time.perf_counter()
            ind1 = cc.index_of_label(k1)
            extremum_amplitude1 = np.abs(cc.clusters[ind1]['extremum_amplitude'])
            centroid1 = cc.get_one_centroid(k1)
            waveforms1 = cc.get_cached_waveforms(k1)
            #~ t2 = time.perf_counter()
            #~ print('get_cached_waveforms(k1)', (t2-t1)*1000.)

            
            for j in range(i+1, n):
                k2 = labels[j]
                if k2 == -1:
                    # this can have been removed yet
                    continue
                
                #~ print(k1, k2)
                #~ print('  k2', k2)
                
                if not share_channel_mask[i, j]:
                    #~ print('skip')
                    continue
                


                
                
                #~ t1 = time.perf_counter()
                ind2 = cc.index_of_label(k2)
                extremum_amplitude2 = np.abs(cc.clusters[ind2]['extremum_amplitude'])
                centroid2 = cc.get_one_centroid(k2)
                waveforms2 = cc.get_cached_waveforms(k2)
                #~ t2 = time.perf_counter()
                #~ print('get_cached_waveforms(k2)', (t2-t1)*1000.)
                
                
                #~ t2 = time.perf_counter()
                thresh = max(extremum_amplitude1, extremum_amplitude2) * amplitude_factor_thresh
                thresh = max(thresh, auto_merge_threshold)
                should_merge = equal_template_with_distance(centroid1, centroid2, thresh=thresh, n_shift=n_shift)
                if not should_merge:
                    continue
                #~ t2 = time.perf_counter()
                #~ print('equal_template_with_distance', (t2-t1)*1000.)
                #~ print('should_merge', should_merge)
                
                #~ t1 = time.perf_counter()
                do_merge = equal_template_with_distrib_overlap(centroid1, waveforms1, centroid2, waveforms2,
                                        n_shift=n_shift, quantile_limit=quantile_limit, sparse_thresh=sparse_thresh)
                #~ t2 = time.perf_counter()
                #~ print('equal_template_with_distrib_overlap', (t2-t1)*1000.)
                #~ print('do_merge', do_merge)
                
                #~ if should_merge != do_merge:
                    #~ # this is for debug
                    #~ print('debug merge', k1,k2)
                    #~ do_merge = equal_template_with_distrib_overlap(centroid1, waveforms1, centroid2, waveforms2,
                                        #~ n_shift=n_shift, quantile_limit=quantile_limit, sparse_thresh=sparse_thresh, debug=True)
                    #~ print('*' *50)
                
                #~ do_merge = False
                
                
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
                    if debug_plot:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.plot(centroid1.T.flatten(), color='m')
                        ax.plot(centroid2.T.flatten(), color='c')
                        ax.set_title('merge '+str(k1)+' '+str(k2))

                        fig, ax = plt.subplots()
                        ax.plot(waveforms1.swapaxes(1,2).reshape(waveforms1.shape[0], -1).T, color='m', alpha=0.2)
                        ax.plot(waveforms2.swapaxes(1,2).reshape(waveforms2.shape[0], -1).T, color='c', alpha=0.2)
                        ax.plot(centroid1.T.flatten(), color='k')
                        ax.plot(centroid2.T.flatten(), color='k')
                        ax.set_title('merge '+str(k1)+' '+str(k2))

                        #~ fig, ax = plt.subplots()
                        #~ ax.plot(np.median(waveforms1, axis=0).T.flatten(), color='m')
                        #~ ax.plot(np.median(waveforms2, axis=0).T.flatten(), color='c')

                        plt.show()
                    
                    
                    mask = cc.all_peaks['cluster_label'] == k2
                    cc.all_peaks['cluster_label'][mask] = k1
                    
                    new_centroids.append(k1)
                    pop_from_cluster.append(k2)
                    labels[j] = -1
                    nb_merge += 1
                    

        
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
        if recursive_loop:
            if nb_merge == 0:
                break
        else:
            # one loop only
            break


def trash_low_extremum(cc, min_extremum_amplitude=None):
    if min_extremum_amplitude is None:
        threshold = cc.info['peak_detector']['relative_threshold']
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




def _shift_add(wf0, wf1, shift):
    wf_add = wf0.copy()
    if shift == 0:
        wf_add += wf1
    elif shift>0:
        wf_add[shift:, :] += wf1[:-shift, :]
    else:
        wf_add[:-shift, :] += wf1[shift:, :]
    
    return wf_add

def remove_overlap(cc, thresh_mad=6):
    print(cc)

    print(cc.clusters['cluster_label'])
    
    
    # TODO change this masking suff
    print(cc.centroids_mad.shape)
    #~ keep,  = np.nonzero(cc.clusters['cluster_label']>=0)
    keep = cc.clusters['cluster_label']>=0
    print(keep)
    mads = cc.centroids_mad[keep, :, :]
    print(mads.shape)
    centers = cc.centroids_median[keep, :, :]
    
    max_mad = np.max(mads, axis=(1,2))
    print(max_mad)

    
    n_left = cc.info['extract_waveforms']['n_left']
    n_right = cc.info['extract_waveforms']['n_right']
    width = n_right - n_left
    
    for i, label in enumerate(cc.positive_cluster_labels):
        
        
        #~ mad = cc.get_one_centroid(label, metric='mad', long=False)
        mad = mads[i, :, :]
        print('i', i, 'mad', np.max(mad))
        if np.max(mad) < thresh_mad:
            continue
        
        #~ continue # DEBUG
        
        center = centers[i]
        
        
        pairs = []
        distances = []
        for j, label0 in enumerate(cc.positive_cluster_labels):
            if i == j:
                continue
            for k, label1 in enumerate(cc.positive_cluster_labels):
                if j == k or i==k:
                    continue
                
                print('j, k', j, k)
                
                center0 = centers[j]
                center1 = centers[k]
                #~ center_add = center0 + center1
                
                for shift in range(-width//2, width//2):
                    
                    center_add = _shift_add(center0, center1, shift)
                    
                    dist = np.sum((center - center_add)**2)
                    
                
                    #~ import matplotlib.pyplot as plt
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(center.T.flatten(), color='b')
                    #~ ax.plot(center_add.T.flatten(), color='c', ls='--')
                    #~ ax.set_title(f'{label} = {label0} + {label1} {shift}')
                    #~ plt.show()                
                
                
                    pairs.append((j,k, shift))
                    distances.append(dist)
        
        if len(distances) > 0:
            #~ print(distances)
            #~ print(pairs)
            ind_min = np.argmin(distances)
            
            j, k, shift = pairs[ind_min]
            print('best', j, k)
            center0 = centers[j]
            center1 = centers[k]
            #~ center_add = center0 + center1
            center_add = _shift_add(center0, center1, shift)
            
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(nrows=2, sharex=True)
            ax = axs[0]
            ax.plot(center.T.flatten(), color='b')
            ax.plot(center0.T.flatten(), color='r')
            ax.plot(center1.T.flatten(), color='g')
            ax.plot(center_add.T.flatten(), color='c', ls='--')
            ax = axs[1]
            ax.plot(center_add.T.flatten() - center.T.flatten())
            label0 = cc.positive_cluster_labels[j]
            label1 = cc.positive_cluster_labels[k]
            ax.set_title(f'{label} = {label0} + {label1} shift{shift}')
            plt.show()
            
            
            #~ fig, axs = plt.subplots(nrows=2, sharex=True)
            #~ axs[0].plot(center.T.flatten())
            #~ axs[1].plot(mad.T.flatten())
            #~ axs[0].set_title(f'{label}')
            #~ plt.show()
        
        
        
        
        

    
