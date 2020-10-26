import numpy as np
import scipy.linalg
import joblib

from .tools import median_mad




def extract_chunks(signals, left_sample_indexes, width, 
                channel_indexes=None, channel_adjacency=None, 
                peak_channel_indexes=None, chunks=None):
    """
    This cut small chunks on signals and return concatenate them.
    This use numpy.array for input/output.
    
    
    Arguments
    ---------------
    signals: np.ndarray 
        shape (nb_campleXnb_channel)
    left_sample_indexes: np.ndarray
        sample postion of the first sample
    width: int
        Width in sample of chunks.
    channel_indexes: 
        Force a limited number of channels. If None then all channel.
    
    channel_adjacency: dict
        dict of channel adjacency
        If not None, then the wavzform extraction is sprase given adjency.
        In that case peak_channel_indexes must be provide and channel_indexes 
        must be None.
    peak_channel_indexes: np.array None
        Position of the peak on channel space used only when channel_adjacency not None.
        
    Returns
    -----------
    chunks : np.ndarray
        shape = (left_sample_indexes.size, width, signals.shape[1], )
    
    """
    if channel_adjacency is not None:
        assert channel_indexes is None
        assert peak_channel_indexes is not None, 'For sparse eaxtraction peak_channel_indexes must be provide'
    
    if chunks is None:
        if channel_indexes is None:
            chunks = np.empty((left_sample_indexes.size, width, signals.shape[1]), dtype = signals.dtype)
        else:
            chunks = np.empty((left_sample_indexes.size, width, len(channel_indexes)), dtype = signals.dtype)
        
    keep = (left_sample_indexes>=0) & (left_sample_indexes<(signals.shape[0] - width))
    left_sample_indexes2 = left_sample_indexes[keep]
    if peak_channel_indexes is not None:
        peak_channel_indexes2 = peak_channel_indexes[keep]
    
    if channel_indexes is None and channel_adjacency is None:
        # all channels
        for i, ind in enumerate(left_sample_indexes2):
            chunks[i,:,:] = signals[ind:ind+width,:]
    elif channel_indexes is not None:
        for i, ind in enumerate(left_sample_indexes2):
            chunks[i,:,:] = signals[ind:ind+width,:][:, channel_indexes]
            
    elif channel_adjacency is not None:
        # sparse
        assert peak_channel_indexes is not None
        for i, ind in enumerate(left_sample_indexes2):
            chan = peak_channel_indexes2[i]
            chans = channel_adjacency[chan]
            chunks[i,:,:][:, chans] = signals[ind:ind+width,:][:, chans]

    return chunks



def equal_template_with_distance(centroid0, centroid1, thresh=2.0, n_shift = 2):
    """
    Test template equality based on distance between centroids.
    Do this for some jitter (n_shift)
    
    """
    wf0 = centroid0[n_shift:-n_shift, :]
    
    equal = False
    for shift in range(n_shift*2+1):
        wf1 = centroid1[shift:wf0.shape[0]+shift, :]
        
        d = np.max(np.abs(wf1 - wf0))

        #~ print('shift', shift, 'd', d)

        #~ import matplotlib.pyplot as plt
        #~ fig, ax = plt.subplots()
        #~ ax.plot(centroid0.T.flatten())
        #~ ax.plot(centroid1.T.flatten())
        #~ ax.set_title(f'shift {shift} thresh {thresh:0.2f} d {d:0.2f} merge ' + str(d<thresh))
        #~ plt.show()
        
        if d<thresh:
            equal = True
                
            #~ import matplotlib.pyplot as plt
            #~ fig, ax = plt.subplots()
            #~ ax.plot(centroid0.T.flatten())
            #~ ax.plot(centroid1.T.flatten())
            #~ ax.set_title(f'shift {shift} thresh {thresh:0.2f} d {d:0.2f} merge ' + str(d<thresh))
            #~ plt.show()
            
            break
    
    return equal



def equal_template_with_distrib_overlap(centroid0, waveforms0, centroid1, waveforms1,
                    n_shift = 2, quantile_limit=0.8, sparse_thresh=1.5,
                    debug=False
                    ):
    """
    Test template equality with checking distribution overlap.
    Do this for some jitter (n_shift)
    
    """
    mask0 = np.any(np.abs(centroid0) > sparse_thresh, axis=0)
    mask1 = np.any(np.abs(centroid1) > sparse_thresh, axis=0)
    mask = mask1 | mask0
    
    #~ print(centroid0.shape)

    #~ centroid0_ = centroid0[np.newaxis, n_shift:-n_shift, :]
    
    centroid0_ = centroid0[n_shift:-n_shift, :]
    centroid1_ = centroid1[n_shift:-n_shift, :]
    vector_0_1 = (centroid1_ - centroid0_)
    vector_0_1 =vector_0_1[:, mask]
    vector_0_1 /= np.sum(vector_0_1**2)
    

    
    wfs0 = waveforms0[:, n_shift:-n_shift, :].copy()
    #~ dist_0_to_0 = np.sum((centroid0_[:,:, mask] - wfs0[:,:, mask]) ** 2, axis=(1,2))
    #~ print(dist_0_to_0)
    
    #~ limit = np.quantile(dist_0_to_0, quantile_limit)

    equal = False
    for shift in range(n_shift*2+1):
        #~ print('shift', shift)
        wfs1 = waveforms1[:, shift:centroid0_.shape[0]+shift, :].copy()
        
        inner_sp = np.sum((wfs0[:,:,mask] - centroid0_[np.newaxis,:,:][:,:,mask]) * vector_0_1[np.newaxis,:,:], axis=(1,2))
        cross_sp = np.sum((wfs1[:,:,mask] - centroid0_[np.newaxis,:,:][:,:,mask]) * vector_0_1[np.newaxis,:,:], axis=(1,2))
        
        l0 = np.quantile(inner_sp, quantile_limit)
        l1 = np.quantile(cross_sp, 1 - quantile_limit)
        
        equal = l0 >= l1
        

        
        #~ if np.median(dist_1_to_0)<limit:
        #~ if True:
        #~ if equal:
        if debug:
                
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            count, bins = np.histogram(inner_sp, bins=100)
            ax.plot(bins[:-1], count, color='g')
            count, bins = np.histogram(cross_sp, bins=100)
            ax.plot(bins[:-1], count, color='r')
            ax.axvline(l0)
            ax.axvline(l1)
            ax.set_title(f'merge {equal} shift {shift}')

            
            fig, ax = plt.subplots()
            ax.plot(centroid0.T.flatten())
            ax.plot(centroid1.T.flatten())
            ax.set_title(f'shift {shift}')
            
            plt.show()
            
        
        if equal:
            break
    
    return equal



#~ def nearest_neighbor_similarity(cc):
    
    #~ labels = cc.positive_cluster_labels
    #~ n = len(labels)
    
    #~ snn = np.zeros((n,n), dtype='float32')
    
    #~ for i, label0 in enumerate(labels):
        #~ for j, label1 in enumerate(labels):
            #~ if j<i:
                #~ continue
            
            #~ waveforms0 = cc.get_cached_waveforms(label0)
            #~ waveforms1 = cc.get_cached_waveforms(label1)
            #~ snn[i, j] = one_nearest_neighbor_similarity(waveforms0, waveforms1, n_neighbors=10)
            
            #~ print('i', i, 'j', j, 'snn',  snn[i, j])
            
    
    #~ return snn


#~ def one_nearest_neighbor_similarity(waveforms0, waveforms1, n_neighbors=10):
        
    #~ from sklearn.neighbors import NearestNeighbors
    
    #~ n0 = waveforms0.shape[0]
    
    #~ waveforms = np.concatenate((waveforms0, waveforms1), axis=0)
    #~ n = waveforms.shape[0]
    #~ wf_flat = waveforms.reshape(n,-1)
    
    #~ nn = NearestNeighbors(n_neighbors=n_neighbors)
    #~ nn.fit(wf_flat)
    #~ distances, indices = nn.kneighbors(wf_flat)
    #~ nn0 = np.sum((indices[:n0] < n0).astype(int), axis=1)
    #~ nn1 = np.sum((indices[n0:] >= n0).astype(int), axis=1)
    #~ one_snn = np.mean(n_neighbors - np.concatenate((nn0, nn1))) / n_neighbors
    
    #~ return one_snn
    



#~ def get_normalized_centroids(centroids, peak_sign, sparse_thresh=None):
    #~ nb_clus, width, nb_chan = centroids.shape
    
    #~ normalized_centroids = np.zeros_like(centroids)
    
    #~ for cluster_idx in range(nb_clus):
        #~ centroid = centroids[cluster_idx, :, :]
        #~ centroid_normalized = centroid.copy()
        
        #~ if sparse_thresh is not None:
            #~ mask = np.any(np.abs(centroid) > sparse_thresh, axis=0)
            #~ centroid_normalized[:, ~mask] = 0
            
        #~ if peak_sign == '-':
            #~ centroid_normalized /= np.abs(np.min(centroid_normalized))
        #~ elif peak_sign == '+':
            #~ centroid_normalized /= np.abs(np.max(centroid_normalized))
        #~ centroid_normalized /= np.sum(centroid_normalized**2)
        #~ normalized_centroids[cluster_idx, :] = centroid_normalized

    #~ return normalized_centroids


def compute_sparse_mask(centroids, mode, method='thresh', thresh=None, nbest=None):
    nb_clus, width, nb_chan = centroids.shape
    
    sparse_mask_level = np.ones((nb_clus, nb_chan), dtype='bool')
    
    if mode == 'sparse':
        if method == 'thresh':
            for cluster_idx in range(nb_clus):
                centroid = centroids[cluster_idx, :, :]
                mask = np.any(np.abs(centroid) > thresh, axis=0)
                sparse_mask_level[cluster_idx, :] = mask
        elif method == 'nbest':
            for cluster_idx in range(nb_clus):
                centroid = centroids[cluster_idx, :, :]
                amplitudes = np.max(np.abs(centroid), axis=0)
                order = np.argsort(amplitudes)[::-1]
                best_channels = order[:nbest]
                sparse_mask_level[cluster_idx, :] = False
                sparse_mask_level[cluster_idx, :][best_channels] = True
                
    
    return sparse_mask_level

def compute_shared_channel_mask(centroids, mode, sparse_thresh):
    n = centroids.shape[0]
    
    share_channel_mask =np.ones((n, n), dtype='bool')
    
    if mode == 'dense':
        return share_channel_mask

    sparse_mask = compute_sparse_mask(centroids, mode, method='thresh', thresh=sparse_thresh)
    
    for cluster_idx0 in range(n):
        
        #~ print('cluster_idx0', cluster_idx0)
        
        centroid0 = centroids[cluster_idx0, :, :]
        
        mask0 = sparse_mask[cluster_idx0]
        n0 = np.sum(mask0)
        
        for cluster_idx1 in range(n):
            mask1 = sparse_mask[cluster_idx1]
            nshared = np.sum(mask0 & mask1)
            
            n1 = np.sum(mask1)
            
            m = min(n0, n1)
            if m <= 4:
                if nshared == 0:
                    channel_shared = False
                else:
                    channel_shared = True
            else:
                if nshared > m * 0.5:
                    channel_shared = True
                else:
                    channel_shared = False
            
            #~ print('channel_shared', channel_shared)
            #~ fig, ax = plt.subplots()
            #~ ax.plot(centroid0.T.flatten())
            #~ centroid1 = centroids[cluster_idx1, :, :]
            #~ ax.plot(centroid1.T.flatten())
            #~ ax.set_title(f'{cluster_idx0} {cluster_idx1} channel_shared{channel_shared}')
            #~ plt.show()
            
            
            share_channel_mask[cluster_idx0, cluster_idx1] = channel_shared

    return share_channel_mask




def compute_projection(centroids, sparse_mask_level1):

    n = centroids.shape[0]
    #~ print('n', n)
    
    #~ flat_shape = n, centroids.shape[1] * centroids.shape[2]
    projections = np.zeros(centroids.shape, dtype='float32')
    
    neighbors = {}
    
    for cluster_idx0 in range(n):
    
        # only centroids / channel that are on sparse mask
        chan_mask = sparse_mask_level1[cluster_idx0, :]
        
        # case1 sharing chan_mask
        #~ nover = np.sum(sparse_mask_level1[:, chan_mask], axis=1)
        #~ clus_mask = (nover > 0)
        
        # case2 5 nearest template
        #~ distances = np.sum((centroids - centroids[[cluster_idx0], :, :])**2, axis=(1,2))
        #~ order = np.argsort(distances)
        #~ nearest = order[:8]
        #~ clus_mask = np.zeros(n, dtype='bool')
        #~ clus_mask[nearest] = True
        
        # case3 all cluster
        clus_mask = np.ones(n, dtype='bool')
        
        # case 4 : sharing chan_mask + N nearest
        #~ nover = np.sum(sparse_mask_level1[:, chan_mask], axis=1)
        #~ sharing_mask = (nover > 0)
        #~ distances = np.sum((centroids - centroids[[cluster_idx0], :, :])**2, axis=(1,2))
        #~ distances[~sharing_mask] = np.inf
        #~ order = np.argsort(distances)
        #~ nearest = order[:8]
        #~ clus_mask = np.zeros(n, dtype='bool')
        #~ clus_mask[nearest] = True
        
        local_indexes0, = np.nonzero(clus_mask)
        local_idx0 = np.nonzero(local_indexes0 == cluster_idx0)[0][0]
            
        local_nclus = np.sum(clus_mask)
        local_chan = np.sum(chan_mask)
        flat_centroids = centroids[clus_mask, :, :][:, :, chan_mask].reshape(local_nclus, -1).T.copy()
        #~ print('local_nclus', local_nclus, 'local_chan', local_chan)
        
        
        flat_centroid0 = flat_centroids[:, local_idx0]
        #~ flat_centroid0 = centroids[cluster_idx0, :, :][:, chan_mask].flatten()
        
        other_mask = np.ones(local_nclus, dtype='bool')
        other_mask[local_idx0] = False
        
        if np.sum(other_mask) > 0:
            other_centroids = flat_centroids[:, other_mask]

            ind_min = np.argmin(np.sum((other_centroids - flat_centroid0[:, np.newaxis])**2, axis=0))
            other_select = [ind_min]
            
            
            while True:
                if len(other_select) == 1:
                    centroid0_proj = other_centroids[:, other_select[0]]
                else:
                    # This find point in orthogonal hyperplan to the centroid
                    centerred_other_centroids = other_centroids[:, other_select].copy()
                    shift = -centerred_other_centroids[:, 0]
                    centerred_other_centroids += shift[:, np.newaxis]
                    centerred_other_centroids = centerred_other_centroids[:, 1:]
                    u,s,vh = scipy.linalg.svd(centerred_other_centroids, full_matrices=False)
                    centroid0_proj = u @ u.T @ (flat_centroid0 + shift) - shift
                
                local_projector = centroid0_proj - flat_centroid0
                local_projector /= np.sum(local_projector**2)                

                # test if noise (0,0,0, ...) must be in the hyperplane
                noise_feat = (- flat_centroid0).T @ local_projector
                if np.abs(noise_feat) < 1.:
                    centerred_other_centroids = other_centroids[:, other_select].copy()
                    u,s,vh = scipy.linalg.svd(centerred_other_centroids, full_matrices=False)
                    centroid0_proj = u @ u.T @ (flat_centroid0)
                    local_projector = centroid0_proj - flat_centroid0
                    local_projector /= np.sum(local_projector**2)
                    #~ print('cluster_idx0', cluster_idx0, 'WITH noise')
                #~ else:
                    #~ print('cluster_idx0', cluster_idx0, 'WITHOUT noise')
                    
                other_feat = (other_centroids - flat_centroid0[:, np.newaxis]).T @ local_projector
                
                ind,  = np.nonzero(np.abs(other_feat) < 1.)
                
                if ind.size == 0:
                    break
                else:
                    # TODO : try to add them one by one and not sevral at each loop
                    not_in = np.ones(other_feat.size, dtype='bool')
                    not_in[other_select] = False
                    too_small = (np.abs(other_feat) < 1.)
                    others_candidate, = np.nonzero(not_in & too_small)
                    if others_candidate.size ==0:
                        break
                    smallest_ind = np.argmin(np.abs(other_feat[others_candidate]))
                    other_select.append(others_candidate[smallest_ind])
                    #~ print(cluster_idx0, 'other_select', other_select)
                
            neighbors[cluster_idx0] = np.nonzero(other_mask)[0][other_select]
            ortho_complement = local_projector
        else:
            # alone one theses channels = make projection with noise
            ortho_complement = 0 - flat_centroid0
            ortho_complement /= np.sum(ortho_complement**2)                
            
            neighbors[cluster_idx0] = {}
        
        #~ print('neighbors[cluster_idx0]', neighbors[cluster_idx0])

        
        
        projections[cluster_idx0, :, :][:, chan_mask] = ortho_complement.reshape(centroids.shape[1], local_chan)

    
    return projections, neighbors




def compute_boundaries(cc, centroids, sparse_mask_level1, projections, neighbors, plot_debug=False):
    #~ n = len(cluster_labels)
    n = centroids.shape[0]

    keep = cc.cluster_labels>=0
    cluster_labels = cc.clusters[keep]['cluster_label'].copy()

    
    scalar_products = np.zeros((n+1, n+1), dtype=object)
    boundaries = np.zeros((n, 4), dtype='float32')
    
    # compute all scalar product with projection
    for cluster_idx0, label0 in enumerate(cluster_labels):
        chan_mask = sparse_mask_level1[cluster_idx0, :]
        projector =projections[cluster_idx0, :][:, chan_mask].flatten()
        
        wf0 = cc.get_cached_waveforms(label0)
        flat_centroid0 = centroids[cluster_idx0, :, :][:, chan_mask].flatten()
        #~ print('cluster_idx0',cluster_idx0, wf0.shape)
        
        wf0 = wf0[:, :, chan_mask].copy()
        flat_wf0 = wf0.reshape(wf0.shape[0], -1)
        feat_wf0 = (flat_wf0 - flat_centroid0) @ projector
        feat_centroid0 = (flat_centroid0 - flat_centroid0) @ projector
        
        scalar_products[cluster_idx0, cluster_idx0] = feat_wf0
        
        for cluster_idx1, label1 in enumerate(cluster_labels):
            if cluster_idx0 == cluster_idx1:
                continue
            
            centroid1 = centroids[cluster_idx1, :, :][:, chan_mask]
            wf1 = cc.get_cached_waveforms(label1)
            wf1 = wf1[:, :, chan_mask].copy()
            flat_wf1 = wf1.reshape(wf1.shape[0], -1) 
            feat_centroid1 = (centroid1.flatten() - flat_centroid0) @ projector
            feat_wf1 = (flat_wf1- flat_centroid0) @ projector
            
            scalar_products[cluster_idx0, cluster_idx1] = feat_wf1
        
        noise = cc.some_noise_snippet
        noise = noise[:, :, chan_mask].copy()
        flat_noise = noise.reshape(noise.shape[0], -1)
        feat_noise = (flat_noise - flat_centroid0) @ projector
        scalar_products[cluster_idx0, -1] = feat_noise
    
    # find boudaries
    for cluster_idx0, label0 in enumerate(cluster_labels):
        inner_sp = scalar_products[cluster_idx0, cluster_idx0]
        med, mad = median_mad(inner_sp)
        
        mad_factor = 6
        low = med - mad_factor * mad
        initial_low = low
        high = med + mad_factor * mad
        initial_high = high


        # optimze boudaries with accuracy
        high_clust = []
        high_lim = []
        low_clust = []
        low_lim = []
        for cluster_idx1, label1 in enumerate(cluster_labels):
            # select dangerous cluster
            if cluster_idx1 == cluster_idx0:
                continue
            cross_sp = scalar_products[cluster_idx0, cluster_idx1]
            med, mad = median_mad(cross_sp)
            if med > high and (med - mad_factor * mad) < high :
                high_clust.append(cluster_idx1)
                high_lim.append(med - mad_factor * mad)
            if med < low and (med + mad_factor * mad) > low:
                low_clust.append(cluster_idx1)
                low_lim.append(med + mad_factor * mad)

        noise_sp = scalar_products[cluster_idx0, -1]
        med, mad = median_mad(noise_sp)
        if med > high and (med - mad_factor * mad) < high :
            high_clust.append(-1)
            high_lim.append(med - mad_factor * mad)
        # TODO if noise in low limits
        #~ print()
        #~ print('initial_low', initial_low)
        #~ print('initial_high', initial_high)
        #~ print('high_clust', high_clust, 'high_lim', high_lim)
        #~ print('low_clust', low_clust, 'low_lim', low_lim)

        if len(high_clust) > 0:
            l0 = min(high_lim)
            l1 = initial_high
            step = (l1-l0)/20.
            
            all_sp = np.concatenate([scalar_products[cluster_idx0, idx1] for idx1 in high_clust])
            limits = np.arange(l0, l1, step) 
            accuracies = []
            for l in limits:
                tp = scalar_products[cluster_idx0, cluster_idx0].size
                fn = np.sum(scalar_products[cluster_idx0, cluster_idx0] > l)
                fp = np.sum(all_sp < l)
                accuracy = tp / (tp + fn + fp)
                accuracies.append(accuracy)
            best_lim = limits[np.argmax(accuracies)]
            boundaries[cluster_idx0, 1] = min(best_lim, 0.5)
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(limits, accuracies)
            #~ ax.axvline(best_lim)
            #~ ax.set_title(f'high {cluster_idx0}')
            #~ plt.show()
            
        else:
            boundaries[cluster_idx0, 1] = min(initial_high, 0.5)
        
        if len(low_clust) > 0:
            l1 = max(low_lim)
            l0 = initial_low
            step = (l1-l0)/20.
            
            all_sp = np.concatenate([scalar_products[cluster_idx0, idx1] for idx1 in low_clust])
            limits = np.arange(l0, l1, step) 
            accuracies = []
            for l in limits:
                tp = scalar_products[cluster_idx0, cluster_idx0].size
                fn = np.sum(scalar_products[cluster_idx0, cluster_idx0] <l)
                fp = np.sum(all_sp > l)
                accuracy = tp / (tp + fn + fp)
                accuracies.append(accuracy)
            best_lim = limits[np.argmax(accuracies)]
            boundaries[cluster_idx0, 0] = max(best_lim, -0.5)
        
        else:
            boundaries[cluster_idx0, 0] = max(initial_low, -0.5)

        
        boundaries[cluster_idx0, 2] = max(initial_low, -0.5)
        boundaries[cluster_idx0, 3] = min(initial_high, 0.5)                    
    
    #~ if plot_debug:
        #~ import matplotlib.pyplot as plt
        
        
    
    
    return boundaries, scalar_products



