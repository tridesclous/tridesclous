import numpy as np
import joblib





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



def equal_template_old(centroid0, centroid1, thresh=2.0, n_shift = 2):
    """
    Check if two centroid are mor or less equal for some jitter.
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



def equal_template(centroid0, waveforms0, centroid1, waveforms1, n_shift = 2, quantile_limit=0.8, sparse_thresh=1.5):
    mask0 = np.any(np.abs(centroid0) > sparse_thresh, axis=0)
    mask1 = np.any(np.abs(centroid1) > sparse_thresh, axis=0)
    mask = mask1 | mask0
    
    print(centroid0.shape)

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
        #~ dist_1_to_0 = np.sum((centroid0_[:,:, mask] - wfs1[:,:, mask]) ** 2, axis=(1,2))
        #~ print(dist_1_to_0)
        
        #~ print(centroid0_.shape)
        #~ print(wfs0.shape)
        #~ print(wfs1.shape)
        #~ exit()
        #~ print(vector_0_1.shape)

        #~ print(wfs0)
        #~ print(vector_0_1)
        inner_sp = np.sum((wfs0[:,:,mask] - centroid0_[np.newaxis,:,:][:,:,mask]) * vector_0_1[np.newaxis,:,:], axis=(1,2))
        cross_sp = np.sum((wfs1[:,:,mask] - centroid0_[np.newaxis,:,:][:,:,mask]) * vector_0_1[np.newaxis,:,:], axis=(1,2))
        #~ print(inner_sp)
        
        #~ print(cross_sp)
        
        
        l0 = np.quantile(inner_sp, quantile_limit)
        l1 = np.quantile(cross_sp, 1 - quantile_limit)
        
        equal = l0 >= l1
        
        #~ print(l0, l1, equal)
        

        

        
        #~ if np.median(dist_1_to_0)<limit:
        #~ if True:
        if equal:
                
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            count, bins = np.histogram(inner_sp, bins=100)
            ax.plot(bins[:-1], count, color='g')
            count, bins = np.histogram(cross_sp, bins=100)
            ax.plot(bins[:-1], count, color='r')
            ax.axvline(l0)
            ax.axvline(l1)
            ax.set_title(f'merge {equal} shift {shift}')

            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(centroid0.T.flatten())
            #~ ax.plot(centroid1.T.flatten())
            #~ ax.set_title(f'shift {shift}')
            
            #~ plt.show()
            
            #~ fig, ax = plt.subplots()
            #~ count, bins = np.histogram(dist_0_to_0, bins=100)
            #~ ax.plot(bins[:-1], count, color='g')
            #~ count, bins = np.histogram(dist_1_to_0, bins=100)
            #~ ax.plot(bins[:-1], count, color='r')
            #~ ax.axvline(limit)


            #~ fig, ax = plt.subplots()
            #~ ax.plot(wfs0.swapaxes(1,2).reshape(wfs0.shape[0], -1).T, color='m')
            #~ ax.plot(wfs1.swapaxes(1,2).reshape(wfs1.shape[0], -1).T, color='c')
            #~ ax.plot(centroid2.T.flatten())
            #~ ax.set_title('merge '+str(k1)+' '+str(k2))
            
            
            #~ plt.show()
        
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
    



def get_normalized_centroids(centroids, peak_sign, sparse_thresh=None):
    nb_clus, width, nb_chan = centroids.shape
    
    normalized_centroids = np.zeros_like(centroids)
    
    for cluster_idx in range(nb_clus):
        centroid = centroids[cluster_idx, :, :]
        centroid_normalized = centroid.copy()
        
        if sparse_thresh is not None:
            mask = np.any(np.abs(centroid) > sparse_thresh, axis=0)
            centroid_normalized[:, ~mask] = 0
            
        if peak_sign == '-':
            centroid_normalized /= np.abs(np.min(centroid_normalized))
        elif peak_sign == '+':
            centroid_normalized /= np.abs(np.max(centroid_normalized))
        centroid_normalized /= np.sum(centroid_normalized**2)
        normalized_centroids[cluster_idx, :] = centroid_normalized

    return normalized_centroids


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





