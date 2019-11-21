try:
    import numba
    from numba import jit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

import numpy as np

@jit(parallel=True)
def numba_loop_sparse_dist(waveform, centers,  mask):
    nb_clus, width, nb_chan = centers.shape
    
    rms_waveform_channel = np.sum(waveform**2, axis=0)#.astype('float32')
    waveform_distance = np.zeros((nb_clus,), dtype=np.float32)
    
    for clus in range(nb_clus):
        sum = 0
        for c in range(nb_chan):
            if mask[clus, c]:
                for s in range(width):
                    d = waveform[s, c] - centers[clus, s, c]
                    sum += d*d
            else:
                sum +=rms_waveform_channel[c]
        waveform_distance[clus] = sum
    
    return waveform_distance
    
    
@jit(parallel=True)
def numba_loop_sparse_dist_with_geometry(waveform, centers,  mask, possibles_cluster_idx, channels):
    nb_total_clus, width, nb_chan = centers.shape
    nb_clus = possibles_cluster_idx.size
    
    rms_waveform_channel = np.sum(waveform**2, axis=0)#.astype('float32')
    waveform_distance = np.zeros((nb_clus,), dtype=np.float32)
    
    for clus, cluster_idx in enumerate(possibles_cluster_idx):
        sum = 0
        #~ for c in range(nb_chan):
        for c in channels:
            if mask[cluster_idx, c]:
                for s in range(width):
                    d = waveform[s, c] - centers[cluster_idx, s, c]
                    sum += d*d
            else:
                sum +=rms_waveform_channel[c]
        waveform_distance[clus] = sum
    
    return waveform_distance


@jit(parallel=True)
def numba_explore_shifts(long_waveform, one_center,  one_mask, maximum_jitter_shift):
    width, nb_chan = one_center.shape
    n = maximum_jitter_shift*2 +1
    
    all_dist = np.zeros((n, ), dtype=np.float32)
    
    for shift in range(n):
        # waveform = long_waveform[shift:shift+width]
        sum = 0
        for c in range(nb_chan):
            if one_mask[c]:
                for s in range(width):
                    d = long_waveform[shift+s, c] - one_center[s, c]
                    sum += d*d
            #~ else:
                #~ for pos in range(width):
                    #~ d = long_waveform[shift+s, c]
                    #~ sum += d*d
        all_dist[shift] = sum
    
    return all_dist


@jit(parallel=True)
def numba_get_mask_spatiotemporal_peaks(sigs, n_span, thresh, peak_sign, neighbours):
    sig_center = sigs[n_span:-n_span, :]

    if peak_sign == '+':
        mask_peaks = sig_center>thresh
        for chan in range(sigs.shape[1]):
            for neighbour in neighbours[chan, :]:
                for i in range(n_span):
                    if chan != neighbour:
                        mask_peaks[:, chan] &= sig_center[:, chan] >= sig_center[:, neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan] > sigs[i:i+sig_center.shape[0], neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan]>=sigs[n_span+i+1:n_span+i+1+sig_center.shape[0], neighbour]
        
    elif peak_sign == '-':
        mask_peaks = sig_center<-thresh
        for chan in range(sigs.shape[1]):
            #~ print('chan', chan, 'neigh', neighbours[chan, :])
            for neighbour in neighbours[chan, :]:
                for i in range(n_span):
                    if chan != neighbour:
                        mask_peaks[:, chan] &= sig_center[:, chan] <= sig_center[:, neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan] < sigs[i:i+sig_center.shape[0], neighbour]
                    mask_peaks[:, chan] &= sig_center[:, chan]<=sigs[n_span+i+1:n_span+i+1+sig_center.shape[0], neighbour]
    
    return mask_peaks




