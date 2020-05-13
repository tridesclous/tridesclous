try:
    import numba
    from numba import jit, prange
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
def numba_loop_sparse_dist_with_geometry(waveform, centers,  possibles_cluster_idx, rms_waveform_channel,channel_adjacency):
    
    nb_total_clus, width, nb_chan = centers.shape
    nb_clus = possibles_cluster_idx.size
    
    #rms_waveform_channel = np.sum(waveform**2, axis=0)#.astype('float32')
    waveform_distance = np.zeros((nb_clus,), dtype=np.float32)
    
    for clus in prange(len(possibles_cluster_idx)):
        cluster_idx = possibles_cluster_idx[clus]
        sum = 0
        for c in channel_adjacency:
            #~ if mask[cluster_idx, c]:
            for s in range(width):
                d = waveform[s, c] - centers[cluster_idx, s, c]
                sum += d*d
            #~ else:
                #~ sum +=rms_waveform_channel[c]
        waveform_distance[clus] = sum
    
    return waveform_distance


@jit(parallel=True)
def numba_explore_shifts(long_waveform, one_center,  one_mask, maximum_jitter_shift):
    width, nb_chan = one_center.shape
    n = maximum_jitter_shift*2 +1
    
    all_dist = np.zeros((n, ), dtype=np.float32)
    
    for shift in prange(n):
        sum = 0
        for c in range(nb_chan):
            if one_mask[c]:
                for s in range(width):
                    d = long_waveform[shift+s, c] - one_center[s, c]
                    sum += d*d
        all_dist[shift] = sum
    
    return all_dist


@jit(parallel=True)
def peak_loop_plus(sigs, sig_center, mask_peaks, n_span, thresh, peak_sign, neighbours):
    for chan in prange(sig_center.shape[1]):
        for s in range(mask_peaks.shape[0]):
            if not mask_peaks[s, chan]:
                continue
            for neighbour in neighbours[chan, :]:
                if neighbour<0:
                    continue
                for i in range(n_span):
                    if chan != neighbour:
                        mask_peaks[s, chan] &= sig_center[s, chan] >= sig_center[s, neighbour]
                    mask_peaks[s, chan] &= sig_center[s, chan] > sigs[s+i, neighbour]
                    mask_peaks[s, chan] &= sig_center[s, chan]>=sigs[n_span+s+i+1, neighbour]
                    if not mask_peaks[s, chan]:
                        break
                if not mask_peaks[s, chan]:
                    break
    return mask_peaks


@jit(parallel=True)
def peak_loop_minus(sigs, sig_center, mask_peaks, n_span, thresh, peak_sign, neighbours):
    for chan in prange(sig_center.shape[1]):
        for s in range(mask_peaks.shape[0]):
            if not mask_peaks[s, chan]:
                continue
            for neighbour in neighbours[chan, :]:
                if neighbour<0:
                    continue
                for i in range(n_span):
                    if chan != neighbour:
                        mask_peaks[s, chan] &= sig_center[s, chan] <= sig_center[s, neighbour]
                    mask_peaks[s, chan] &= sig_center[s, chan] < sigs[s+i, neighbour]
                    mask_peaks[s, chan] &= sig_center[s, chan]<=sigs[n_span+s+i+1, neighbour]                        
                    if not mask_peaks[s, chan]:
                        break
                if not mask_peaks[s, chan]:
                    break
    return mask_peaks


def numba_get_mask_spatiotemporal_peaks(sigs, n_span, thresh, peak_sign, neighbours):
    sig_center = sigs[n_span:-n_span, :]

    if peak_sign == '+':
        mask_peaks = sig_center>thresh
        mask_peaks = peak_loop_plus(sigs, sig_center, mask_peaks, n_span, thresh, peak_sign, neighbours)

    elif peak_sign == '-':
        mask_peaks = sig_center<-thresh
        mask_peaks = peak_loop_minus(sigs, sig_center, mask_peaks, n_span, thresh, peak_sign, neighbours)

    return mask_peaks




