try:
    import numba
    from numba import jit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

import numpy as np

# OLD
@jit(nopython=True, parallel=True)
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



@jit(nopython=True, parallel=True)
def numba_sparse_scalar_product(fifo_residuals, left_ind, centers, projector, peak_chan_ind,
                        sparse_mask_level1, ):
    nb_clus, width, nb_chan = centers.shape
    
    scalar_product = np.zeros((nb_clus,), dtype=np.float32)
    
    for clus_idx in prange(nb_clus):
        if not sparse_mask_level1[clus_idx, peak_chan_ind]:
            scalar_product[clus_idx] = 10. # equivalent to np.inf
        else:
            sum_sp = 0.
            for chan in range(nb_chan):
                if sparse_mask_level1[clus_idx, chan]:
                    for s in range(width):
                        v = fifo_residuals[left_ind+s, chan]
                        ct = centers[clus_idx, s, chan]
                        w = projector[clus_idx, s, chan]
                        
                        sum_sp += (v - ct) * w
                    
            scalar_product[clus_idx] = sum_sp
    
    return scalar_product
    





@jit(nopython=True, parallel=True)
def numba_explore_best_shift(fifo_residuals, left_ind, centers, projector, candidates_idx,  maximum_jitter_shift, common_mask, sparse_mask_level1):

    nb_clus, width, nb_chan = centers.shape
    
    n_shift = maximum_jitter_shift*2 +1
    n_clus = len(candidates_idx)
    
    shift_scalar_product = np.zeros((n_clus, n_shift), dtype=np.float32)
    shift_distance = np.zeros((n_clus, n_shift), dtype=np.float32)
    
    for shi in prange(n_shift):
        shift =  shi - maximum_jitter_shift
        
        for clus in prange(len(candidates_idx)):
            clus_idx = candidates_idx[clus]
            
            sum_sp = 0.
            sum_d = 0.
            for chan in range(nb_chan):
                if common_mask[chan] or sparse_mask_level1[clus_idx, chan]:
                    for s in range(width):
                        v = fifo_residuals[left_ind+s+shift, chan]
                        ct = centers[clus_idx, s, chan]
                        
                        if sparse_mask_level1[clus_idx, chan]:
                            sum_sp += (v - ct) * projector[clus_idx, s, chan]
                        
                        if common_mask[chan]:
                            sum_d += (v - ct) * (v - ct)
                    
            shift_scalar_product[clus, shi] = sum_sp
            shift_distance[clus, shi] = sum_d
    
    return shift_scalar_product, shift_distance





@jit(nopython=True, parallel=True)
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


@jit(nopython=True, parallel=True)
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




