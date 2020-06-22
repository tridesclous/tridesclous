try:
    import numba
    from numba import jit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

import numpy as np

# OLD
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

# OLD
@jit(parallel=True)
#~ def numba_loop_sparse_dist_with_geometry(waveform, centers,  possibles_cluster_idx, rms_waveform_channel,channel_considered):
def numba_loop_sparse_dist_with_geometry(waveform, centers,  possibles_cluster_idx, channel_considered, template_weight):
    
    nb_total_clus, width, nb_chan = centers.shape
    nb_clus = possibles_cluster_idx.size
    
    #rms_waveform_channel = np.sum(waveform**2, axis=0)#.astype('float32')
    waveform_distance = np.zeros((nb_clus,), dtype=np.float32)
    
    for clus in prange(len(possibles_cluster_idx)):
        cluster_idx = possibles_cluster_idx[clus]
        sum = 0
        sum_w = 0
        for c in channel_considered:
            #~ if mask[cluster_idx, c]:
            for s in range(width):
                d = waveform[s, c] - centers[cluster_idx, s, c]
                #~ sum += d*d
                sum += d*d * template_weight[s, c]
                
                #~ w = np.abs(centers[cluster_idx, s, c])
                #~ sum += d*d*w
                #~ sum_w += w

                #~ sum += waveform[s, c] * centers[cluster_idx, s, c]

                
            #~ else:
                #~ sum +=rms_waveform_channel[c]
        waveform_distance[clus] = sum
        #~ waveform_distance[clus] = sum / sum_w
    
    return waveform_distance


# OLD
@jit(parallel=True)
def numba_explore_shifts(long_waveform, one_center,  one_mask, maximum_jitter_shift, template_weight):
    width, nb_chan = one_center.shape
    n = maximum_jitter_shift*2 +1
    
    all_dist = np.zeros((n, ), dtype=np.float32)
    
    for shift in prange(n):
        sum = 0
        sum_w = 0
        for c in range(nb_chan):
            if one_mask[c]:
                for s in range(width):
                    d = long_waveform[shift+s, c] - one_center[s, c]
                    #~ w = np.abs(one_center[s, c])
                    w = template_weight[s, c]
                    #~ sum += d*d
                    sum += d*d*w
                    sum_w += w
        #~ all_dist[shift] = sum
        all_dist[shift] = sum / sum_w
    
    return all_dist



@jit(parallel=True)
def numba_explore_best_template(fifo_residuals, left_ind, peak_chan_ind, centers, centers_normed, template_weight,
                                    sparse_mask_level1, sparse_mask_level2, sparse_mask_level3, sparse_mask_level4):
    #~ full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
    
    nb_clus, width, nb_chan = centers.shape
    
    scalar_product = np.zeros((nb_clus,), dtype=np.float32)
    weighted_distance = np.zeros((nb_clus,), dtype=np.float32)
    
    for clus_idx in prange(nb_clus):
        
        if not sparse_mask_level3[clus_idx, peak_chan_ind]:
            # peak chan is not in template at all!!
            # do not compute  : faster
            scalar_product[clus_idx] = -1.
            weighted_distance[clus_idx] = np.inf
        else:
            sum_sc = 0.
            sum_wd = 0.
            for chan in range(nb_chan):
                if sparse_mask_level1[clus_idx, chan]:
                    for s in range(width):
                        v = fifo_residuals[left_ind+s, chan]
                        
                        w = template_weight[s, chan]
                        
                        sum_sc += v * centers_normed[clus_idx, s, chan] # * w
                        
                        #~ d = v - centers[clus_idx, s, chan]
                        #~ sum_wd += d*d * w
            
            scalar_product[clus_idx] = sum_sc
            weighted_distance[clus_idx] = sum_wd
    
    return scalar_product, weighted_distance


@jit(parallel=True)
def numba_explore_best_shift(fifo_residuals, left_ind, centers, centers_normed, candidates_idx, template_weight,
                    common_sparse_mask, maximum_jitter_shift, sparse_mask_level1):

    nb_clus, width, nb_chan = centers.shape
    
    n_shift = maximum_jitter_shift*2 +1
    n_clus = len(candidates_idx)
    
    shift_scalar_product = np.zeros((n_clus, n_shift), dtype=np.float32)
    shift_distance = np.zeros((n_clus, n_shift), dtype=np.float32)
    
    for shi in prange(n_shift):
        shift =  shi - maximum_jitter_shift
        
        for clus in prange(len(candidates_idx)):
            clus_idx = candidates_idx[clus]
            
            sum_sc = 0.
            sum_wd = 0.
            for chan in range(nb_chan):
                if common_sparse_mask[chan] or sparse_mask_level1[clus_idx, chan]:
                    
                    for s in range(width):
                        v = fifo_residuals[left_ind+s+shift, chan]
                        w = template_weight[s, chan]
                        
                        if sparse_mask_level1[clus_idx, chan]:
                            sum_sc += v * centers_normed[clus_idx, s, chan] # * w
                        
                        if common_sparse_mask[chan]:
                            d = v - centers[clus_idx, s, chan]
                            sum_wd += d * d * w
            
            shift_distance[clus, shi] = sum_wd
            shift_scalar_product[clus, shi] = sum_sc
    
    return shift_scalar_product, shift_distance




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




