"""
Here the implementation of the OLD peeler.
It was used for tridesclous<=1.2.2

This is implementation is kept for comparison reason.
In some situtaion this peeler engine was making some mistake.
It was a bit faster.

Please do not use it anymore

"""

import time
import numpy as np

from .peeler_engine_classic import PeelerEngineClassic

from .peeler_tools import *
from .peeler_tools import _dtype_spike

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags



class PeelerEngineOldClassic(PeelerEngineClassic):
    def process_one_chunk(self,  pos, sigs_chunk):
    
        #~ print('*'*5)
        #~ print('chunksize', self.chunksize, '=', self.chunksize/self.sample_rate*1000, 'ms')
        
        t1 = time.perf_counter()
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        #~ t2 = time.perf_counter()
        #~ print('process_data', (t2-t1)*1000)
        
        
        #note abs_head_index is smaller than pos because prepcorcessed chunk
        # is late because of local filfilt in signalpreprocessor
        
        #shift rsiruals buffer and put the new one on right side
        t1 = time.perf_counter()
        fifo_roll_size = self.fifo_residuals.shape[0]-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_residuals.shape[0]:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk
        #~ t2 = time.perf_counter()
        #~ print('fifo move', (t2-t1)*1000.)

        
        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        # TODO remove from peak the very begining of the signal because of border filtering effects
        
     
        good_spikes = []
        #~ already_tested = []
        
        # negative mask 1: not tested 0: already tested
        mask_already_tested = np.ones(self.fifo_residuals.shape[0] - 2 * self.n_span, dtype='bool')
        
        local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        #~ print('sum(local_peaks_mask)', np.sum(local_peaks_mask))
        
        n_loop = 0
        t3 = time.perf_counter()
        while True:
            #detect peaks
            #~ t3 = time.perf_counter()
            
            #~ local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
            
            local_peaks_indexes,  = np.nonzero(local_peaks_mask & mask_already_tested)
            #~ print(local_peaks_indexes)
            local_peaks_indexes += self.n_span
            #~ exit()
            
            #~ t4 = time.perf_counter()
            #~ print('  detect peaks', (t4-t3)*1000.)pythran_loop_sparse_dist
            
            #~ if len(already_tested)>0:
                #~ local_peaks_to_check = local_peaks_indexes[~np.in1d(local_peaks_indexes, already_tested)]
            #~ else:
                #~ local_peaks_to_check = local_peaks_indexes
            
            n_ok = 0
            for local_ind in local_peaks_indexes:
                #~ print('    local_peak', local_peak, 'i', i)
                t1 = time.perf_counter()
                spike = self.classify_and_align_one_spike(local_ind, self.fifo_residuals, self.catalogue)
                t2 = time.perf_counter()
                #~ print('    classify_and_align_one_spike', (t2-t1)*1000., spike.cluster_label)
                
                if spike.cluster_label>=0:
                    #~ print('     >>spike.index', spike.index, spike.cluster_label, 'abs index', spike.index+shift)
                    
                    
                    #~ spikes = np.array([spike], dtype=_dtype_spike)
                    #~ prediction = make_prediction_signals(spikes, self.fifo_residuals.dtype, self.fifo_residuals.shape, self.catalogue, safe=False)
                    #~ self.fifo_residuals -= prediction
                    #~ spikes['index'] += shift
                    #~ good_spikes.append(spikes)
                    #~ n_ok += 1
                    
                    # substract one spike
                    pos, pred = make_prediction_on_spike_with_label(spike.index, spike.cluster_label, spike.jitter, self.fifo_residuals.dtype, self.catalogue)
                    self.fifo_residuals[pos:pos+self.peak_width, :] -= pred
                    
                    # append
                    spikes = np.array([spike], dtype=_dtype_spike)
                    spikes['index'] += shift
                    good_spikes.append(spikes)
                    n_ok += 1
                                    
                    # recompute peak in neiborhood
                    # here indexing is tricky 
                    # sl1 : we need n_pan more in each side
                    # sl2: we need a shift of n_span because smaler shape
                    sl1 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1 + self.n_span)
                    sl2 = slice(local_ind + self.n_left - 1 - self.n_span, local_ind + self.n_right + 1- self.n_span)
                    local_peaks_mask[sl2] = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals[sl1, :])
                    
                    
                    #~ print('    already_tested before', already_tested)
                    #~ already_tested = [ind for ind in already_tested if np.abs(spike.index-ind)>self.peak_width]
                    
                    # set neighboor untested
                    mask_already_tested[local_ind - self.peak_width - self.n_span:local_ind + self.peak_width - self.n_span] = True

                    
                    #~ print('    already_tested new deal', already_tested)
                else:
                    # set peak tested
                    #~ print(mask_already_tested.shape)
                    #~ print(self.fifo_residuals.shape)
                    #~ print(self.n_span)
                    mask_already_tested[local_ind - self.n_span] = False
                    #~ print('already tested', local_ind)
                    #~ already_tested.append(local_peak)
                n_loop += 1
            
            if n_ok==0:
                # no peak can be labeled
                # reserve bad spikes on the right limit for next time

                #~ local_peaks_indexes = local_peaks_indexes[local_peaks_indexes<(self.chunksize+self.n_span)]
                #~ bad_spikes = np.zeros(local_peaks_indexes.shape[0], dtype=_dtype_spike)
                #~ bad_spikes['index'] = local_peaks_indexes + shift
                #~ bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                
                nolabel_indexes, = np.nonzero(~mask_already_tested)
                nolabel_indexes += self.n_span
                nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
                bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
                bad_spikes['index'] = nolabel_indexes + shift
                bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
                
                break
        
        #~ t4 = time.perf_counter()
        #~ print('LOOP classify_and_align_one_spike', (t4-t3)*1000)
        #~ print('n_good', len(good_spikes), 'n_loop', n_loop)
        
        
        #concatenate, sort and count
        # here the trick is to keep spikes at the right border
        # and keep then until the next loop this avoid unordered spike
        if len(good_spikes)>0:
            good_spikes = np.concatenate(good_spikes)
            near_border = (good_spikes['index'] - shift)>=(self.chunksize+self.n_span)
            near_border_good_spikes = good_spikes[near_border].copy()
            good_spikes = good_spikes[~near_border]

            all_spikes = np.concatenate([good_spikes] + [bad_spikes] + self.near_border_good_spikes)
            self.near_border_good_spikes = [near_border_good_spikes] # for next chunk
        else:
            all_spikes = np.concatenate([bad_spikes] + self.near_border_good_spikes)
            self.near_border_good_spikes = []
        
        # all_spikes = all_spikes[np.argsort(all_spikes['index'])]
        all_spikes = all_spikes.take(np.argsort(all_spikes['index']))
        self.total_spike += all_spikes.size
        
        return abs_head_index, preprocessed_chunk, self.total_spike, all_spikes
    
    
    def classify_and_align_one_spike(self, local_index, residual, catalogue):
        # local_index is index of peaks inside residual and not
        # the absolute peak_pos. So time scaling must be done outside.
        
        peak_width = catalogue['peak_width']
        n_left = catalogue['n_left']
        #~ alien_value_threshold = catalogue['clean_waveforms_params']['alien_value_threshold']
        
        
        #ind is the windows border!!!!!
        left_ind = local_index + n_left

        if left_ind+peak_width+self.maximum_jitter_shift+1>=residual.shape[0]:
            # too near right limits no label
            label = LABEL_RIGHT_LIMIT
            jitter = 0
        elif left_ind<=self.maximum_jitter_shift:
            # too near left limits no label
            #~ print('     LABEL_LEFT_LIMIT', left_ind)
            label = LABEL_LEFT_LIMIT
            jitter = 0
        elif catalogue['centers0'].shape[0]==0:
            # empty catalogue
            label  = LABEL_UNCLASSIFIED
            jitter = 0
        else:
            waveform = residual[left_ind:left_ind+peak_width,:]
            
            if self.alien_value_threshold is not None and \
                    np.any((waveform>self.alien_value_threshold) | (waveform<-self.alien_value_threshold)) :
                label  = LABEL_ALIEN
                jitter = 0
            else:
                
                #~ t1 = time.perf_counter()
                label, jitter = self.estimate_one_jitter(waveform)
                #~ t2 = time.perf_counter()
                #~ print('  estimate_one_jitter', (t2-t1)*1000.)

                #~ jitter = -jitter
                #TODO debug jitter sign is positive on right and negative to left
                
                #~ print('label, jitter', label, jitter)
                
                # if more than one sample of jitter
                # then we try a peak shift
                # take it if better
                #TODO debug peak shift
                if np.abs(jitter) > 0.5 and label >=0:
                    prev_ind, prev_label, prev_jitter =left_ind, label, jitter
                    
                    shift = -int(np.round(jitter))
                    #~ print('classify and align shift', shift)
                    
                    if np.abs(shift) >self.maximum_jitter_shift:
                        #~ print('     LABEL_MAXIMUM_SHIFT avec shift')
                        label = LABEL_MAXIMUM_SHIFT
                    else:
                        left_ind = left_ind + shift
                        if left_ind+peak_width>=residual.shape[0]:
                            #~ print('     LABEL_RIGHT_LIMIT avec shift')
                            label = LABEL_RIGHT_LIMIT
                        elif left_ind < 0:
                            #~ print('     LABEL_LEFT_LIMIT avec shift')
                            label = LABEL_LEFT_LIMIT
                            #TODO: force to label anyway the spike if spike is at the left of FIFO
                        else:
                            waveform = residual[left_ind:left_ind+peak_width,:]
                            #~ print('    second estimate jitter')
                            new_label, new_jitter = self.estimate_one_jitter(waveform, label=label)
                            #~ new_label, new_jitter = self.estimate_one_jitter(waveform, label=None)
                            if np.abs(new_jitter)<np.abs(prev_jitter):
                                #~ print('keep shift')
                                label, jitter = new_label, new_jitter
                                local_index += shift
                            else:
                                #~ print('no keep shift worst jitter')
                                pass

        #security if with jitter the index is out
        if label>=0:
            local_pos = local_index - np.round(jitter).astype('int64') + n_left
            if local_pos<0:
                label = LABEL_LEFT_LIMIT
            elif (local_pos+peak_width) >=residual.shape[0]:
                label = LABEL_RIGHT_LIMIT
        
        return Spike(local_index, label, jitter)
    
    
    def estimate_one_jitter(self, waveform, label=None):
        """
        Estimate the jitter for one peak given its waveform
        
        label=None general case
        label not None when estimate_one_jitter is the second call
        frequently happen when abs(jitter)>0.5
        
        
        Method proposed by Christophe Pouzat see:
        https://hal.archives-ouvertes.fr/hal-01111654v1
        http://christophe-pouzat.github.io/LASCON2016/SpikeSortingTheElementaryWay.html
        
        for best reading (at least for me SG):
          * wf = the wafeform of the peak
          * k = cluster label of the peak
          * wf0, wf1, wf2 : center of catalogue[k] + first + second derivative
          * jitter0 : jitter estimation at order 0
          * jitter1 : jitter estimation at order 1
          * h0_norm2: error at order0
          * h1_norm2: error at order1
          * h2_norm2: error at order2
        """
        # This line is the slower part !!!!!!
        # cluster_idx = np.argmin(np.sum(np.sum((catalogue['centers0']-waveform)**2, axis = 1), axis = 1))
        
        catalogue = self.catalogue
        
        if label is None:
            #~ if self.use_opencl_with_sparse:
            if self.argmin_method == 'opencl':
                t1 = time.perf_counter()
                rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
                
                pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)
                pyopencl.enqueue_copy(self.queue,  self.rms_waveform_channel_cl, rms_waveform_channel)
                event = self.kern_waveform_distance(self.queue,  self.cl_global_size, self.cl_local_size,
                            self.one_waveform_cl, self.catalogue_center_cl, self.sparse_mask_cl, 
                            self.rms_waveform_channel_cl, self.waveform_distance_cl)
                pyopencl.enqueue_copy(self.queue,  self.waveform_distance, self.waveform_distance_cl)
                cluster_idx = np.argmin(self.waveform_distance)
                t2 = time.perf_counter()
                #~ print('       np.argmin opencl_with_sparse', (t2-t1)*1000., cluster_idx)


            #~ elif self.use_pythran_with_sparse:
            elif self.argmin_method == 'pythran':
                s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                    catalogue['centers0'],  self.sparse_mask)
                cluster_idx = np.argmin(s)
            elif self.argmin_method == 'numba':
                s = numba_loop_sparse_dist(waveform, catalogue['centers0'],  self.sparse_mask)
                cluster_idx = np.argmin(s)
            
            elif self.argmin_method == 'numpy':
                # replace by this (indentique but faster, a but)
                #~ t1 = time.perf_counter()
                d = catalogue['centers0']-waveform[None, :, :]
                d *= d
                #s = d.sum(axis=1).sum(axis=1)  # intuitive
                #s = d.reshape(d.shape[0], -1).sum(axis=1) # a bit faster
                s = np.einsum('ijk->i', d) # a bit faster
                cluster_idx = np.argmin(s)
                #~ t2 = time.perf_counter()
                #~ print('    np.argmin V2', (t2-t1)*1000., cluster_idx)
            else:
                raise(NotImplementedError())
            
            k = catalogue['cluster_labels'][cluster_idx]
        else:
            t1 = time.perf_counter()
            cluster_idx = np.nonzero(catalogue['cluster_labels']==label)[0][0]
            k = label
            t2 = time.perf_counter()
            #~ print('       second argmin', (t2-t1)*1000., cluster_idx)
        
        
        chan_max = catalogue['extremum_channel'][cluster_idx]
        #~ print('cluster_idx', cluster_idx, 'k', k, 'chan', chan)

        
        #~ return k, 0.

        wf0 = catalogue['centers0'][cluster_idx,: , chan_max]
        wf1 = catalogue['centers1'][cluster_idx,: , chan_max]
        wf2 = catalogue['centers2'][cluster_idx,: , chan_max]
        wf = waveform[:, chan_max]
        #~ print()
        #~ print(wf0.shape, wf.shape)
        
        
        #it is  precompute that at init speedup 10%!!! yeah
        #~ wf1_norm2 = wf1.dot(wf1)
        #~ wf2_norm2 = wf2.dot(wf2)
        #~ wf1_dot_wf2 = wf1.dot(wf2)
        wf1_norm2= catalogue['wf1_norm2'][cluster_idx]
        wf2_norm2 = catalogue['wf2_norm2'][cluster_idx]
        wf1_dot_wf2 = catalogue['wf1_dot_wf2'][cluster_idx]
        
        
        h = wf - wf0
        h0_norm2 = h.dot(h)
        h_dot_wf1 = h.dot(wf1)
        jitter0 = h_dot_wf1/wf1_norm2
        h1_norm2 = np.sum((h-jitter0*wf1)**2)
        #~ print(h0_norm2, h1_norm2)
        #~ print(h0_norm2 > h1_norm2)
        
        
        
        if h0_norm2 > h1_norm2:
            #order 1 is better than order 0
            h_dot_wf2 = np.dot(h,wf2)
            rss_first = -2*h_dot_wf1 + 2*jitter0*(wf1_norm2 - h_dot_wf2) + 3*jitter0**2*wf1_dot_wf2 + jitter0**3*wf2_norm2
            rss_second = 2*(wf1_norm2 - h_dot_wf2) + 6*jitter0*wf1_dot_wf2 + 3*jitter0**2*wf2_norm2
            jitter1 = jitter0 - rss_first/rss_second
            #~ h2_norm2 = np.sum((h-jitter1*wf1-jitter1**2/2*wf2)**2)
            #~ if h1_norm2 <= h2_norm2:
                #when order 2 is worse than order 1
                #~ jitter1 = jitter0
        else:
            jitter1 = 0.
        #~ print('jitter1', jitter1)
        #~ return k, 0.
        
        #~ print(np.sum(wf**2), np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2))
        #~ print(np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2))
        #~ return k, jitter1
        
        
        
        # criteria mono channel = old implementation
        #~ keep_template = np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2)
        
        # criteria multi channel
        mask = self.sparse_mask[cluster_idx]
        full_wf0 = catalogue['centers0'][cluster_idx,: , :][:, mask]
        full_wf1 = catalogue['centers1'][cluster_idx,: , :][:, mask]
        full_wf2 = catalogue['centers2'][cluster_idx,: , :][:, mask]
        full_wf = waveform[:, :][:, mask]
        weight = self.weight_per_template[k]
        wf_nrj = np.sum(full_wf**2, axis=0)
        res_nrj = np.sum((full_wf-(full_wf0+jitter1*full_wf1+jitter1**2/2*full_wf2))**2, axis=0)
        # criteria per channel
        crietria_weighted = (wf_nrj>res_nrj).astype('float') * weight
        accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        
        
        if accept_template:
            # keep prediction
            return k, jitter1
        else:
            #otherwise the prediction is bad
            return LABEL_UNCLASSIFIED, 0.
    
    