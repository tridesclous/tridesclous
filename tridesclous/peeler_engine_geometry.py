"""
Here implementation that tale in account the geometry
of the probe to speed up template matching.

"""

import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor



from .peeler_engine_classic import PeelerEngineClassic

from .peeler_tools import *
from .peeler_tools import _dtype_spike

import sklearn.metrics.pairwise

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags

from .peakdetector import PeakDetectorSpatiotemporal, PeakDetectorSpatiotemporal_OpenCL

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_loop_sparse_dist_with_geometry
except ImportError:
    HAVE_NUMBA = False




import matplotlib.pyplot as plt


class PeelerEngineGeometry(PeelerEngineClassic):
    pass
    def change_params(self, adjacency_radius_um=200, **kargs):
        PeelerEngineClassic.change_params(self, **kargs)
        
        self.adjacency_radius_um = adjacency_radius_um
        
        assert self.use_sparse_template
        
        

    def initialize_before_each_segment(self, **kargs):
        PeelerEngineClassic.initialize_before_each_segment(self, **kargs)

        p = dict(self.catalogue['peak_detector_params'])
        _ = p.pop('peakdetector_engine', 'numpy')
        #~ self.peakdetector = PeakDetectorSpatiotemporal(self.sample_rate, self.nb_channel,
                        #~ self.chunksize+self.n_side, self.internal_dtype, self.geometry)
        
        # TODO size fido
        self.peakdetector = PeakDetectorSpatiotemporal_OpenCL(self.sample_rate, self.nb_channel,
                                                        self.fifo_residuals.shape[0]-2*self.n_span, self.internal_dtype, self.geometry)
        self.peakdetector.change_params(**p)

        
        self.mask_not_already_tested = np.ones((self.fifo_residuals.shape[0] - 2 * self.n_span,self.nb_channel),  dtype='bool')

        d = sklearn.metrics.pairwise.euclidean_distances(self.geometry)
        self.channels_adjacency = {}
        for c in range(self.nb_channel):
            nearest, = np.nonzero(d[c, :]<self.adjacency_radius_um)
            self.channels_adjacency[c] = nearest
            #~ print(c, nearest)
        
    
    
#~ self.sparse_mask

    def process_one_chunk(self,  pos, sigs_chunk):
        print('process_one_chunk', pos)
        t1 = time.perf_counter()
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        
        #shift rsiruals buffer and put the new one on right side
        t1 = time.perf_counter()
        fifo_roll_size = self.fifo_residuals.shape[0]-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_residuals.shape[0]:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk

        # relation between inside chunk index and abs index
        shift = abs_head_index - self.fifo_residuals.shape[0]
        
        # negative mask 1: not tested 0: already tested
        self.mask_not_already_tested[:] = True
        
        #~ self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        
        
        good_spikes = []
        
        n_loop = 0
        t3 = time.perf_counter()
        while True:
            #~ print('peeler level +1')
            nb_good_spike = 0
            peak_ind, peak_chan = self.select_next_peak()
            
            
            #~ print('start inner loop')
            while peak_ind != LABEL_NO_MORE_PEAK:
            
                t1 = time.perf_counter()
                spike = self.classify_and_align_next_spike(peak_ind, peak_chan)
                #~ print(spike)
                
                if spike.cluster_label == LABEL_NO_MORE_PEAK:
                    break
                
                if (spike.cluster_label >=0):
                    #~ good_spikes.append(np.array([spike], dtype=_dtype_spike))
                    good_spikes.append(spike)
                    nb_good_spike+=1
                    
                    # remove from residulals
                    self.on_accepted_spike(spike)
                    
                    self.mask_not_already_tested[peak_ind - self.n_span, peak_chan] = False # this save lot of time
                else:
                    # set this peak_ind as already tested
                    self.mask_not_already_tested[peak_ind - self.n_span, peak_chan] = False

                peak_ind, peak_chan = self.select_next_peak()
                #~ print(peak_ind, peak_chan)
                
                
                #~ # debug
                n_loop +=1 
                
            if nb_good_spike == 0:
                #~ print('break main loop')
                break
            else:
                
                t1 = time.perf_counter()
                for spike in good_spikes[-nb_good_spike:]:
                    peak_ind = spike.index
                    # TODO here make enlarge a bit with maximum_jitter_shift
                    #~ sl1 = slice(peak_ind + self.n_left - 1 - self.n_span, peak_ind + self.n_right + 1 + self.n_span)
                    #~ sl2 = slice(peak_ind + self.n_left - 1 - self.n_span, peak_ind + self.n_right + 1- self.n_span)
                    #~ self.local_peaks_mask[sl2] = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals[sl1, :])
                    #TODO
                    self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
                    
                    # set neighboor untested
                    # TODO more efficient !!!!!!!!!
                    cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
                    mask = self.catalogue['sparse_mask'][cluster_idx, :]
                    for c in np.nonzero(mask)[0]:
                        #~ print(c)
                        self.mask_not_already_tested[peak_ind - self.peak_width - self.n_span:peak_ind + self.peak_width - self.n_span, c] = True
                #~ t2 = time.perf_counter()
                #~ print('  update mask', (t2-t1)*1000)


        #~ t4 = time.perf_counter()
        #~ print('LOOP classify_and_align_one_spike', (t4-t3)*1000)
        #~ print('nb_good_spike', len(good_spikes), 'n_loop', n_loop)
        
        nolabel_indexes, chan_indexes = np.nonzero(~self.mask_not_already_tested)
        nolabel_indexes += self.n_span
        nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
        bad_spikes['index'] = nolabel_indexes + shift
        bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        
        if len(good_spikes)>0:
            # TODO remove from peak the very begining of the signal because of border filtering effects
            
            good_spikes = np.array(good_spikes, dtype=_dtype_spike)
            good_spikes['index'] += shift
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


    def select_next_peak(self):
        #~ print('select_next_peak')
        # TODO find faster
        local_peaks_indexes, chan_indexes  = np.nonzero(self.local_peaks_mask & self.mask_not_already_tested)
        #~ print('select_next_peak')
        #~ print(local_peaks_indexes + self.n_span )
        if local_peaks_indexes.size>0:
            local_peaks_indexes += self.n_span
            amplitudes = np.abs(self.fifo_residuals[local_peaks_indexes, chan_indexes])
            ind = np.argmax(amplitudes)
            peak_ind = local_peaks_indexes[ind]
            chan_ind = chan_indexes[ind]
            return peak_ind, chan_ind
            #~ return local_peaks_indexes[0]
        else:
            return LABEL_NO_MORE_PEAK, None

    def classify_and_align_next_spike(self, proposed_peak_ind, chan_ind):
        # left_ind is the waveform left border
        left_ind = proposed_peak_ind + self.n_left
        
        #~ print('classify_and_align_next_spike')
        #~ print('chan_ind', chan_ind)
        #~ print(self.catalogue['sparse_mask'].shape)
        
        #~ fig, ax = plt.subplots()
        #~ ax.plot(self.fifo_residuals[:, chan_ind])
        #~ ax.scatter([proposed_peak_ind], [self.fifo_residuals[proposed_peak_ind, chan_ind]], color='r')
        #~ plt.show()
        
        
        

        if left_ind+self.peak_width+self.maximum_jitter_shift+1>=self.fifo_residuals.shape[0]:
            # TODO : remove this because maybe unecessry
            # too near right limits no label
            label = LABEL_RIGHT_LIMIT
            jitter = 0
        elif left_ind<=self.maximum_jitter_shift:
            # TODO : remove this because maybe unecessry
            # too near left limits no label
            #~ print('     LABEL_LEFT_LIMIT', left_ind)
            label = LABEL_LEFT_LIMIT
            jitter = 0
        elif self.catalogue['centers0'].shape[0]==0:
            # empty catalogue
            label  = LABEL_UNCLASSIFIED
            jitter = 0
        else:
            waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            
            if self.alien_value_threshold is not None and \
                    np.any((waveform>self.alien_value_threshold) | (waveform<-self.alien_value_threshold)) :
                label  = LABEL_ALIEN
                jitter = 0
            else:
                
                #~ t1 = time.perf_counter()
                #TODO try usewaveform to avoid new buffer ????
                
                cluster_idx = self.get_best_template(left_ind, chan_ind)
                #~ t2 = time.perf_counter()
                #~ print('    get_best_template', (t2-t1)*1000)
                


                
                
                #~ t1 = time.perf_counter()
                jitter = self.estimate_jitter(left_ind, cluster_idx)
                #~ t2 = time.perf_counter()
                #~ print('    estimate_jitter', (t2-t1)*1000)
                
                #~ t1 = time.perf_counter()
                ok = self.accept_tempate(left_ind, cluster_idx, jitter)
                #~ t2 = time.perf_counter()
                #~ print('    accept_tempate', (t2-t1)*1000)

                # DEBUG
                #~ label = self.catalogue['cluster_labels'][cluster_idx]
                #~ if label in (5, 8):
                #~ if label in (10, ):
                    #~ print('label', label, 'ok', ok, 'jitter', jitter)
                # END DEBUG                

                if  not ok:
                    label  = LABEL_UNCLASSIFIED
                    jitter = 0
                else:
                    #~ print('cluster_idx', cluster_idx, 'jitter', jitter)
                    shift = -int(np.round(jitter))
                    if (np.abs(jitter) > 0.5) and \
                            (left_ind+shift+self.peak_width<self.fifo_residuals.shape[0]) and\
                            ((left_ind + shift) >= 0):
                        #~ shift = -int(np.round(jitter))
                        
                        # debug
                        #~ new_cluster_idx = self.get_best_template(left_ind+shift)
                        #~ new_jitter = self.estimate_jitter(left_ind + shift, new_cluster_idx)
                        #~ ok = self.accept_tempate(left_ind+shift, new_cluster_idx, new_jitter)
                        # end debug
                        new_jitter = self.estimate_jitter(left_ind + shift, cluster_idx)
                        ok = self.accept_tempate(left_ind+shift, cluster_idx, new_jitter)
                        if ok and np.abs(new_jitter)<np.abs(jitter):
                            jitter = new_jitter
                            left_ind += shift
                            shift = -int(np.round(jitter))
                            
                            # debug
                            #~ if cluster_idx != new_cluster_idx:
                                #~ print('cluster_idx != new_cluster_idx')
                            #~ cluster_idx = new_cluster_idx
                            
                            
                    
                    # ensure jitter in range [-0.5, 0.5]
                    # WRONG IDEA because the mask_not_already_tested will not updated at the good place
                    #~ if shift !=0:
                        #~ jitter = jitter + shift
                        #~ left_ind = left_ind + shift
                    
                    # security to not be outside the fifo
                    if np.abs(shift) >self.maximum_jitter_shift:
                        label = LABEL_MAXIMUM_SHIFT
                    elif (left_ind+shift+self.peak_width)>=self.fifo_residuals.shape[0]:
                        # normally this should be resolve in the next chunk
                        label = LABEL_RIGHT_LIMIT
                    elif (left_ind + shift) < 0:
                        # TODO assign the previous label ???
                        label = LABEL_LEFT_LIMIT
                    else:
                        label = self.catalogue['cluster_labels'][cluster_idx]

        #security if with jitter the index is out
        if label>=0:
            left_ind_check = left_ind - np.round(jitter).astype('int64')
            if left_ind_check<0:
                label = LABEL_LEFT_LIMIT
            elif (left_ind_check+self.peak_width) >=self.fifo_residuals.shape[0]:
                label = LABEL_RIGHT_LIMIT
        
        if label < 0:
            # set peak tested to not test it again
            #~ self.mask_not_already_tested[proposed_peak_ind - self.n_span] = False
            peak_ind = proposed_peak_ind

        #~ self.update_peak_mask(peak_ind, label)
        #~ t2 = time.perf_counter()
        #~ print('    update_peak_mask', (t2-t1)*1000)
        else:
            # ensure jitter in range [-0.5, 0.5]
            shift = -int(np.round(jitter))
            if shift !=0:
                jitter = jitter + shift
                left_ind = left_ind + shift
            
            peak_ind = left_ind - self.n_left
        
        return Spike(peak_ind, label, jitter)


    def get_best_template(self, left_ind, chan_ind):
        
        assert self.argmin_method == 'numba'
        
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]

        #~ print('get_best_template')
        #~ print('chan_ind', chan_ind)
        
        possibles_cluster_idx, = np.nonzero(self.catalogue['sparse_mask'][:, chan_ind])
        #~ print(possibles_cluster_idx)
        #~ print('possibles_cluster_idx', possibles_cluster_idx.size)
        
        #~ print(self.catalogue['sparse_mask'].shape)
        
        #~ fig, ax = plt.subplots()
        #~ ax.plot(self.fifo_residuals[:, chan_ind])
        #~ ax.scatter([left_ind-self.n_left], [self.fifo_residuals[left_ind-self.n_left, chan_ind]], color='r')
        #~ plt.show()
        
        
        #~ for cluster_idx in possibles_cluster_idx:
            #~ fig, ax = plt.subplots()
            #~ ax.plot(waveform.T.flatten(), color='k')
            #~ ax.plot(self.catalogue['centers0'][cluster_idx, :, :].T.flatten(), color='m')
            #~ ax.set_title(str(cluster_idx))
            #~ plt.show()
            
            
        
        
        
        
        if self.argmin_method == 'opencl':
            rms_waveform_channel = np.sum(waveform**2, axis=0).astype('float32')
            
            pyopencl.enqueue_copy(self.queue,  self.one_waveform_cl, waveform)
            pyopencl.enqueue_copy(self.queue,  self.rms_waveform_channel_cl, rms_waveform_channel)
            event = self.kern_waveform_distance(self.queue,  self.cl_global_size, self.cl_local_size,
                        self.one_waveform_cl, self.catalogue_center_cl, self.sparse_mask_cl, 
                        self.rms_waveform_channel_cl, self.waveform_distance_cl)
            pyopencl.enqueue_copy(self.queue,  self.waveform_distance, self.waveform_distance_cl)
            cluster_idx = np.argmin(self.waveform_distance)
        
        elif self.argmin_method == 'pythran':
            s = pythran_tools.pythran_loop_sparse_dist(waveform, 
                                self.catalogue['centers0'],  self.catalogue['sparse_mask'])
            cluster_idx = np.argmin(s)
        
        elif self.argmin_method == 'numba':
            s = numba_loop_sparse_dist_with_geometry(waveform, self.catalogue['centers0'],  self.catalogue['sparse_mask'], possibles_cluster_idx, self.channels_adjacency[chan_ind])
            cluster_idx = possibles_cluster_idx[np.argmin(s)]
        
        elif self.argmin_method == 'numpy':
            # replace by this (indentique but faster, a but)
            d = self.catalogue['centers0']-waveform[None, :, :]
            d *= d
            #s = d.sum(axis=1).sum(axis=1)  # intuitive
            #s = d.reshape(d.shape[0], -1).sum(axis=1) # a bit faster
            s = np.einsum('ijk->i', d) # a bit faster
            cluster_idx = np.argmin(s)
        else:
            raise(NotImplementedError())
        
        
        #~ print('cluster_idx', cluster_idx)
        
        #~ plot_chan = self.channels_adjacency[chan_ind].tolist().index(chan_ind)
        #~ fig, ax = plt.subplots()
        #~ ax.axvline(plot_chan * self.peak_width - self.n_left)
        mask = self.catalogue['sparse_mask'][cluster_idx, :]
        #~ mask = self.channels_adjacency[chan_ind]
        #~ wf = waveform[:, mask]
        #~ wf0 = self.catalogue['centers0'][cluster_idx, :, :][:, mask]
        #~ ax.plot(wf.T.flatten(), color='k')
        #~ ax.plot(wf0.T.flatten(), color='m')
        #~ plt.show()

        
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        return cluster_idx


    def accept_tempate(self, left_ind, cluster_idx, jitter):
        # criteria mono channel = old implementation
        #~ keep_template = np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2)
        
        if np.abs(jitter) > (self.maximum_jitter_shift - 0.5):
            return False
        
        # criteria multi channel
        mask = self.catalogue['sparse_mask'][cluster_idx]
        full_wf0 = self.catalogue['centers0'][cluster_idx,: , :][:, mask]
        full_wf1 = self.catalogue['centers1'][cluster_idx,: , :][:, mask]
        full_wf2 = self.catalogue['centers2'][cluster_idx,: , :][:, mask]
        
        # waveform L2 on mask
        waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        full_wf = waveform[:, :][:, mask]
        wf_nrj = np.sum(full_wf**2, axis=0)
        
        # prediction L2 on mask
        label = self.catalogue['cluster_labels'][cluster_idx]
        weight = self.weight_per_template[label]
        pred_wf = (full_wf0+jitter*full_wf1+jitter**2/2*full_wf2)
        
        residual_nrj = np.sum((full_wf-pred_wf)**2, axis=0)
        
        # criteria per channel
        crietria_weighted = (wf_nrj>residual_nrj).astype('float') * weight
        #~ accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        accept_template = np.sum(crietria_weighted) >= 0.7 * np.sum(weight)

        #~ if True:
            
            #~ fig, axs = plt.subplots(nrows=2, sharex=True)
            #~ axs[0].plot(full_wf.T.flatten(), color='b')
            #~ if accept_template:
                #~ axs[0].plot(pred_wf.T.flatten(), color='g')
            #~ else:
                #~ axs[0].plot(pred_wf.T.flatten(), color='r')
            
            #~ axs[0].plot((full_wf-pred_wf).T.flatten(), color='m')
            
            #~ plt.show()
            



        
        #DEBUG
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        #~ if label in (10, ):
            
            #~ print('accept_tempate',accept_template, 'label', label)
            #~ print(wf_nrj>res_nrj)
            #~ print(weight)
            #~ print(crietria_weighted)
            #~ print(np.sum(crietria_weighted), np.sum(weight), np.sum(crietria_weighted)/np.sum(weight))
            
            #~ print()
        #~ #ENDDEBUG
        
        
        return accept_template



