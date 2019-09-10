"""
Here implementation for testing new ideas of peeler.

"""

import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor



from .peeler_engine_classic import PeelerEngineClassic

from .peeler_tools import *
from .peeler_tools import _dtype_spike

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags



import matplotlib.pyplot as plt


class PeelerEngineTesting(PeelerEngineClassic):
    #~ pass

    #~ def estimate_jitter(self, left_ind, cluster_idx):
        #~ return 0
    def initialize_before_each_segment(self, **kargs):
        PeelerEngineClassic.initialize_before_each_segment(self, **kargs)

        self.num_worker = 4
        
        self.executor = ThreadPoolExecutor(max_workers=self.num_worker)
        
        
        #~ self.joblib_pcall = joblib.Parallel(backend='loky', n_jobs=self.num_worker)
        #~ self.joblib_pcall = joblib.Parallel(backend='threading', n_jobs=self.num_worker)
        

        

    def process_one_chunk(self,  pos, sigs_chunk):
        #~ print('*'*10)
        t1 = time.perf_counter()
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        #~ t2 = time.perf_counter()
        #~ print('process_data', (t2-t1)*1000)
        
        
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
        
        
        
        # negative mask 1: not tested 0: already tested
        self.mask_not_already_tested[:] = True
        
        self.local_peaks_mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        
        
        good_spikes = []
        
        n_loop = 0
        t3 = time.perf_counter()
        while True:
            #~ print('peeler level +1')
            nb_good_spike = 0
            peak_inds = self.select_next_peaks()
            
            #~ print('start inner loop')
            while len(peak_inds)>0:
            
                #~ print('  peak_ind', peak_ind)
                #~ t2 = time.perf_counter()
                #~ print('  select_next_peak', (t2-t1)*1000)
                
                #~ if peak_ind == LABEL_NO_MORE_PEAK:
                    #~ print('break inner loop 1')
                    #~ break
                
                #~ t1 = time.perf_counter()
                #~ spike = self.classify_and_align_next_spike(peak_ind)
                
                ## *** test with ThreadPoolExecutor *** bad benchmark on 4 cores
                ## to be tested on more
                futures = [self.executor.submit(self.classify_and_align_next_spike, peak_ind) for peak_ind in peak_inds]
                for future in futures:
                    spike = future.result()
                    if (spike.cluster_label >=0):
                        good_spikes.append(spike)
                        nb_good_spike+=1
                
                
                ## *** test with joblib.Parralel *** very bad benchmark on 4 cores
                #~ spikes = self.joblib_pcall(joblib.delayed(self.classify_and_align_next_spike)(peak_ind) for peak_ind in peak_inds)
                #~ for spike in spikes:
                    #~ if (spike.cluster_label >=0):
                        #~ good_spikes.append(spike)
                        #~ nb_good_spike+=1
                    
                
                
                #~ peak_ind = self.select_next_peak()
                peak_inds = self.select_next_peaks()
            
            if nb_good_spike == 0:
                #~ print('break main loop')
                break
            else:
                
                t1 = time.perf_counter()
                for spike in good_spikes[-nb_good_spike:]:
                    peak_ind = spike.index
                    # TODO here make enlarge a bit with maximum_jitter_shift
                    sl1 = slice(peak_ind + self.n_left - 1 - self.n_span, peak_ind + self.n_right + 1 + self.n_span)
                    sl2 = slice(peak_ind + self.n_left - 1 - self.n_span, peak_ind + self.n_right + 1- self.n_span)
                    self.local_peaks_mask[sl2] = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals[sl1, :])
                    
                    # set neighboor untested
                    self.mask_not_already_tested[peak_ind - self.peak_width - self.n_span:peak_ind + self.peak_width - self.n_span] = True
                #~ t2 = time.perf_counter()
                #~ print('  update mask', (t2-t1)*1000)


        #~ t4 = time.perf_counter()
        #~ print('LOOP classify_and_align_one_spike', (t4-t3)*1000)
        #~ print('nb_good_spike', len(good_spikes), 'n_loop', n_loop)
        
        nolabel_indexes, = np.nonzero(~self.mask_not_already_tested)
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

    
    def select_next_peaks(self):
        # TODO find faster
        local_peaks_indexes,  = np.nonzero(self.local_peaks_mask & self.mask_not_already_tested)
        #~ print('select_next_peak')
        #~ print(local_peaks_indexes + self.n_span )
        if local_peaks_indexes.size>0:
            local_peaks_indexes += self.n_span
            
            next_peaks = []
            next_peaks.append(local_peaks_indexes[0])
            for i in range(self.num_worker-1):
                sel = local_peaks_indexes>(next_peaks[-1] + self.peak_width + 2*self.maximum_jitter_shift+1)
                #~ print(sel)
                if np.any(sel):
                    next_peaks.append(local_peaks_indexes[np.nonzero(sel)[0][0]])
                else:
                    break
            print('next_peaks', next_peaks)
            return next_peaks
            
            
            #~ amplitudes = np.max(np.abs(self.fifo_residuals[local_peaks_indexes, :]), axis=1)
            #~ print(self.fifo_residuals[local_peaks_indexes, :])
            #~ print(amplitudes)
            #~ print(amplitudes.shape)
            #~ ind = np.argmax(amplitudes)
            #~ print(ind)
            #~ return local_peaks_indexes[ind]
            #~ return local_peaks_indexes[0] + self.n_span
        else:
            return []
    
    def accept_tempate_OK(self, left_ind, cluster_idx, jitter):

        if np.abs(jitter) > (self.maximum_jitter_shift - 0.5):
            return False
        
        mask = self.catalogue['sparse_mask'][cluster_idx]
        
        shift = -int(np.round(jitter))
        jitter = jitter + shift
        left_ind = left_ind + shift
        if left_ind<0:
            return False
        new_left, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        pred_wf = pred_wf[:, :][:, mask]
        

        
        #~ full_wf0 = self.catalogue['centers0'][cluster_idx,: , :][:, mask]
        #~ full_wf1 = self.catalogue['centers1'][cluster_idx,: , :][:, mask]
        #~ full_wf2 = self.catalogue['centers2'][cluster_idx,: , :][:, mask]
        #~ pred_wf = (full_wf0+jitter*full_wf1+jitter**2/2*full_wf2)
        #~ new_left = left_ind

        

        # waveform L2 on mask
        waveform = self.fifo_residuals[new_left:new_left+self.peak_width,:]
        full_wf = waveform[:, :][:, mask]
        wf_nrj = np.sum(full_wf**2, axis=0)

        
        #~ thresh_ratio = 0.7
        thresh_ratio = 0.8
        
        # criteria per channel
        #~ residual_nrj = np.sum((full_wf-pred_wf)**2, axis=0)
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        #~ weight = self.weight_per_template[label]
        #~ crietria_weighted = (wf_nrj>residual_nrj).astype('float') * weight
        #~ accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        
        weigth = pred_wf ** 2
        residual = (full_wf-pred_wf)
        s = np.sum((full_wf**2>residual**2).astype(float) * weigth)
        #~ s = np.sum((pred_wf**2*weigth)>(residual*weigth))
        accept_template = s >np.sum(weigth) * thresh_ratio
        #~ print(s, np.sum(weigth) , np.sum(weigth)  * thresh_ratio)
        #~ exit()
        
        
        #DEBUG
        label = self.catalogue['cluster_labels'][cluster_idx]
        #~ if label in (0, ):
        if False:
        #~ if True:
            
            #~ print('accept_tempate',accept_template, 'label', label)
            #~ print(wf_nrj>residual_nrj)
            #~ print(weight)
            #~ print(crietria_weighted)
            #~ print(np.sum(crietria_weighted), np.sum(weight), np.sum(crietria_weighted)/np.sum(weight))
            #~ print()
            
            #~ if not accept_template:
                #~ print(wf_nrj>residual_nrj)
                #~ print(weight)
                #~ print(crietria_weighted)
                #~ print()
            print(s, np.sum(weigth) , np.sum(weigth)  * thresh_ratio)
            
            fig, axs = plt.subplots(nrows=3, sharex=True)
            axs[0].plot(full_wf.T.flatten(), color='b')
            if accept_template:
                axs[0].plot(pred_wf.T.flatten(), color='g')
            else:
                axs[0].plot(pred_wf.T.flatten(), color='r')
            
            axs[0].plot((full_wf-pred_wf).T.flatten(), color='m')
            
            axs[1].plot((full_wf**2).T.flatten(), color='b')
            axs[1].plot((residual**2).T.flatten(), color='m')
            
            criterium = (full_wf**2>residual**2).astype(float) * weigth
            axs[2].plot(criterium.T.flatten(), color='k')
            
            plt.show()
            
        
        #~ #ENDDEBUG
        
        
        return accept_template

