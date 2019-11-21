"""
Here implementation for testing parallel implementaion of peeler.
At the moment it is totally slower :(

"""

import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor



from .peeler_engine_classic import PeelerEngineClassic
from .peeler_engine_testing import PeelerEngineTesting

from .peeler_tools import *
from .peeler_tools import _dtype_spike


#~ class PeelerEngineParallel(PeelerEngineClassic):
class PeelerEngineParallel(PeelerEngineTesting):


    def initialize_before_each_segment(self, **kargs):
        PeelerEngineClassic.initialize_before_each_segment(self, **kargs)

        self.num_worker = 12 #  TODO move to set_params(...)
        
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
                for i, future in enumerate(futures):
                    spike = future.result()
                    if (spike.cluster_label >=0):
                        good_spikes.append(spike)
                        nb_good_spike+=1
                        # remove from residulals
                        self.on_accepted_spike(spike)
                        self.mask_not_already_tested[peak_inds[i] - self.n_span] = False
                    else:
                        # set this peak_ind as already tested
                        #~ print('unset', i, peak_inds[i])
                        self.mask_not_already_tested[peak_inds[i] - self.n_span] = False
                
                
                ## *** test with joblib.Parralel *** very bad benchmark on 4 cores
                #~ spikes = self.joblib_pcall(joblib.delayed(self.classify_and_align_next_spike)(peak_ind) for peak_ind in peak_inds)
                #~ for spike in spikes:
                    #~ if (spike.cluster_label >=0):
                        #~ good_spikes.append(spike)
                        #~ nb_good_spike+=1
                        #~ # remove from residulals
                        #~ self.on_accepted_spike(spike)
                    #~ else:
                        #~ # set this peak_ind as already tested
                        #~ self.mask_not_already_tested[peak_ind - self.n_span] = False


                    
                
                
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
            #~ print('next_peaks', next_peaks)
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



