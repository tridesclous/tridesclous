import time

from .peeler_tools import *
from .peeler_tools import _dtype_spike
from .tools import make_color_dict

from .signalpreprocessor import signalpreprocessor_engines
#~ from .peakdetector import get_peak_detector_class


from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags

from . import pythran_tools
if hasattr(pythran_tools, '__pythran__'):
    HAVE_PYTHRAN = True
else:
    HAVE_PYTHRAN = False

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_loop_sparse_dist
except ImportError:
    HAVE_NUMBA = False





class PeelerEngineBase(OpenCL_Helper):
    def change_params(self, catalogue=None,
                                        chunksize=1024, 
                                        internal_dtype='float32', 
                                        maximum_jitter_shift = 4,
                                        
                                        save_bad_label=False,
                                        
                                        cl_platform_index=None,
                                        cl_device_index=None,
                                        ):
        """
        Set parameters for the Peeler.
        
        
        Parameters
        ----------
        catalogue: the catalogue (a dict)
            The catalogue made by CatalogueConstructor.
        chunksize: int (1024 by default)
            the size of chunk for processing.
        internal_dtype: 'float32' or 'float64'
            dtype of internal processing. float32 is OK. float64 is totally useless.
        maximum_jitter_shift
            Maximum allowed shift alignement between peak index and its template.
            
        """
        assert catalogue is not None
        self.catalogue = catalogue
        self.chunksize = chunksize
        self.internal_dtype= internal_dtype
        self.maximum_jitter_shift = maximum_jitter_shift
        self.save_bad_label = save_bad_label
        
        self.inter_sample_oversampling = self.catalogue['inter_sample_oversampling']
        
        self.cl_platform_index=None
        self.cl_device_index=None
        
        if self.catalogue['mode'] == 'sparse':
            self.use_sparse_template = True
        elif self.catalogue['mode'] == 'dense':
            self.use_sparse_template = False
        
        self.colors = make_color_dict(self.catalogue['clusters'])
        
        
        # DEBUG
        #~ print('DEBUG peakdetector all neigboor')
        #~ self.catalogue['peak_detector_params']['adjacency_radius_um'] = 0.
        #~ self.catalogue['peak_detector_params']['peak_span_ms'] = 0.
        # / DEBUG
        
        # precompute some value for jitter estimation
        if self.inter_sample_oversampling:
            n = self.catalogue['cluster_labels'].size
            self.catalogue['wf1_norm2'] = np.zeros(n)
            self.catalogue['wf2_norm2'] = np.zeros(n)
            self.catalogue['wf1_dot_wf2'] = np.zeros(n)
            for i, k in enumerate(self.catalogue['cluster_labels']):
                chan = self.catalogue['extremum_channel'][i]
                wf0 = self.catalogue['centers0'][i,: , chan]
                wf1 = self.catalogue['centers1'][i,: , chan]
                wf2 = self.catalogue['centers2'][i,: , chan]

                self.catalogue['wf1_norm2'][i] = wf1.dot(wf1)
                self.catalogue['wf2_norm2'][i] = wf2.dot(wf2)
                self.catalogue['wf1_dot_wf2'][i] = wf1.dot(wf2)
        
        
        #~ print('self.use_sparse_template', self.use_sparse_template)

        # make kernels
        #~ centers = self.catalogue['centers0']
        #~ self.nb_cluster = centers.shape[0]
        #~ self.peak_width = centers.shape[1]
        #~ self.nb_channel = centers.shape[2]
        
        
        self.threshold = self.catalogue['peak_detector_params']['relative_threshold']
        
        #~ print(centers.shape)
        #~ abs_centers = np.abs(centers)

        self.sparse_mask_level1 = self.catalogue['sparse_mask_level1']
        self.sparse_mask_level2 = self.catalogue['sparse_mask_level2']
        self.sparse_mask_level3 = self.catalogue['sparse_mask_level3']
        

        
        # weight of template per channel masked with level3
        #~ self.weight_per_template = np.zeros((self.nb_cluster, self.nb_channel), dtype='float32')
        #~ self.weight_per_template_dict = {}
        #~ centers = self.catalogue['centers0']
        #~ for i, k in enumerate(self.catalogue['cluster_labels']):
            #~ mask = self.sparse_mask_level2[i, :]
            #~ wf = centers[i, :, :][:, mask]
            #~ w = np.sum(wf**2, axis=0)
            #~ self.weight_per_template[i, mask] = w
            #~ self.weight_per_template_dict[i] = w
        
        
        self.projections = self.catalogue['projections']
        self.boundaries = self.catalogue['boundaries']
        

        #~ import matplotlib.pyplot as plt
        #~ for i in range(centers.shape[0]):
            #~ fig, ax = plt.subplots()
            #~ center = centers[i,:,:].copy()
            #~ center_sparse = center.copy()
            #~ center_sparse[:, ~mask[i, :]] = 0.
            #~ ax.plot(center.T.flatten(), color='g')
            #~ ax.plot(center_sparse.T.flatten(), color='r', ls='--')
            #~ ax.axhline(sparse_threshold_mad)
            #~ ax.axhline(-sparse_threshold_mad)
            #~ plt.show()
    
    def initialize(self, sample_rate=None, nb_channel=None, source_dtype=None, geometry=None, already_processed=False, processor_engine=None):
        self.nb_channel = nb_channel
        self.sample_rate = sample_rate
        self.source_dtype = source_dtype
        self.geometry = geometry
        self.already_processed = already_processed # this is globally set but can be change segment per segment
        

        if not self.already_processed:
            # signal processor class
            p = dict(self.catalogue['signal_preprocessor_params'])
            self.signalpreprocessor_engine = p.pop('engine')
            if processor_engine is not None:
                # can be force to opencl by geometrial_opencl
                self.signalpreprocessor_engine = processor_engine
            
            SignalPreprocessor_class = signalpreprocessor_engines[self.signalpreprocessor_engine]
            self.signalpreprocessor = SignalPreprocessor_class(sample_rate, nb_channel, self.chunksize, source_dtype)
            p['normalize'] = True
            p['signals_medians'] = self.catalogue['signals_medians']
            p['signals_mads'] = self.catalogue['signals_mads']
            
            if hasattr(self, 'ctx') and self.ctx is not None and self.signalpreprocessor_engine == 'opencl':
                # use local ctx and queue if exists for processor
                #~ print('yep', self.ctx)
                p['cl_platform_index'] = None
                p['cl_device_index'] = None
                p['ctx'] = self.ctx
                p['queue'] = self.queue
            
            self.signalpreprocessor.change_params(**p)
            self.internal_dtype = self.signalpreprocessor.output_dtype
            
            assert self.chunksize>self.signalpreprocessor.pad_width, 'pad_width ({}) is greater than chunksize ({})!'.format(self.signalpreprocessor.pad_width, self.chunksize)
            
        else:
            # no need
            self.signalpreprocessor = None
            self.internal_dtype = source_dtype

        # peak detector class
        self.peak_sign = self.catalogue['peak_detector_params']['peak_sign']
        self.relative_threshold = self.catalogue['peak_detector_params']['relative_threshold']
        peak_span_ms = self.catalogue['peak_detector_params']['peak_span_ms']
        self.n_span = int(sample_rate * peak_span_ms / 1000.)//2
        self.n_span = max(1, self.n_span)
        #~ self.peak_width = self.catalogue['peak_width']
        
        centers = self.catalogue['centers0']
        self.nb_cluster = centers.shape[0]
        #~ self.peak_width = centers.shape[1]
        self.nb_channel = centers.shape[2]
        
        self.n_left = self.catalogue['n_left']
        self.n_right = self.catalogue['n_right']
        self.n_left_long = self.catalogue['n_left_long']
        self.n_right_long = self.catalogue['n_right_long']
        self.peak_width = self.n_right - self.n_left
        self.peak_width_long = self.n_right_long - self.n_left_long
        
        self.extra_size = self.peak_width_long + self.maximum_jitter_shift*2 + self.n_span*2 + 1
        self.fifo_size = self.chunksize + self.extra_size
        
        assert self.chunksize > (self.extra_size+1), 'chunksize is too small because of n_size'
        
        alien_thresh = self.catalogue['clean_peaks_params']['alien_value_threshold']
        if alien_thresh is not None:
            if (alien_thresh < 0.) or np.isnan(alien_thresh):
                # -1 is equivalent to None
                alien_thresh = None
        self.alien_value_threshold = alien_thresh
        
        self.total_spike = 0
        
        self.near_border_good_spikes = []
        
        self.fifo_residuals = np.zeros((self.fifo_size, nb_channel), dtype=self.internal_dtype)
    
    def initialize_before_each_segment(self, already_processed=False):
        self.total_spike = 0
        self.near_border_good_spikes = []
        self.fifo_residuals = np.zeros((self.fifo_size, self.nb_channel), dtype=self.internal_dtype)
        
        if self.signalpreprocessor is not None:
            self.signalpreprocessor.initialize_stream()
        
        self.already_processed = already_processed

    def get_remaining_spikes(self):
        if len(self.near_border_good_spikes)>0:
            # deal with extra remaining spikes
            extra_spikes = self.near_border_good_spikes[0]
            extra_spikes = extra_spikes.take(np.argsort(extra_spikes['index']))
            self.total_spike += extra_spikes.size
            return extra_spikes




class PeelerEngineGeneric(PeelerEngineBase):
    # common base for PeelerEngineGeometrical and PeelerEngineClassic 
    # andPeelerEngineGeometricalCl

    def process_one_chunk(self,  pos, sigs_chunk):
        #~ if 16000 <pos<16400:
            #~ self._plot_debug = True
        #~ else:
            #~ self._plot_debug = False
        self._plot_debug = False
        #~ self._plot_debug = True
        
        #~ if pos >= 732 * self.chunksize:
            #~ self._plot_debug = True
            
            
            
        if self._plot_debug:
        #~ if True:
            print('*'*10)
            print('process_one_chunk', pos)
        
        abs_head_index, preprocessed_chunk = self.apply_processor( pos, sigs_chunk)
        
        # relation between inside chunk index and abs index
        to_local_shift = abs_head_index - self.fifo_size
        
        
        #~ t1 = time.perf_counter()
        self.detect_local_peaks_before_peeling_loop()
        #~ t2 = time.perf_counter()
        #~ print()
        #~ print('  detect_local_peaks_before_peeling_loop', (t2-t1)*1000)
        
        #~ self._debug_nb_accept_tempate = 0
        
        good_spikes = []
        

        if self._plot_debug:
            self._plot_before_peeling_loop()

        n_loop = 0
        t3 = time.perf_counter()
        
        while True: # main loop
            if self._plot_debug:
            #~ if True:
                print('** peeler level +1 **')
            nb_good_spike = 0
            
            
            # loop : one more peeler level
            while True: 
                #~ print()
                #~ t1 = time.perf_counter()
                spike = self.classify_and_align_next_spike()
                #~ t2 = time.perf_counter()
                #~ print('  classify_and_align_next_spike', (t2-t1)*1000, spike)
                #~ if spike.cluster_label <0:
                    #~ print('   spike.label', spike.cluster_label, 'spike.index', spike.index)

                if spike.cluster_label == LABEL_NO_MORE_PEAK:
                    break
                
                if (spike.cluster_label >=0):
                    #~ good_spikes.append(np.array([spike], dtype=_dtype_spike))
                    good_spikes.append(spike)
                    nb_good_spike+=1
                n_loop +=1 
            
            #~ if self._plot_debug:
                #~ print('***end inner loop', n_loop,'nb_good_spike', nb_good_spike)
            
            if nb_good_spike == 0:
                break
            else:
                #~ t1 = time.perf_counter()
                self.reset_to_not_tested(good_spikes[-nb_good_spike:])
                #~ t2 = time.perf_counter()
                #~ print('  reset_to_not_tested', (t2-t1)*1000)
            
            if self._plot_debug:
                self._plot_after_inner_peeling_loop()
        

        #~ t4 = time.perf_counter()
        #~ print('mainloop classify_and_align ', len(good_spikes), ' spike', (t4-t3)*1000, 'ms', 'n_loop', n_loop)
        
        if self._plot_debug:
        #~ if True:
            t4 = time.perf_counter()
            print('mainloop classify_and_align ', len(good_spikes), ' spike', (t4-t3)*1000, 'ms', 'n_loop', n_loop)
            self._plot_after_peeling_loop(good_spikes)
        
        #~ print(self._debug_nb_accept_tempate)
        #~ if  len(good_spikes)>0:
            #~ print('nb_good_spike', len(good_spikes), 'n_loop', n_loop, 'per spike', (t4-t3)*1000/len(good_spikes))
        
        if self.save_bad_label:
            bad_spikes = self.get_no_label_peaks()
            bad_spikes['index'] += to_local_shift
        
        
        if len(good_spikes)>0:
            # TODO remove from peak the very begining of the signal because of border filtering effects
            
            good_spikes = np.array(good_spikes, dtype=_dtype_spike)
            good_spikes['index'] += to_local_shift
            near_border = (good_spikes['index'] - to_local_shift)>=(self.chunksize+self.n_span)
            near_border_good_spikes = good_spikes[near_border].copy()
            good_spikes = good_spikes[~near_border]
            
            if self.save_bad_label:
                all_spikes = np.concatenate([good_spikes] + [bad_spikes] + self.near_border_good_spikes)
            else:
                all_spikes = np.concatenate([good_spikes] + self.near_border_good_spikes)
            self.near_border_good_spikes = [near_border_good_spikes] # for next chunk
        else:
            if self.save_bad_label:
                all_spikes = np.concatenate([bad_spikes] + self.near_border_good_spikes)
            elif len(self.near_border_good_spikes) > 0:
                all_spikes = np.concatenate(self.near_border_good_spikes)
            else:
                all_spikes = np.array([], dtype=_dtype_spike)
            self.near_border_good_spikes = []
        
        # all_spikes = all_spikes[np.argsort(all_spikes['index'])]
        all_spikes = all_spikes.take(np.argsort(all_spikes['index']))
        self.total_spike += all_spikes.size
        
        #~ print(good_spikes.size, all_spikes.size)
        return abs_head_index, preprocessed_chunk, self.total_spike, all_spikes

    def apply_processor(self, pos, sigs_chunk):
        if self.already_processed:
            abs_head_index, preprocessed_chunk =  pos, sigs_chunk
        else:
            abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_buffer_stream(pos, sigs_chunk)
        
        #shift residuals buffer and put the new one on right side
        fifo_roll_size = self.fifo_size-preprocessed_chunk.shape[0]
        if fifo_roll_size>0 and fifo_roll_size!=self.fifo_size:
            self.fifo_residuals[:fifo_roll_size,:] = self.fifo_residuals[-fifo_roll_size:,:]
            self.fifo_residuals[fifo_roll_size:,:] = preprocessed_chunk
        
        return abs_head_index, preprocessed_chunk 


    def classify_and_align_next_spike(self):
        
        
        if self._plot_debug:
            print('classify_and_align_next_spike')
        
        proposed_peak_ind, peak_chan = self.select_next_peak()
        if self._plot_debug:        
            print('peak ', 'index',  proposed_peak_ind, 'chan', peak_chan)
            
        if proposed_peak_ind == LABEL_NO_MORE_PEAK:
            return Spike(0, LABEL_NO_MORE_PEAK, 0)
        
        # left_ind is the waveform left border
        left_ind = proposed_peak_ind + self.n_left
        left_ind_long = proposed_peak_ind + self.n_left_long

        if left_ind_long+self.peak_width_long+self.maximum_jitter_shift+1>=self.fifo_size:
        #~ if left_ind+self.peak_width >=self.fifo_size:
            # TODO : remove this because maybe unecessry
            # too near right limits no label
            label = LABEL_RIGHT_LIMIT
            jitter = 0
            if self._plot_debug:
                print('LABEL_RIGHT_LIMIT', proposed_peak_ind, peak_chan)
        elif left_ind_long<=self.maximum_jitter_shift:
        #~ elif left_ind<0:
            # TODO : remove this because maybe unecessry
            # too near left limits no label
            #~ print('     LABEL_LEFT_LIMIT', left_ind)
            label = LABEL_LEFT_LIMIT
            jitter = 0
            if self._plot_debug:
                print('LABEL_LEFT_LIMIT', proposed_peak_ind, peak_chan)
        elif self.catalogue['centers0'].shape[0]==0:
            # empty catalogue
            label  = LABEL_UNCLASSIFIED
            jitter = 0
            #~ if self._plot_debug:
                #~ print('LABEL_UNCLASSIFIED', proposed_peak_ind, peak_chan)
        else:
            #~ t1 = time.perf_counter()
            waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            #~ t2 = time.perf_counter()
            #~ print('    get_waveform', (t2-t1)*1000)
            
            #~ if self.alien_value_threshold is not None and self.alien_value_threshold > 0 and \
            if self.alien_value_threshold is not None and \
                    np.any((waveform>self.alien_value_threshold) | (waveform<-self.alien_value_threshold)) :
                label  = LABEL_ALIEN
                jitter = 0
                #~ if self._plot_debug:
                    #~ print('LABEL_ALIEN', proposed_peak_ind, peak_chan)

            else:
                
                #~ t1 = time.perf_counter()
                #TODO try usewaveform to avoid new buffer ????
                
                cluster_idx, shift, best_template_info = self.get_best_template(left_ind, peak_chan)
                
                
                if shift is not None:
                    left_ind += shift
                    left_ind_long +=shift
                
                # cluster_idx can be <0 when no possible cluster with geometrical
                
                #~ t2 = time.perf_counter()
                #~ print('    get_best_template', (t2-t1)*1000)
                


                
                if cluster_idx is None or cluster_idx<0:
                    #~ print('LABEL_UNCLASSIFIED cluster_idx<0')
                    label  = LABEL_UNCLASSIFIED
                    jitter = 0
                    if self._plot_debug:
                    #~ if True:
                        self._plot_label_unclassified(left_ind, peak_chan, cluster_idx, jitter)
                    
                else:
                    label = None
                    #~ t1 = time.perf_counter()
                    #~ print('left_ind', left_ind, 'proposed_peak_ind', proposed_peak_ind)
                    if self.inter_sample_oversampling:
                        raise NotImplementedError # TODO propagate center_long to this section and peeler
                        jitter = self.estimate_jitter(left_ind, cluster_idx)
                        shift = -int(np.round(jitter))
                        
                        if (np.abs(jitter) > 0.5) and \
                                        (left_ind_long+shift+self.peak_width_long<self.fifo_size) and\
                                        ((left_ind_long + shift) >= 0):
                            # try better jitter
                            new_jitter = self.estimate_jitter(left_ind + shift, cluster_idx)
                            if np.abs(new_jitter)<np.abs(jitter):
                                jitter = new_jitter
                                left_ind += shift
                                left_ind_long += shift
                                shift = -int(np.round(new_jitter))
                        
                        # security to not be outside the fifo
                        if np.abs(shift) > self.maximum_jitter_shift:
                            label = LABEL_MAXIMUM_SHIFT
                            if self._plot_debug:
                                print('LABEL_MAXIMUM_SHIFT', proposed_peak_ind, peak_chan)
                        elif (left_ind_long+shift+self.peak_width_long)>=self.fifo_size:
                            # normally this should be resolve in the next chunk
                            label = LABEL_RIGHT_LIMIT
                            if self._plot_debug:
                                print('LABEL_RIGHT_LIMIT 2', proposed_peak_ind, peak_chan)
                        elif (left_ind_long + shift) < 0:
                            label = LABEL_LEFT_LIMIT
                            if self._plot_debug:
                                print('LABEL_LEFT_LIMIT 2', proposed_peak_ind, peak_chan)
                    else:
                        jitter = None
                    
                    if label is None:
                        #~ t1 = time.perf_counter()
                        ok = self.accept_tempate(left_ind, cluster_idx, jitter, best_template_info)
                        if ok:
                            label = self.catalogue['cluster_labels'][cluster_idx]
                        else:
                            label  = LABEL_UNCLASSIFIED
                            jitter = 0
                            if self._plot_debug:
                                self._plot_label_unclassified(left_ind, peak_chan, cluster_idx, jitter)
                        #~ t2 = time.perf_counter()
                        #~ print('    accept_tempate', (t2-t1)*1000)
        
        # second security check for borders
        if label>=0 and jitter is not None:
            left_ind_check = left_ind_long - np.round(jitter).astype('int64')
            if left_ind_check<0:
                label = LABEL_LEFT_LIMIT
                if self._plot_debug:
                    print('!!!!!!!ici LABEL_LEFT_LIMIT', label)

            elif (left_ind_check+self.peak_width_long) >=self.fifo_size:
                label = LABEL_RIGHT_LIMIT
                if self._plot_debug:
                    print('!!!!!!!ici LABEL_RIGHT_LIMIT', label)
                
        
        #~ if self._plot_debug:
            #~ import matplotlib.pyplot as plt
            #~ if label in (LABEL_LEFT_LIMIT, LABEL_RIGHT_LIMIT, LABEL_UNCLASSIFIED):
                #~ fig, ax = plt.subplots()
                #~ waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
                #~ ax.plot(waveform.T.flatten())
                
                #~ if label == LABEL_LEFT_LIMIT:
                    #~ ax.set_title('LABEL_LEFT_LIMIT')
                #~ if label == LABEL_RIGHT_LIMIT:
                    #~ ax.set_title('LABEL_RIGHT_LIMIT')
                #~ if label == LABEL_UNCLASSIFIED:
                    #~ ax.set_title('LABEL_UNCLASSIFIED')
                    


                #~ max_chan_ind = np.argmax(np.abs(waveform[-self.n_left, :]))
                #~ fig, ax = plt.subplots()
                #~ ax.plot(self.fifo_residuals[:, max_chan_ind])
                
                #~ ax.scatter([left_ind-self.n_left], [self.fifo_residuals[left_ind-self.n_left, max_chan_ind]], color='r')
                
                
                #~ plt.show()
        
        
        if label < 0:
            # set peak tested to not test it again
            #~ self.mask_not_already_tested[proposed_peak_ind - self.n_span] = False
            peak_ind = proposed_peak_ind
            #~ jitter = peak_chan
            #~ t1 = time.perf_counter()
            self.set_already_tested(peak_ind, peak_chan)
            #~ t2 = time.perf_counter()
            #~ print('    set_already_tested', (t2-t1)*1000)


        #~ self.update_peak_mask(peak_ind, label)
        #~ t2 = time.perf_counter()
        #~ print('    update_peak_mask', (t2-t1)*1000)
        else:
            if jitter is not None:
                # ensure jitter in range [-0.5, 0.5]
                shift = -int(np.round(jitter))
                if shift !=0:
                    jitter = jitter + shift
                    left_ind = left_ind + shift
                    left_ind_long = left_ind_long + shift
            
            peak_ind = left_ind - self.n_left
            
            
            # remove from residulals
            #~ t1 = time.perf_counter()
            self.on_accepted_spike(peak_ind, cluster_idx, jitter)
            #~ t2 = time.perf_counter()
            #~ print('    on_accepted_spike', (t2-t1)*1000)
            
        
        if self._plot_debug:
            print('Spike', peak_ind, label, jitter)
        
        return Spike(peak_ind, label, jitter)


    def estimate_jitter(self, left_ind, cluster_idx):
        
        chan_max = self.catalogue['extremum_channel'][cluster_idx]
        
        wf0 = self.catalogue['centers0'][cluster_idx,: , chan_max]
        wf1 = self.catalogue['centers1'][cluster_idx,: , chan_max]
        wf2 = self.catalogue['centers2'][cluster_idx,: , chan_max]

        wf = self.fifo_residuals[left_ind:left_ind+self.peak_width,chan_max]
        
        
        #it is  precompute that at init for speedup
        wf1_norm2= self.catalogue['wf1_norm2'][cluster_idx]
        wf2_norm2 = self.catalogue['wf2_norm2'][cluster_idx]
        wf1_dot_wf2 = self.catalogue['wf1_dot_wf2'][cluster_idx]
        
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
        
        #~ print(jitter1)
        return jitter1

