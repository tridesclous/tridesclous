"""
Here implementation that tale in account the geometry
of the probe to speed up template matching.

"""

import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor
import itertools



from .peeler_engine_base import PeelerEngineGeneric

from .peeler_tools import *
from .peeler_tools import _dtype_spike

import sklearn.metrics.pairwise

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags

from .peakdetector import get_peak_detector_class

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_explore_best_shift, numba_sparse_scalar_product
except ImportError:
    HAVE_NUMBA = False







class PeelerEngineGeometrical(PeelerEngineGeneric):
    def change_params(self,  **kargs):
        PeelerEngineGeneric.change_params(self, **kargs)

    def initialize(self, **kargs):
        PeelerEngineGeneric.initialize(self, **kargs)
        
        # create peak detector
        p = dict(self.catalogue['peak_detector_params'])

        self.peakdetector_engine = p.pop('engine')
        self.peakdetector_method = p.pop('method')
        
        PeakDetector_class = get_peak_detector_class(self.peakdetector_method, self.peakdetector_engine)
        
        chunksize = self.fifo_size-2*self.n_span # not the real chunksize here
        self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        chunksize, self.internal_dtype, self.geometry)
        
        self.peakdetector.change_params(**p)
        
        # some attrs
        self.shifts = np.arange(-self.maximum_jitter_shift, self.maximum_jitter_shift+1)
        self.nb_shift = self.shifts.size
        
        #~ self.channel_distances = sklearn.metrics.pairwise.euclidean_distances(self.geometry).astype('float32')
        #~ self.channels_adjacency = {}
        #~ for c in range(self.nb_channel):
            #~ if self.use_sparse_template:
                #~ nearest, = np.nonzero(self.channel_distances[c, :]<self.adjacency_radius_um)
                #~ self.channels_adjacency[c] = nearest
            #~ else:
                #~ self.channels_adjacency[c] = np.arange(self.nb_channel, dtype='int64')
        
        self.mask_already_tested = np.zeros((self.fifo_size, self.nb_channel), dtype='bool')


    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        self.peakdetector.initialize_stream()


    def detect_local_peaks_before_peeling_loop(self):
        # reset tested mask
        self.mask_already_tested[:] = False
        # and detect peak
        self.re_detect_local_peak()
        
        #~ print('detect_local_peaks_before_peeling_loop', self.pending_peaks.size)

    def re_detect_local_peak(self):
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        if mask.ndim ==1:
            #~ mask &= ~self.mask_already_tested[self.n_span:-self.n_span, 0]
            sample_indexes,   = np.nonzero(mask)
            sample_indexes += self.n_span
            tested = self.mask_already_tested[sample_indexes, 0]
            sample_indexes = sample_indexes[~tested]
            chan_indexes = np.zeros(sample_indexes.size, dtype='int64')
        else:
            #~ mask &= ~self.mask_already_tested[self.n_span:-self.n_span, :]
            sample_indexes, chan_indexes  = np.nonzero(mask)
            sample_indexes += self.n_span
            tested = self.mask_already_tested[sample_indexes, chan_indexes]
            sample_indexes = sample_indexes[~tested]
            chan_indexes = chan_indexes[~tested]
        
        
        amplitudes = np.abs(self.fifo_residuals[sample_indexes, chan_indexes])
        order = np.argsort(amplitudes)[::-1]
        
        dtype_peak = [('sample_index', 'int32'), ('chan_index', 'int32'), ('peak_value', 'float32')]
        self.pending_peaks = np.zeros(sample_indexes.size, dtype=dtype_peak)
        self.pending_peaks['sample_index'] = sample_indexes
        self.pending_peaks['chan_index'] = chan_indexes
        self.pending_peaks['peak_value'] = amplitudes
        self.pending_peaks = self.pending_peaks[order]
        #~ print('re_detect_local_peak', self.pending_peaks.size)

    def select_next_peak(self):
        #~ print(len(self.pending_peaks))
        if len(self.pending_peaks)>0:
            sample_ind, chan_ind, ampl = self.pending_peaks[0]
            self.pending_peaks = self.pending_peaks[1:]
            return sample_ind, chan_ind
        else:
            return LABEL_NO_MORE_PEAK, None

    def on_accepted_spike(self, sample_ind, cluster_idx, jitter):
        # remove spike prediction from fifo residuals
        #~ t1 = time.perf_counter()
        pos, pred = make_prediction_one_spike(sample_ind, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        #~ t2 = time.perf_counter()
        #~ print('    make_prediction_one_spike', (t2-t1)*1000)
        
        #~ t1 = time.perf_counter()
        self.fifo_residuals[pos:pos+self.peak_width_long, :] -= pred
        #~ t2 = time.perf_counter()
        #~ print('     self.fifo_residuals -', (t2-t1)*1000)

        # this prevent search peaks in the zone until next "reset_to_not_tested"
        #~ t1 = time.perf_counter()
        self.clean_pending_peaks_zone(sample_ind, cluster_idx)
        #~ t2 = time.perf_counter()
        #~ print('     self.clean_pending_peaks_zone -', (t2-t1)*1000)


    def clean_pending_peaks_zone(self, sample_ind, cluster_idx):
        # TODO test with sparse_mask_level3s!!!!!
        mask = self.sparse_mask_level1[cluster_idx, :]

        
        #~ t1 = time.perf_counter()
        #~ keep = np.zeros(self.pending_peaks.size, dtype='bool')
        #~ for i, peak in enumerate(self.pending_peaks):
            #~ in_zone = mask[peak['chan_index']] and \
                                #~ (peak['sample_index']+self.n_left)<sample_ind and \
                                #~ sample_ind<(peak['sample_index']+self.n_right)
            #~ keep[i] = not(in_zone)
        
        peaks = self.pending_peaks
        in_zone = mask[peaks['chan_index']] &\
                            ((peaks['sample_index']+self.n_left)<sample_ind) & \
                            ((peaks['sample_index']+self.n_right)>sample_ind)
        keep = ~ in_zone
        #~ t2 = time.perf_counter()
        #~ print('     clean_pending_peaks_zone loop', (t2-t1)*1000)
        
        self.pending_peaks = self.pending_peaks[keep]
        
        #~ print('clean_pending_peaks_zone', self.pending_peaks.size)
    
    def set_already_tested(self, sample_ind, peak_chan):
        self.mask_already_tested[sample_ind, peak_chan] = True

    def reset_to_not_tested(self, good_spikes):
        for spike in good_spikes:
            # each good spike can remove from
            cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
            chan_mask = self.sparse_mask_level1[cluster_idx, :]
            self.mask_already_tested[spike.index + self.n_left_long:spike.index + self.n_right_long][:, chan_mask] = False
        
        self.re_detect_local_peak()
        

    def get_no_label_peaks(self):
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        nolabel_indexes, chan_indexes = np.nonzero(mask)
        #~ nolabel_indexes, chan_indexes = np.nonzero(~self.mask_not_already_tested)
        
        nolabel_indexes += self.n_span
        nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
        bad_spikes['index'] = nolabel_indexes
        bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        return bad_spikes

    def get_best_template(self, left_ind, chan_ind):

        full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        
        centers0 = self.catalogue['centers0']
        projections = self.catalogue['projections']

        strict_low = self.catalogue['boundaries'][:, 0]
        strict_high = self.catalogue['boundaries'][:, 1]
        flexible_low = self.catalogue['boundaries'][:, 2]
        flexible_high = self.catalogue['boundaries'][:, 3]
        
        
        n = centers0.shape[0]
        flat_waveform = full_waveform.flatten()
        flat_centers0 = centers0.reshape(n, -1)
        
        #~ scalar_products = np.zeros(n, dtype='float32')
        #~ for i in range(n):
            #~ sp = np.sum((flat_waveform - flat_centers0[i, :]) * projections[i, :])
            #~ scalar_products[i] = sp
        #~ scalar_products = np.sum((flat_waveform[np.newaxis, :] - flat_centers0[:, :]) * projections[:, :], axis=1)
        #~ print(scalar_products)
        
        #~ t1 = time.perf_counter()
        scalar_products = numba_sparse_scalar_product(self.fifo_residuals, left_ind, centers0, projections, chan_ind,
                    self.sparse_mask_level1, )
        #~ t2 = time.perf_counter()
        #~ print('numba_sparse_scalar_product', (t2-t1)*1000)

        #~ print(scalar_products)
        
        
        possible_idx, = np.nonzero((scalar_products < strict_high) & (scalar_products > strict_low))
        #~ possible_idx, = np.nonzero((scalar_products < flexible_high) & (scalar_products > flexible_low))
        
        #~ print('possible_idx', possible_idx)
        #~ print('scalar_products[possible_idx]', scalar_products[possible_idx])
        
        
        #~ do_plot = False
        if len(possible_idx) == 1:
            extra_idx = None
            candidates_idx =possible_idx
        elif len(possible_idx) == 0:
            #~ extra_idx, = np.nonzero((np.abs(scalar_products) < 0.5))
            extra_idx, = np.nonzero((scalar_products < flexible_high) & (scalar_products > flexible_low))
            #~ if len(extra_idx) ==0:
                # give a try to very far ones.
                #~ extra_idx, = np.nonzero((np.abs(scalar_products) < 1.))
                #~ print('extra_idx', extra_idx)
            #~ if len(extra_idx) ==0:
                #~ candidates_idx = []
            #~ else:
                #~ candidates_idx = extra_idx
            candidates_idx = extra_idx
            #~ candidates_idx =possible_idx
            #~ pass
        elif len(possible_idx) > 1 :
            extra_idx = None
            candidates_idx = possible_idx
        
        debug_plot_change = False
        if len(candidates_idx) > 0:
            #~ t1 = time.perf_counter()
            candidates_idx = np.array(candidates_idx, dtype='int64')
            common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
            shift_scalar_product, shift_distance = numba_explore_best_shift(self.fifo_residuals, left_ind, self.catalogue['centers0'],
                            self.catalogue['projections'], candidates_idx, self.maximum_jitter_shift, common_mask, self.sparse_mask_level1)
            #~ i0, i1 = np.unravel_index(np.argmin(np.abs(shift_scalar_product), axis=None), shift_scalar_product.shape)
            i0, i1 = np.unravel_index(np.argmin(shift_distance, axis=None), shift_distance.shape)
            #~ best_idx = candidates_idx[i0]
            shift = self.shifts[i1]
            cluster_idx = candidates_idx[i0]
            final_scalar_product = shift_scalar_product[i0, i1]
            #~ t2 = time.perf_counter()
            #~ print('numba_explore_best_shift', (t2-t1)*1000)



            
            
            #~ print('shift', shift)
            #~ print('cluster_idx', cluster_idx)
            #~ print('final_scalar_product', final_scalar_product)
            
            
            if np.abs(shift) == self.maximum_jitter_shift:
                cluster_idx = None
                shift = None
                final_scalar_product = None
                #~ print('maximum_jitter_shift >> cluster_idx = None ')
                #~ do_plot = True
            #~ i0_bis, i1_bis = np.unravel_index(np.argmin(np.abs(shift_scalar_product), axis=None), shift_scalar_product.shape)
            #~ if i0 != i0_bis:
                
                #~ debug_plot_change = True
                #~ print('Warning')
                #~ print(possible_idx)
                #~ print(shift_scalar_product)
                #~ print(shift_distance)
            
            
            
            
            #~ if best_idx != cluster_idx:
                #~ print('*'*50)
                #~ print('best_idx != cluster_idx', best_idx, cluster_idx)
                #~ print('*'*50)
                #~ cluster_idx = best_idx
                #~ debug_plot_change = True
        else:
            cluster_idx = None
            shift = None
            final_scalar_product = None
            
            #~ import matplotlib.pyplot as plt
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.shifts, shift_scalar_product.T)
            #~ plt.show()
            
        
        #~ print('ici',)
        

        # DEBUG OMP
        #~ from sklearn.linear_model import orthogonal_mp_gram
        #~ from sklearn.linear_model import OrthogonalMatchingPursuit
        #~ n_nonzero_coefs = 2
        #~ omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
        #~ X = self.catalogue['centers0'].reshape(self.catalogue['centers0'].shape[0], -1).T
        #~ waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:].flatten()
        #~ y = waveform
        #~ omp.fit(X, y)
        #~ coef = omp.coef_
        #~ idx_r, = coef.nonzero()
        #~ cluster_idx_omp = np.argmin(np.abs(coef - 1))
        
        
        #~ if cluster_idx_omp != cluster_idx and coef[cluster_idx_omp] > 0.5:
        #~ if True:
        if False:
            
        
        
        
        #~ if cluster_idx in (3,6):
        #~ if do_plot:
        #~ if False:
        #~ if final_scalar_product is not None and np.abs(final_scalar_product) > 0.5:
            
        #~ if True:
        #~ if len(possible_idx) != 1:
        #~ if len(possible_idx) > 1:
        #~ if len(candidates_idx) > 1:
        
        #~ if 7 in possible_idx or  cluster_idx == 7:
        #~ if cluster_idx not in possible_idx and len(possible_idx) > 0:
        #~ if debug_plot_change:
            
            import matplotlib.pyplot as plt
            
            print()
            print('best cluster_idx', cluster_idx)
            print('possible_idx', possible_idx)
            print('extra_idx', extra_idx)
            print(scalar_products[possible_idx])
            print(strict_high[possible_idx])
            
            print('cluster_idx_omp', cluster_idx_omp)

            
            fig, ax = plt.subplots()
            ax.plot(coef)
            if cluster_idx is not None:
                ax.axvline(cluster_idx)
            ax.set_title(f'{cluster_idx} omp {cluster_idx_omp}')
            #~ plt.show()

            
            
            
            
            fig, ax = plt.subplots()
            shift2 = 0 if shift is None else shift
            full_waveform2 = self.fifo_residuals[left_ind+shift2:left_ind+shift2+self.peak_width,:]
            
            ax.plot(full_waveform2.T.flatten(), color='k')
            if shift !=0 and shift is not None:
                ax.plot(full_waveform.T.flatten(), color='grey', ls='--')
            
            for idx in candidates_idx:
                ax.plot(self.catalogue['centers0'][idx, :].T.flatten(), color='m')
            
            ax.plot(self.catalogue['centers0'][cluster_idx_omp, :].T.flatten(), color='y')
            
            
            
                
            if cluster_idx is not None:
                ax.plot(self.catalogue['centers0'][cluster_idx, :].T.flatten(), color='c', ls='--')
            ax.set_title(f'best {cluster_idx} shift {shift} possible_idx {possible_idx}')
            
            if shift is not None:
                fig, ax = plt.subplots()
                #~ ax.plot(self.shifts, np.abs(shift_scalar_product).T)
                ax.plot(self.shifts, shift_scalar_product.T)
                ax.axhline(0)
                

                fig, ax = plt.subplots()
                ax.plot(self.shifts, np.abs(shift_distance).T)
            
            plt.show()
        
        
        best_template_info = {'nb_candidate' : len(candidates_idx), 'final_scalar_product':final_scalar_product}
        
        
        return cluster_idx, shift, best_template_info


    def accept_tempate(self, left_ind, cluster_idx, jitter, best_template_info):
        if jitter is None:
            # this must have a jitter
            jitter = 0
        
        #~ if np.abs(jitter) > (self.maximum_jitter_shift - 0.5):
            #~ return False
        
        strict_low = self.catalogue['boundaries'][:, 0]
        strict_high = self.catalogue['boundaries'][:, 1]
        flexible_low = self.catalogue['boundaries'][:, 2]
        flexible_high = self.catalogue['boundaries'][:, 3]
        
        
        #~ flat_waveform = full_waveform.flatten()
        #~ sp2 = np.sum((flat_waveform - centers0[cluster_idx, :].flatten()) * projections[cluster_idx, :])
        sp = best_template_info['final_scalar_product']
        nb_candidate = best_template_info['nb_candidate']
        
        if nb_candidate == 1:
            
            #~ accept_template = strict_low[cluster_idx] < sp < strict_high[cluster_idx]
            accept_template = flexible_low[cluster_idx] < sp < flexible_high[cluster_idx]
            
        else:
            accept_template = flexible_low[cluster_idx] < sp < flexible_high[cluster_idx]
        
                
            # waveform L2 on mask
            #~ full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            #~ wf = full_waveform[:, mask]
            
            # prediction with interpolation
            #~ _, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue, long=False)
            #~ pred_wf = pred_wf[:, mask]
        
            #~ dist = (pred_wf - wf) ** 2
            
            
            # criteria per channel
            #~ residual_nrj_by_chan = np.sum(dist, axis=0)
            #~ wf_nrj = np.sum(wf**2, axis=0)
            #~ weight = self.weight_per_template_dict[cluster_idx]
            #~ crietria_weighted = (wf_nrj>residual_nrj_by_chan).astype('float') * weight
            #~ accept_template = np.sum(crietria_weighted) >= 0.7 * np.sum(weight)
            
            # criteria per sample
            #~ dist * np.abs(pred_wf) < 
            #~ dist_w = dist / np.abs(pred_wf)
            #~ gain = (dist < wf**2).astype('float') * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
            #~ gain = (wf / pred_wf - 1) * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
            #~ gain = (pred_wf**2 / wf**1 - 1) * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
            #~ accept_template = np.sum(gain) > 0.8
            #~ accept_template = np.sum(gain) > 0.7
            #~ accept_template0 = np.sum(gain) > 0.6
            #~ accept_template = np.sum(gain) > 0.5
            
            # criteria max residual
            #~ max_res = np.max(np.abs(pred_wf - wf))
            #~ max_pred = np.max(np.abs(pred_wf))
            #~ accept_template1 = max_pred > max_res
            
            
            

            
            
            
            #~ accept_template = False
            
            # debug
            #~ limit_sp =self.catalogue['sp_normed_limit'][cluster_idx, :]
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform * self.catalogue['template_weight'])
            #~ print('limit_sp', limit_sp, 'sp', sp)
            
            
            
            #~ accept_template = False
            #~ immediate_accept = False
            
            # DEBUG always refuse!!!!!
            #~ accept_template = False
        
        
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        
        # debug
        #~ if label == 13:
            #~ if accept_template and not immediate_accept:
                #~ accept_template = False
        
        # debug
        #~ if label == 13:
            #~ if not hasattr(self, 'count_accept'):
                #~ self.count_accept = {}
                #~ self.count_accept[label] = {'accept_template':0, 'immediate_accept':0, 'not_accepted':0}
            
            #~ if accept_template:
                #~ self.count_accept[label]['accept_template'] += 1
                #~ if immediate_accept:
                    #~ self.count_accept[label]['immediate_accept'] += 1
            #~ else:
                #~ self.count_accept[label]['not_accepted'] += 1
            #~ print(self.count_accept)
            
        #~ if self._plot_debug:
        #~ if not accept_template and label in []:
        #~ if not accept_template:
        #~ if accept_template:
        #~ if True:
        if False:
            
        #~ if not immediate_accept:
        #~ if immediate_accept:
        #~ if immediate_accept:
        #~ if label == 7 and not accept_template:
        #~ if label == 7:
        #~ if label == 121:
        #~ if label == 5:
        #~ if nb_candidate > 1:
        
        #~ if label == 13 and accept_template and not immediate_accept:
        #~ if label == 13 and not accept_template:
            
        #~ if label in (7,9):
        #~ nears = np.array([ 5813767,  5813767, 11200038, 11322540, 14989650, 14989673, 14989692, 14989710, 15119220, 15830377, 16138346, 16216666, 17078883])
        #~ print(np.abs((left_ind - self.n_left) - nears))
        #~ print(np.abs((left_ind - self.n_left) - nears) < 2)
        #~ if label == 5 and np.any(np.abs((left_ind - self.n_left) - nears) < 50):
            
            #~ if immediate_accept:
            
            import matplotlib.pyplot as plt
            
            mask = self.sparse_mask_level2[cluster_idx]
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            wf = full_waveform[:, mask]
            _, pred_waveform = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue, long=False)
            pred_wf = pred_waveform[:, mask]
    
            if accept_template:
                color = 'g'
            else:
                color = 'r'
            
            #~ if accept_template:
                #~ if immediate_accept:
                    #~ color = 'g'
                #~ else:
                    #~ color = 'c'
            #~ else:
                #~ color = 'r'
            
            #~ if not immediate_accept:
                #~ fig, ax = plt.subplots()
                #~ ax.plot(gain.T.flatten(), color=color)
                #~ ax.set_title('{}'.format(np.sum(gain)))

            #~ fig, ax = plt.subplots()
            #~ ax.plot(feat_centroids.T, alpha=0.5)
            #~ ax.plot(feat_waveform, color='k')

            fig, ax = plt.subplots()
            ax.plot(full_waveform.T.flatten(), color='k')
            ax.plot(pred_waveform.T.flatten(), color=color)
            
            l0, l1 = strict_low[cluster_idx], strict_high[cluster_idx]
            l2, l3 = flexible_low[cluster_idx], flexible_high[cluster_idx]
            title = f'{cluster_idx} {sp:0.3f} lim [{l0:0.3f} {l1:0.3f}] [{l2:0.3f} {l3:0.3f}] {nb_candidate}'
            ax.set_title(title)
                
            #~ fig, ax = plt.subplots()
            #~ ax.plot(wf.T.flatten(), color='k')
            #~ ax.plot(pred_wf.T.flatten(), color=color)
            
            #~ ax.plot( wf.T.flatten() - pred_wf.T.flatten(), color=color, ls='--')
            
            print()
            print('cluster_idx',cluster_idx, 'accept_template', accept_template)
            #~ print(distance, self.distance_limit[cluster_idx])
            #~ print('distance', distance, distance2, 'limit_distance', self.distance_limit[cluster_idx])

            #~ limit_sp =self.catalogue['sp_normed_limit'][cluster_idx, :]
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform * self.catalogue['template_weight'])
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform)
            #~ print('limit_sp', limit_sp, 'sp', sp)
            
            #~ if not immediate_accept:
                #~ print('np.sum(gain)', np.sum(gain))


            #~ fig, ax = plt.subplots()
            #~ res = wf - pred_wf
            #~ count, bins = np.histogram(res, bins=150, weights=np.abs(pred_wf))
            #~ ax.plot(bins[:-1], count)
            #~ plt.show()

            
            
            #~ if distance2 >= self.distance_limit[cluster_idx]:
                #~ print(crietria_weighted, weight)
                #~ print(np.sum(crietria_weighted),  np.sum(weight))
            
            #~ ax.plot(full_wf0.T.flatten(), color='y')
            #~ ax.plot( full_wf.T.flatten() - full_wf0.T.flatten(), color='y')
            
            #~ ax.set_title('not accepted')
            plt.show()
        
        return accept_template
    
    
    def _plot_after_inner_peeling_loop(self):
        pass

    def _plot_before_peeling_loop(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_sigs = self.fifo_residuals.copy()
        self._plot_sigs_before = plot_sigs
        #~ chan_order = np.argsort(self.channel_distances[0, :])
        
        for c in range(self.nb_channel):
        #~ for c in chan_order:
            plot_sigs[:, c] += c*30
        
        ax.plot(plot_sigs, color='k')

        ax.axvline(self.fifo_size - self.n_right_long, color='r')
        ax.axvline(-self.n_left_long, color='r')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        sample_inds, chan_inds= np.nonzero(mask)
        sample_inds += self.n_span
        
        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        ax.set_title(f'nb peak {sample_inds.size}')
        
        #~ plt.show()
        
    
    def _plot_label_unclassified(self, left_ind, peak_chan, cluster_idx, jitter):
        return
        import matplotlib.pyplot as plt
        #~ print('LABEL UNCLASSIFIED', left_ind, cluster_idx)
        fig, ax = plt.subplots()
        
        wf = self.fifo_residuals[left_ind:left_ind+self.peak_width, :]
        wf0 = self.catalogue['centers0'][cluster_idx, :, :]
        
        ax.plot(wf.T.flatten(), color='b')
        #~ ax.plot(wf0.T.flatten(), color='g')
        
        ax.set_title(f'label_unclassified {left_ind-self.n_left} {cluster_idx} chan{peak_chan}')
        
        ax.axvline(peak_chan*self.peak_width-self.n_left)
        
        plt.show()

    def _plot_after_peeling_loop(self, good_spikes):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_sigs = self.fifo_residuals.copy()
        
        
        for c in range(self.nb_channel):
            plot_sigs[:, c] += c*30
        ax.plot(plot_sigs, color='k')
        
        ax.plot(self._plot_sigs_before, color='b')
        
        ax.axvline(self.fifo_size - self.n_right_long, color='r')
        ax.axvline(-self.n_left_long, color='r')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        sample_inds, chan_inds= np.nonzero(mask)
        sample_inds += self.n_span
        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        
        
        
        good_spikes = np.array(good_spikes, dtype=_dtype_spike)
        pred = make_prediction_signals(good_spikes, self.internal_dtype, plot_sigs.shape, self.catalogue, safe=True)
        plot_pred = pred.copy()
        for c in range(self.nb_channel):
            plot_pred[:, c] += c*30
        
        ax.plot(plot_pred, color='m')
        
        plt.show()





