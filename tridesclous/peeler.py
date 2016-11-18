import os
import json
from collections import OrderedDict
import time

import numpy as np
import scipy.signal


from . import signalpreprocessor
from . import  peakdetector
from . import waveformextractor

import matplotlib.pyplot as plt
import seaborn as sns

#~ try:
    #~ from tqdm import tqdm
    #~ HAVE_TQDM = True
#~ except ImportError:
    #~ HAVE_TQDM = False

#TODO: put this when finish
HAVE_TQDM = False

_dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]

LABEL_TRASH = -1
LABEL_UNSLASSIFIED = -10
LABEL_LEFT_LIMIT = -11
LABEL_RIGHT_LIMIT = -12
# good label are >=0


class Peeler:
    """
    The peeler is core of online spike sorting.
    
    Take as input preprocess data by chunk.
    Detect peak on it.
    For each peak classify and detect jitter.
    With all peak/jitters create a prediction.
    Substract the prediction until there is no peak or unknown cluster.
    
    
    """
    def __init__(self, dataio):
        #for online dataio is None
        self.dataio = dataio
        
    def change_params(self, catalogue=None, n_peel_level=2,chunksize=1024, 
                                        internal_dtype='float32', 
                                        signalpreprocessor_engine='signalpreprocessor_numpy',
                                        peakdetector_engine='peakdetector_numpy'):
        assert catalogue is not None
        self.catalogue = catalogue
        self.n_peel_level = n_peel_level
        self.chunksize = chunksize
        self.internal_dtype= internal_dtype
        self.signalpreprocessor_engine = signalpreprocessor_engine
        self.peakdetector_engine = peakdetector_engine
        
    
    def process_one_chunk(self,  pos, sigs_chunk):
        abs_head_index, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        #note abs_head_index is smaller than pos because prepcorcessed chunk
        # is late because of local filfilt in signalpreprocessor
        if preprocessed_chunk is  None:
            return
        
        local_size = preprocessed_chunk.shape[0]
        residual = preprocessed_chunk.copy()
        
        all_spikes = []
        
        for level in range(self.n_peel_level):
            #detect peaks
            n_peaks, chunk_peaks = self.peakdetectors[level].process_data(abs_head_index, residual)
            if chunk_peaks is  None:
                chunk_peaks =np.array([], dtype='int64')
            
            # relation between inside chunk index and abs index
            shift = abs_head_index - local_size
            
            local_index = chunk_peaks - shift
            spikes  = classify_and_align(local_index, residual, self.catalogue)
            good_spikes = spikes[spikes['label']>=0]
            prediction = make_prediction_signals(good_spikes, residual.dtype, residual.shape, self.catalogue)
            residual -= prediction

            # for output
            good_spikes['index'] += shift
            all_spikes.append(good_spikes)
        
        # append bad spike
        bad_spikes = spikes[spikes['label']==LABEL_UNSLASSIFIED]
        bad_spikes['index'] += shift
        all_spikes.append(bad_spikes)

        if self.prev_preprocessed_chunk is not None:
            #This is the tricky part, this peel spikes that are:
            #  * at right limit of previous chunk
            #  * at left limit of actual chunk
            # so construct a small chunk
            n_left = self.catalogue['n_left']
            n_right = self.catalogue['n_right']
            peak_width = self.catalogue['peak_width']
            #~ print()
            #~ print('Spike at limits')
            #~ print('abs_head_index', abs_head_index)
            #~ print('abs_head_index - local_size', abs_head_index-local_size)
            #~ print('n_left', n_left,'n_right', n_right)
            
            overlap_residual = np.concatenate([self.prev_preprocessed_chunk[-peak_width-2:, :], 
                                                                preprocessed_chunk[:peak_width+2]], axis=0)
            shift2 = abs_head_index - local_size - peak_width -2
            
            #~ print(overlap_residual.shape)
            #~ print('left abs', spikes[spikes['label']==LABEL_LEFT_LIMIT]['index']+shift)
            
            index_left = spikes[spikes['label']==LABEL_LEFT_LIMIT]['index'] +shift - shift2
            #~ print('right abs', self.prev_spikes_at_right_limit['index'])
            index_right = self.prev_spikes_at_right_limit['index'] - shift2
            index_limit = np.concatenate([index_left, index_right])
            #~ print('index_limit', index_limit)
            #~ print('shift2', shift2)
            for level in range(self.n_peel_level):
                #~ print('level', level)
                spikes_limit  = classify_and_align(index_limit, overlap_residual, self.catalogue)
                if spikes_limit.size>0:
                    #~ print('spikes_limit', spikes_limit)
                    assert np.all(spikes_limit['label']!=LABEL_LEFT_LIMIT), 'hop'#for debug
                    assert np.all(spikes_limit['label']!=LABEL_RIGHT_LIMIT), 'hop'#for debug
                    good_spikes_limit = spikes_limit[spikes_limit['label']>=0]
                    good_spikes_limit['index'] += shift2
                    all_spikes.append(good_spikes_limit)
                    # for next level
                    index_limit = good_spikes_limit[good_spikes_limit['label']==LABEL_UNSLASSIFIED]['index']
        
            bad_spikes_limit = spikes_limit[spikes_limit['label']==LABEL_UNSLASSIFIED]
            bad_spikes_limit['index'] += shift2
            all_spikes.append(bad_spikes_limit)
        
        
        # chunk and spike at right limit for next chunk
        self.prev_preprocessed_chunk = preprocessed_chunk
        self.prev_spikes_at_right_limit = spikes[spikes['label']==LABEL_RIGHT_LIMIT].copy()
        self.prev_spikes_at_right_limit['index'] += shift
        #~ print('for next chunk', self.prev_spikes_at_right_limit)
                
        #concatenate sort and count
        all_spikes = np.concatenate(all_spikes)
        #~ all_spikes = all_spikes[np.argsort(all_spikes['index'])]
        all_spikes = all_spikes.take(np.argsort(all_spikes['index']))
        self.total_spike += all_spikes.size
        
        return abs_head_index, preprocessed_chunk, self.total_spike, all_spikes
            
    
    
    def _initialize_before_each_segment(self, sample_rate=None, nb_channel=None, input_dtype=None):

        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[self.signalpreprocessor_engine]
        self.signalpreprocessor = SignalPreprocessor_class(sample_rate, nb_channel, self.chunksize, input_dtype)
        
        #there is one peakdetectior by level because each one have
        # its own ringbuffer for each residual level
        PeakDetector_class = peakdetector.peakdetector_engines[self.peakdetector_engine]
        self.peakdetectors = []
        for level in range(self.n_peel_level):
            self.peakdetectors.append(PeakDetector_class(sample_rate, nb_channel, self.chunksize, self.internal_dtype))

        p = dict(self.catalogue['params_signalpreprocessor'])
        p['normalize'] = True
        p['signals_medians'] = self.catalogue['signals_medians']
        p['signals_mads'] = self.catalogue['signals_mads']
        self.signalpreprocessor.change_params(**p)
        
        for level in range(self.n_peel_level):
            self.peakdetectors[level].change_params(**self.catalogue['params_peakdetector'])

        self.total_spike = 0
        self.prev_preprocessed_chunk = None
        self.prev_spikes_at_right_limit = np.zeros(0, dtype=_dtype_spike)
        
    def initialize_online_loop(self, sample_rate=None, nb_channel=None, input_dtype=None):
        self._initialize_before_each_segment(sample_rate=sample_rate, nb_channel=nb_channel, input_dtype=input_dtype)
    
    def run_offline_loop_one_segment(self, seg_num=0, duration=None):
        kargs = {}
        kargs['sample_rate'] = self.dataio.sample_rate
        kargs['nb_channel'] = self.dataio.nb_channel
        kargs['input_dtype'] = self.dataio.dtype
        self._initialize_before_each_segment(**kargs)
        
        if duration is not None:
            length = int(duration*self.dataio.sample_rate)
        else:
            length = self.dataio.get_segment_shape(seg_num)[0]
        length -= length%self.chunksize
                #initialize engines
        
        self.dataio.reset_processed_signals(seg_num=seg_num, dtype=self.internal_dtype)
        self.dataio.reset_spikes(seg_num=seg_num, dtype=_dtype_spike)

        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chunksize=self.chunksize, i_stop=length,
                                                    signal_type='initial', return_type='raw_numpy')
        if HAVE_TQDM:
            iterator = tqdm(iterable=iterator, total=length//self.chunksize)
        for pos, sigs_chunk in iterator:
            #~ print(pos, length, pos/length)
            sig_index, preprocessed_chunk, total_spike, spikes = self.process_one_chunk(pos, sigs_chunk)
            
            # save preprocessed_chunk to file
            # TODO optional ???
            self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num,
                        i_start=sig_index-preprocessed_chunk.shape[0], i_stop=sig_index,
                        signal_type='processed')
            
            if spikes is not None and spikes.size>0:
                self.dataio.append_spikes(seg_num=seg_num, spikes=spikes)

        self.dataio.flush_processed_signals(seg_num=seg_num)
        self.dataio.flush_spikes(seg_num=seg_num)

    def run_offline_all_segment(self):
        for seg_num in range(self.dataio.nb_segment):
            self.run_offline_loop_one_segment(seg_num=seg_num, duration=None)
    
    run = run_offline_all_segment



def classify_and_align(local_indexes, residual, catalogue):
    """
    local_indexes is index of peaks inside residual and not
    the absolute peak_pos. So time scaling must be done outside.
    
    """
    width = catalogue['peak_width']
    n_left = catalogue['n_left']
    spikes = np.zeros(local_indexes.shape[0], dtype=_dtype_spike)
    spikes['index'] = local_indexes

    for i, ind in enumerate(local_indexes+n_left):
        #~ print('classify_and_align', i, ind)
        #~ waveform = waveforms[i,:,:]
        if ind+width>=residual.shape[0]:
            # too near right limits no label
            #~ print('     LABEL_RIGHT_LIMIT', ind, width, ind+width, residual.shape[0])
            spikes['label'][i] = LABEL_RIGHT_LIMIT
            continue
        elif ind<0:
            #TODO fix this
            # too near left limits no label
            #~ print('     LABEL_LEFT_LIMIT', ind)
            spikes['label'][i] = LABEL_LEFT_LIMIT
            continue
        else:
            waveform = residual[ind:ind+width,:]
        
        label, jitter = estimate_one_jitter(waveform, catalogue)
        #~ print('label, jitter', label, jitter)
        
        # if more than one sample of jitter
        # then we try a peak shift
        # take it if better
        #TODO debug peak shift
        if np.abs(jitter) > 0.5 and label >=0:
            prev_ind, prev_label, prev_jitter = label, jitter, ind
            shift = -int(np.round(jitter))
            #~ print('classify_and_align shift', shift)
            ind = ind + shift
            if ind+width>=residual.shape[0]:
                #~ print('     LABEL_RIGHT_LIMIT avec shift')
                spikes['label'][i] = LABEL_RIGHT_LIMIT
                continue
            elif ind<0:
                #~ print('     LABEL_LEFT_LIMIT avec shift')
                spikes['label'][i] = LABEL_LEFT_LIMIT
                continue
            else:
                waveform = residual[ind:ind+width,:]
                new_label, new_jitter = estimate_one_jitter(waveform, catalogue)
                if np.abs(new_jitter)<np.abs(prev_jitter):
                    #~ print('keep shift')
                    label, jitter = new_label, new_jitter
                    spikes['index'][i] += shift
                else:
                    #~ print('no keep shift worst jitter')
                    pass
        
        spikes['jitter'][i] = jitter
        spikes['label'][i] = label
    
    #~ print(spikes)
    return spikes


def estimate_one_jitter(waveform, catalogue):
    """
    Estimate the jitter for one peak given its waveform
    
    Method proposed by Christophe Pouzat see:
    https://hal.archives-ouvertes.fr/hal-01111654v1
    http://christophe-pouzat.github.io/LASCON2016/SpikeSortingTheElementaryWay.html
    
    for best reading (at for me SG):
      * wf = the wafeform of the peak
      * k = cluster label of the peak
      * wf0, wf1, wf2 : center of catalogue[k] + first + second derivative
      * jitter0 : jitter estimation at order 0
      * jitter1 : jitter estimation at order 1
      * h0_norm2: error at order0
      * h1_norm2: error at order1
      * h2_norm2: error at order2
    """
    
    cluster_idx = np.argmin(np.sum(np.sum((catalogue['centers0']-waveform)**2, axis = 1), axis = 1))
    k = catalogue['cluster_labels'][cluster_idx]
    chan = catalogue['max_on_channel'][cluster_idx]
    #~ print('cluster_idx', cluster_idx, 'k', k, 'chan', chan)
    
    #~ return k, 0.

    wf0 = catalogue['centers0'][cluster_idx,: , chan]
    wf1 = catalogue['centers1'][cluster_idx,: , chan]
    wf2 = catalogue['centers2'][cluster_idx,: , chan]
    wf = waveform[:, chan]
    #~ print()
    #~ print(wf0.shape, wf.shape)
    #TODO put that in make_catalogue
    wf1_norm2 = wf1.dot(wf1)
    wf2_norm2 = wf2.dot(wf2)
    wf1_dot_wf2 = wf1.dot(wf2)
    
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
    
    if np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2):
        #prediction should be smaller than original (which have noise)
        return k, jitter1
    else:
        #otherwise the prediction is bad
        #~ print('bad prediction')
        return LABEL_UNSLASSIFIED, 0.
    

def make_prediction_signals(spikes, dtype, shape, catalogue):
    #~ n_left, peak_width, 
    
    prediction = np.zeros(shape, dtype=dtype)
    for i in range(spikes.size):
        k = spikes[i]['label']
        if k<0: continue
        
        cluster_idx = np.nonzero(catalogue['cluster_labels']==k)[0][0]
        #~ print('make_prediction_signals', 'k', k, 'cluster_idx', cluster_idx)
        
        # prediction with no interpolation
        #~ wf0 = catalogue['centers0'][cluster_idx,:,:]
        #~ pred = wf0
        
        # predict with tailor approximate with derivative
        #~ wf1 = catalogue['centers1'][cluster_idx,:,:]
        #~ wf2 = catalogue['centers2'][cluster_idx]
        #~ pred = wf0 +jitter*wf1 + jitter**2/2*wf2
        
        #predict with with precilputed splin
        #TODO debug this!!!!!
        #~ print()
        r = catalogue['subsample_ratio']
        int_jitter = int(spikes[i]['jitter']*r) + r//2
        
        #TODO this is wrong we should move index first
        int_jitter = max(int_jitter, 0)
        int_jitter = min(int_jitter, r-1)
        
        pred = catalogue['interp_centers0'][cluster_idx, int_jitter::r, :]
        #~ print(pred.shape)
        #~ print(int_jitter, spikes[i]['jitter'])
        
        pos = spikes[i]['index'] + catalogue['n_left']
        #~ print(prediction[pos:pos+catalogue['peak_width'], :].shape)
        if pos>0 and  pos+catalogue['peak_width']<shape[0]:
            prediction[pos:pos+catalogue['peak_width'], :] += pred
        
    return prediction

