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

_dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]


LABEL_BAD_PREDICTION = -10
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
        self.dataio = dataio
        
    def change_params(self, catalogue=None,
                                                relative_threshold=7,
                                                peak_span=0.0005,
                                                n_peel_level=2,
                                                chunk_size=1024,
                                                
                                                ):
        assert catalogue is not None
        self.catalogue = catalogue
        
        self.relative_threshold=relative_threshold
        self.peak_span=peak_span
        self.n_peel_level = n_peel_level
    
    def process_one_chunk(self,  pos, sigs_chunk, seg_num):
        pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        if preprocessed_chunk is  None:
            return
        
        self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num, i_start=pos2-preprocessed_chunk.shape[0],
                        i_stop=pos2, signal_type='processed')
        
        
        residual = preprocessed_chunk.copy()
        
        all_spikes = []
        
        for level in range(self.n_peel_level):
            #detect peaks
            n_peaks, chunk_peaks = self.peakdetectors[level].process_data(pos2, residual)
            if chunk_peaks is  None:
                chunk_peaks =np.array([], dtype='int64')
            
            # relation between inside chunk index and abs index
            shift = pos2-residual.shape[0]
            
            peak_pos2 = chunk_peaks - shift
            
            spikes  = classify_and_align(peak_pos2, residual, self.catalogue)
            #~ print(spikes)
            good_spikes = spikes[spikes['label']>=0]
            
            #~ print(good_spikes)
            
            prediction = make_prediction_signals(good_spikes, residual.dtype, residual.shape, self.catalogue)
            residual -= prediction
            
            ###
            #DEBUG
            #~ N=sigs_chunk.shape[1]
            #~ colors = sns.color_palette('husl', len(self.catalogue['cluster_labels']))
            #~ fig, axs = plt.subplots(nrows=N, sharex=True, sharey=True, )
            #~ for ii, k in enumerate(self.catalogue['cluster_labels']):
                #~ for iii in range(N):
                    #~ axs[iii].plot(self.catalogue['centers0'][ii, :, iii], color=colors[ii], label='{}'.format(k))
            #~ axs[0].legend()
            #~ fig, axs = plt.subplots(nrows=N, sharex=True, sharey=True, )
            #~ for iii in range(N):
                #~ axs[iii].plot(prediction[:, iii], color='m')
                #~ axs[iii].plot(preprocessed_chunk[:, iii], color='g')
                #~ axs[iii].plot(residual[:, iii], color='y')
                #~ axs[iii].plot(peak_pos2, preprocessed_chunk[peak_pos2, iii], color='m', marker='o', ls='None')
            #~ plt.show()
            #ENDDEBUG
            ###
            
            # for output
            good_spikes['index'] += shift
            all_spikes.append(good_spikes)
            
            if level == self.n_peel_level-1:
                bad_spikes = spikes[spikes['label']<0]
                bad_spikes['index'] += shift
                all_spikes.append(bad_spikes)
        
        return np.concatenate(all_spikes)
            
    
    
    def initialize_loop(self, chunksize=1024,
                                            signalpreprocessor_engine='signalpreprocessor_numpy',
                                            peakdetector_engine='peakdetector_numpy',
                                            internal_dtype='float32'):
        
        self.chunksize = chunksize
        self.dataio.reset_processed_signals(dtype=internal_dtype)
        self.dataio.reset_spikes(dtype=_dtype_spike)
        
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[signalpreprocessor_engine]
        self.signalpreprocessor = SignalPreprocessor_class(self.dataio.sample_rate, self.dataio.nb_channel, chunksize, self.dataio.dtype)
        
        #there is one peakdetectior by level because each one have
        # its own ringbuffer for each residual level
        PeakDetector_class = peakdetector.peakdetector_engines[peakdetector_engine]
        self.peakdetectors = []
        for level in range(self.n_peel_level):
            self.peakdetectors.append(PeakDetector_class(self.dataio.sample_rate, self.dataio.nb_channel, chunksize, internal_dtype))
        
    def run_loop(self, seg_num=0, duration=60.):
        
        length = int(duration*self.dataio.sample_rate)
        length -= length%self.chunksize
        #initialize engines
        
        p = dict(self.catalogue['params_signalpreprocessor'])
        p['normalize'] = True
        p['signals_medians'] = self.catalogue['signals_medians']
        p['signals_mads'] = self.catalogue['signals_mads']
        self.signalpreprocessor.change_params(**p)
        
        for level in range(self.n_peel_level):
            self.peakdetectors[level].change_params(**self.catalogue['params_peakdetector'])
        
        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chunksize=self.chunksize, i_stop=length,
                                                    signal_type='initial', return_type='raw_numpy')
        for pos, sigs_chunk in iterator:
            spikes = self.process_one_chunk(pos, sigs_chunk, seg_num)
            if spikes is not None and spikes.size>0:
                self.dataio.append_spikes(seg_num=seg_num, spikes=spikes)
        
        
    def finalize_loop(self):
        self.dataio.flush_processed_signals()
        self.dataio.flush_spikes()




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
        #~ waveform = waveforms[i,:,:]
        if ind+width>=residual.shape[0]:
            # too near right limits no label
            spikes['label'][i] = LABEL_RIGHT_LIMIT
            continue
        elif ind<0:
            #TODO fix this
            # too near left limits no label
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
            #~ print('shift', shift)
            ind = ind + shift
            if ind+width>=residual.shape[0]:
                spikes['label'][i] = LABEL_RIGHT_LIMIT
                continue
            elif ind<0:
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
        return LABEL_BAD_PREDICTION, 0.
    

def make_prediction_signals(spikes, dtype, shape, catalogue):
    #~ n_left, peak_width, 
    
    prediction = np.zeros(shape, dtype=dtype)
    for i in range(spikes.size):
        k = spikes[i]['label']
        if k<0: continue
        
        cluster_idx = np.nonzero(catalogue['cluster_labels']==k)[0][0]
        #~ print('make_prediction_signals', 'k', k, 'cluster_idx', cluster_idx)
        
        
        wf0 = catalogue['centers0'][cluster_idx,:,:]
        wf1 = catalogue['centers1'][cluster_idx,:,:]
        wf2 = catalogue['centers2'][cluster_idx]
        
        jitter = spikes[i]['jitter']
        #TODO find better interpolation here
        #~ pred = wf0 +jitter*wf1 + jitter**2/2*wf2
        pred = wf0
        
        pos = spikes[i]['index'] + catalogue['n_left']
        if pos>0 and  pos+catalogue['peak_width']<shape[0]:
            prediction[pos:pos+catalogue['peak_width'], :] += pred
        
    return prediction

