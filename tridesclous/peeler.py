import os
import json
from collections import OrderedDict
import time

import numpy as np
import scipy.signal


from . import signalpreprocessor
from . import  peakdetector
from . import waveformextractor


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
        print('yep')
        pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        if preprocessed_chunk is  None:
            return
        
        self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num, i_start=pos2-preprocessed_chunk.shape[0],
                        i_stop=pos2, signal_type='processed')
        
        
        residual = preprocessed_chunk.copy()
        
        all_spikes = []
        
        print()
        print('pos2', pos2)
        for level in range(self.n_peel_level):
            print('level', level)
            
            #detect peaks
            n_peaks, chunk_peaks = self.peakdetectors[level].process_data(pos2, residual)
            if chunk_peaks is  None:
                chunk_peaks =np.array([], dtype='int64')
            
            print('n_peaks', n_peaks)
            
            # relation between inside chunk index and abs index
            shift = pos2-residual.shape[0]
            
            peak_pos2 = chunk_peaks - shift
            
            spikes  = classify_and_align(peak_pos2, residual, self.catalogue)
            
            good_spikes = spikes[spikes['label']!=-1]
            
            prediction = make_prediction_signals(good_spikes, residual.dtype, residual.shape, self.catalogue)
            residual -= prediction
            
            # for output
            good_spikes['index'] += shift
            #~ yield good_spikes
            all_spikes.append(good_spikes)
        return np.concatenate(all_spikes)
            
    
    
    def initialize_loop(self, chunksize=1024,
                                            signalpreprocessor_engine='signalpreprocessor_numpy',
                                            peakdetector_engine='peakdetector_numpy',
                                            internal_dtype='float32'):
        
        self.chunksize = chunksize
        self.dataio.reset_signals(signal_type='processed', dtype=internal_dtype)
        
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[signalpreprocessor_engine]
        self.signalpreprocessor = SignalPreprocessor_class(self.dataio.sample_rate, self.dataio.nb_channel, chunksize, self.dataio.dtype)
        
        #there is one peakdetectior by level because each one have its own ringbuffer
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
            #~ print(seg_num, pos, sigs_chunk.shape)
            spikes = self.process_one_chunk(pos, sigs_chunk, seg_num)
            print('spikes')
        
        self.dataio.flush_signals(seg_num=seg_num)

    def finalize_loop(self):
        pass
    



_dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]

def classify_and_align(peak_pos, residual, catalogue):
    """
    peak_pos is index of peaks inside residual and not
    the absolute peak_pos. So time scaling must be done outside.
    
    """
    
    width = catalogue['peak_width']
    waveforms = np.empty((peak_pos.size, width, residual.shape[1]), dtype = residual.dtype)
    for i, ind in enumerate(peak_pos):
        #TODO fix limits!!!!
        if ind+width>=residual.shape[1]:
            pass
        else:
            waveforms[i,:,:] = residual[ind:ind+width,:]
    
    spikes = np.zeros(waveforms.shape[0], dtype=_dtype_spike)
    spikes['index'] = peak_pos

    #~ jitters = np.empty(waveforms.shape[0], dtype = 'float64')
    #~ labels = np.empty(waveforms.shape[0], dtype = int)
    for i in range(waveforms.shape[0]):
        wf = waveforms[i,:]
        label, jitter = estimate_one_jitter(wf, catalogue)
        
        # if more than one sample of jitter
        # then we take a new wf at the good place and do estimate again
        # take it if better
        if np.abs(jitter) > 0.5 and label !=-1:
            #TODO
            pass
            #~ prev_label, prev_jitter = label, jitter
            #~ peak_pos[i] -= int(np.round(jitter))
            #~ chunk = cut_chunks(residual.values, np.array([ peak_pos[i]+self.n_left], dtype = 'int32'),
                            #~ -self.n_left + self.n_right )
            #~ wf = waveforms[i,:] = chunk[0,:].reshape(-1)
            #~ new_label, new_jitter = self.estimate_one_jitter(wf)
            #~ if np.abs(new_jitter)<np.abs(prev_jitter):
                #~ label, jitter1 = new_label, new_jitter
        
        spikes['jitter'][i] = jitter
        spikes['label'][i] = label
    
    
    return spikes
        #~ jitters[i] = jitter
        #~ labels[i] = label
    
    #~ keep = labels!=-1
    #~ labels = labels[keep]
    #~ jitters = jitters[keep]
    #~ spike_pos = peak_pos[keep]
    
    #~ return spike_pos, jitters, labels    
    

def estimate_one_jitter(wf, catalogue):
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
    
    cluster_idx = np.argmin(np.sum(np.sum((catalogue['centers0']-wf)**2, axis = 1), axis = 1))
    k = catalogue['cluster_labels'][cluster_idx]
    
    wf0 = catalogue['centers0'][cluster_idx]
    wf1 = catalogue['centers1'][cluster_idx]
    wf2 = catalogue['centers2'][cluster_idx]
    
    #TODO flatten everything in make_catalogue
    wf0 = wf0.T.flatten()
    wf1 = wf1.T.flatten()
    wf2 = wf2.T.flatten()
    wf = wf.T.flatten()
    
    #TODO put that in make_catalogue
    wf1_norm2 = wf1.dot(wf1)
    wf2_norm2 = wf2.dot(wf2)
    wf1_dot_wf2 = wf1.dot(wf2)
    
    h = wf - wf0
    h0_norm2 = h.dot(h)
    h_dot_wf1 = h.dot(wf1)
    jitter0 = h_dot_wf1/wf1_norm2
    h1_norm2 = np.sum((h-jitter0*wf1)**2)
    
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
    
    if np.sum(wf**2) > np.sum((wf-(wf0+jitter1*wf1+jitter1**2/2*wf2))**2):
        #prediction should be smaller than original (which have noise)
        return k, jitter1
    else:
        #otherwise the prediction is bad
        return -1, 0.    
    

def make_prediction_signals(spikes, dtype, shape, catalogue):
    #~ n_left, peak_width, 
    
    prediction = np.zeros(shape, dtype=dtype)
    for i in range(spikes.size):
        k = spikes[i]['label']
        if k<0: continue
        
        #TODO better
        cluster_idx = catalogue['cluster_labels'].tolist().index(k)
        
        #TODO find better interpolation here
        wf0 = catalogue['centers0'][cluster_idx]
        wf1 = catalogue['centers1'][cluster_idx]
        wf2 = catalogue['centers2'][cluster_idx]
        
        jitter = spikes[i]['jitter']
        #TODO better than this
        pred = wf0 +jitter*wf1 + jitter**2/2*wf2
        
        pos = spikes[i]['index'] + catalogue['n_left']
        #TODO fix swapaxes here (change since 0.1.0
        prediction[pos:pos+catalogue['peak_width'], :] = pred.reshape(-1, shape[1])
        
    return prediction

    
