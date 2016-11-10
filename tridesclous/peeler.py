import os
import json
from collections import OrderedDict
import time

import numpy as np
import scipy.signal




class Peeler:
    """
    The peeler is core of online spike sorting.
    
    Take as input preprocess data by chunk.
    Detect peak on it.
    For each peak classify and detect jitter.
    With all peak/jitters create a prediction.
    Substract the prediction until there is no peak or unknown cluster.
    
    
    """
    def __init__(self, dataio, initial_catalogue):
        self.dataio = dataio
        self.initial_catalogue = initial_catalogue
        
        
    def change_params(self, catalogue=None,n_left=-20, n_right=30, n_peel_level=2):
        self.catalogue = catalogue
        self.n_left = n_left
        self.n_right = n_right
        
        self.n_peel_level = n_peel_level
        
    def process_data(self,  pos, preprocessed_chunk):
        #~ data = data.astype(self.output_dtype)
        
        residual = preprocessed_chunk.copy()
        
        all_spikes = []
        
        for level in range(self.n_peel_level):
        
        
            #detect peaks
            n_peaks, chunk_peaks = peakdetector.process_data(pos2, preprocessed_chunk)
            if chunk_peaks is  None:
                chunk_peaks =np.array([], dtype='int64')
            
            for peak_pos, chunk_waveforms in waveformextractor.new_peaks(pos2, preprocessed_chunk, chunk_peaks):

                #~ spike_pos, jitters, labels     = classify_and_align(waveforms, peak_pos, residuals)
                spikes  = classify_and_align(waveforms, peak_pos, residuals)
                
                good_spikes = spikes[spikes['label']!=-1]
                
                yield good_spikes
                
                #~ self.spike_labels[self.level] = labels
                #~ self.spike_jitters[self.level] = jitters
                #~ self.spike_pos[self.level] = spike_pos
                
                prediction = make_prediction_signals(spike_pos, jitters, labels)
                residual -= prediction
        
        

_dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]

def classify_and_align(waveforms, peak_pos, residuals):
    
    
    spikes = np.zeros(waveforms.shape[0], dtype=_dtype_spike)
    spikes['pos'] = peak_pos

    #~ jitters = np.empty(waveforms.shape[0], dtype = 'float64')
    #~ labels = np.empty(waveforms.shape[0], dtype = int)
    for i in range(waveforms.shape[0]):
        wf = waveforms[i,:]
        label, jitter = self.estimate_one_jitter(wf)
        
        # if more than one sample of jitter
        # then we take a new wf at the good place and do estimate again
        # take it if better
        if np.abs(jitter) > 0.5 and label !=-1:
            prev_label, prev_jitter = label, jitter
            peak_pos[i] -= int(np.round(jitter))
            chunk = cut_chunks(residuals.values, np.array([ peak_pos[i]+self.n_left], dtype = 'int32'),
                            -self.n_left + self.n_right )
            wf = waveforms[i,:] = chunk[0,:].reshape(-1)
            new_label, new_jitter = self.estimate_one_jitter(wf)
            if np.abs(new_jitter)<np.abs(prev_jitter):
                label, jitter1 = new_label, new_jitter
        
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
    

def estimate_one_jitter(wf, all_center, cluster_labels, catalogue):
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
    cluster_idx = np.argmin(np.sum((all_center-wf)**2, axis = 1))
    k = cluster_labels[cluster_idx]
    
    wf0 = catalogue[k]['center']
    wf1 = catalogue[k]['centerD']
    wf2 = catalogue[k]['centerDD']
    
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
    

def make_prediction_signals(spikes, n_left, peak_width, dtype, shape, catalogue):
    prediction = np.zeros(shape, dtype=dtype)
    for i in range(spikes.size):
        k = spikes[i]['label']
        if k<0: continue
        
        #TODO find better interpolation here
        wf0 = self.catalogue[k]['center']
        wf1 = self.catalogue[k]['centerD']
        wf2 = self.catalogue[k]['centerDD']
        pred = wf0 + jitters[i]*wf1 + jitters[i]**2/2*wf2
        
        pos = spikes[i]['index'] + n_left
        #TODO fix swapaxes here (change since 0.1.0
        prediction[pos:pos+peak_width, :] = pred.reshape(self.nb_channel, -1).transpose()
        
    return prediction

    
