import numpy as np
import pandas as pd

from .waveformextractor import  WaveformExtractor, cut_chunks
from .peakdetector import PeakDetector






class Peeler:
    """
    Class that 'peel' spike on signal given a catalogue (centre + first and second derivative).
    
    This is a recursive procedure to resolve superposition.
    
    Note that the first peel (level=0) should find the majority of spikes and this result shoudl give
    results similar to classical spikes sorting procedure.
    
    
    For the first peel peak_pos0 and waveforms0 can be option
    
    
    Arguments
    ---------------
    
    signals: pd.DataFrame
        signals, must the normed signals or the signals that correspond to catalogue.
    
     n_left, n_right:
        The good limits
     
    
    
    """
    def __init__(self, signals, catalogue,  n_left, n_right,
                            threshold=-4, peak_sign = '-', n_span = 2,
                            peak_pos0 = None, waveforms0 = None):
        self.signals = signals
        self.catalogue = catalogue
        self.n_left = n_left
        self.n_right = n_right
        self.threshold = threshold
        self.peak_sign = peak_sign
        self.n_span = n_span
        
        
        
        self.cluster_labels = np.array(list(catalogue.keys()))
        self.all_center = np.array([catalogue[k]['center'] for k in self.cluster_labels])
        
        # level of peel alredy done
        self.level = 0
        self.residuals = signals.copy()
        
        #for each peels level
        self.spike_labels = {}
        self.spike_jitters = {}
        
        self.peak_pos0 = peak_pos0
        self.waveforms0 = waveforms0
    



    def estimate_one_jitter(self, wf):
        """
        Estimate the jitter for one peak given its waveform
        
        for best reading (at for me SG):
          * wf = the wafeform of the peak
          * k = cluster label of the peak
          * wf0, wf1, wf2 : center of catalogue[k] + first + second derivative
          * jitter0 : jitter estimation at order 0
          * jitter1 : jitter estimation at order 1
          * h0_norm2: error at order0
          * h1_norm2: error at order1
        """

        cluster_idx = np.argmin(np.sum((self.all_center-wf)**2, axis = 1))
        k = self.cluster_labels[cluster_idx]
        
        wf0 = self.catalogue[k]['center']
        wf1 = self.catalogue[k]['centerD']
        wf2 = self.catalogue[k]['centerDD']
        
        wf0_norm2 = wf0.dot(wf0)
        wf1_norm2 = wf1.dot(wf1)
        wf2_norm2 = wf2.dot(wf2)
        wf1_dot_wf2 = wf1.dot(wf2)
        
        h = wf - wf0
        h0_norm2 = h.dot(h)
        h_dot_wf1 = h.dot(wf1)
        jitter0 = h_dot_wf1/wf1_norm2
        h1_norm2 = np.sum((h-jitter0*wf1)**2)
        
        if h0_norm2 > h1_norm2:
            h_dot_wf2 = np.dot(h,wf2)
            rss_first = -2*h_dot_wf1 + 2*jitter0*(wf1_norm2 - h_dot_wf2) + 3*jitter0**2*wf1_dot_wf2 + jitter0**3*wf2_norm2
            rss_second = 2*(wf1_norm2 - h_dot_wf2) + 6*jitter0*wf1_dot_wf2 + 3*jitter0**2*wf2_norm2
            jitter1 = jitter0 - rss_first/rss_second
            h2_norm2 = np.sum((h-jitter1*wf1-jitter1**2/2*wf2)**2)
            if h1_norm2 <= h2_norm2:
                jitter1 = jitter0
        else:
            jitter1 = 0    
        
        return k, jitter1
        
    
    def classify_and_align(self, waveforms, peak_pos):
        """
        
        """

        jitters = np.empty(waveforms.shape[0])
        labels = np.empty(waveforms.shape[0])
        for i in range(waveforms.shape[0]):
            print('i', i, waveforms.shape[0])
            wf = waveforms[i,:]
            
            label, jitter1 = self.estimate_one_jitter(wf)
            print(label, jitter1)
            
            # if more than one sample of jitter
            # then we take a new wf at the good place and do estimate again
            #~ while np.abs(jitter1) > 0.5:
            if np.abs(jitter1) > 0.5:
                #~ peak_pos[i] -= int(np.round(jitter1))
                peak_pos[i] += int(np.round(jitter1))
                chunk = cut_chunks(self.residuals.values, np.array([ peak_pos[i]+self.n_left], dtype = int), -self.n_left + self.n_right )
                wf = waveforms[i,:] = chunk[0,:].reshape(-1)
                label, jitter1 = self.estimate_one_jitter(wf)
                print('   in while', label, jitter1)
            
            labels[i] = jitter1
            labels[i] = label
            
        return jitters, labels

   
   
    def peel(self):
        print('peel level=', self.level)
        if self.level==0 and self.peak_pos0 is not None and self.waveforms0 is not None:
            peak_pos = self.peak_pos0
            waveforms = self.waveforms0.values
        else:
            # detect peak and take waveform on residuals
            peakdetector = PeakDetector(self.signals)
            peak_pos = peakdetector.detect_peaks(threshold=self.threshold, peak_sign = self.peak_sign, n_span = self.n_span)
            
            #waveforms
            waveformextractor = WaveformExtractor(peakdetector, n_left=self.n_left, n_right=self.n_right)
            # in peeler n_left and n_rigth are th "good limit"
            waveforms = waveformextractor.long_waveforms.values
        
        jitters, labels = self.classify_and_align(waveforms, peak_pos)

        self.spike_labels[self.level] = labels
        self.spike_jitters[self.level] = jitters
        
        
        
        self.level += 1
   
   
                  
            
            
            
    
