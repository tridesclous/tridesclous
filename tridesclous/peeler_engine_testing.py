"""
Here implementation for testing new ideas of peeler.

"""

import time
import numpy as np

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
    
    def accept_tempate(self, left_ind, cluster_idx, jitter):

        if np.abs(jitter) > self.maximum_jitter_shift:
            return False
        
        
        shift = -int(np.round(jitter))
        jitter = jitter + shift
        left_ind = left_ind + shift
        new_left, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        
        mask = self.catalogue['sparse_mask'][cluster_idx]
        pred_wf = pred_wf[:, :][:, mask]

        # waveform L2 on mask
        waveform = self.fifo_residuals[new_left:new_left+self.peak_width,:]
        full_wf = waveform[:, :][:, mask]
        wf_nrj = np.sum(full_wf**2, axis=0)
        
        
        residual_nrj = np.sum((full_wf-pred_wf)**2, axis=0)
        
        # criteria per channel
        label = self.catalogue['cluster_labels'][cluster_idx]
        weight = self.weight_per_template[label]
        crietria_weighted = (wf_nrj>residual_nrj).astype('float') * weight
        accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        
        #DEBUG
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        #~ if label in (5, ):
            
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
                
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(full_wf.T.flatten(), color='b')
            #~ if accept_template:
                #~ ax.plot(pred_wf.T.flatten(), color='g')
            #~ else:
                #~ ax.plot(pred_wf.T.flatten(), color='r')
            
            #~ ax.plot((full_wf-pred_wf).T.flatten(), color='m')
            
            
            #~ plt.show()
            
        
        #~ #ENDDEBUG
        
        
        return accept_template

