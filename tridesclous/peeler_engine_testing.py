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


try:
    import numba
    HAVE_NUMBA = True
    from numba import jit


    @jit(nopython=True, parallel=True)
    def numba_loop_sparse_weigthed_dist(waveform, centers,  mask):
        nb_clus, width, nb_chan = centers.shape
        
        rms_waveform_channel = np.sum(waveform**2, axis=0)#.astype('float32')
        waveform_distance = np.zeros((nb_clus,), dtype=np.float32)
        
        for clus in range(nb_clus):
            sum = 0
            for c in range(nb_chan):
                if mask[clus, c]:
                    for s in range(width):
                        d = waveform[s, c] - centers[clus, s, c]
                        sum += d*d
                else:
                    sum +=rms_waveform_channel[c]
            waveform_distance[clus] = sum
        
        return waveform_distance
except ImportError:
    HAVE_NUMBA = False








class PeelerEngineTesting(PeelerEngineClassic):

    #~ def estimate_jitter(self, left_ind, cluster_idx):
        #~ return 0
   
    def accept_tempate(self, left_ind, cluster_idx, jitter):
        
        #~ print('jitter', jitter)
        if np.abs(jitter) > (self.maximum_jitter_shift - 0.5):
            
            #~ print('too big jitter', jitter, np.abs(jitter))
            return False
        
        mask = self.sparse_mask[cluster_idx, :]
        
        shift = -int(np.round(jitter))
        jitter = jitter + shift
        left_ind = left_ind + shift
        if left_ind<0:
            return False
        new_left, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        pred_wf = pred_wf[:, :][:, mask]
        

        
        full_wf0 = self.catalogue['centers0'][cluster_idx,: , :][:, mask]
        #~ full_wf1 = self.catalogue['centers1'][cluster_idx,: , :][:, mask]
        #~ full_wf2 = self.catalogue['centers2'][cluster_idx,: , :][:, mask]
        #~ pred_wf = (full_wf0+jitter*full_wf1+jitter**2/2*full_wf2)
        #~ new_left = left_ind


        #debug
        #~ waveform_no_shift = self.fifo_residuals[left_ind - shift:left_ind- shift +self.peak_width,:]
        #~ full_wf_no_shift = waveform_no_shift[:, :][:, mask]
        

        # waveform L2 on mask
        waveform = self.fifo_residuals[new_left:new_left+self.peak_width,:]
        full_wf = waveform[:, :][:, mask]
        wf_nrj = np.sum(full_wf**2, axis=0)

        #~ thresh_ratio = 0.6
        thresh_ratio = 0.7
        #~ thresh_ratio = 0.8
        
        # criteria per channel
        #~ residual_nrj = np.sum((full_wf-pred_wf)**2, axis=0)
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        #~ weight = self.weight_per_template[label]
        #~ crietria_weighted = (wf_nrj>residual_nrj).astype('float') * weight
        #~ accept_template = np.sum(crietria_weighted) >= 0.9 * np.sum(weight)
        
        weigth = pred_wf ** 2 # TODO precompute this !!!!!!
        residual = (full_wf-pred_wf)
        s = np.sum((full_wf**2>=residual**2).astype(float) * weigth)
        #~ s = np.sum((pred_wf**2*weigth)>(residual*weigth))
        accept_template = s >np.sum(weigth) * thresh_ratio
        #~ print(s, np.sum(weigth) , np.sum(weigth)  * thresh_ratio)
        
        
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
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(nrows=3, sharex=True)
            axs[0].plot(full_wf.T.flatten(), color='b')
            #~ axs[0].plot(full_wf_no_shift.T.flatten(), color='c')
            
            
            if accept_template:
                axs[0].plot(pred_wf.T.flatten(), color='g')
            else:
                axs[0].plot(pred_wf.T.flatten(), color='r')
            
            #~ axs[0].plot(full_wf0.T.flatten(), color='k')
            
            
            
            axs[0].plot((full_wf-pred_wf).T.flatten(), color='m', lw=1)
            
            axs[1].plot((full_wf**2).T.flatten(), color='b')
            axs[1].plot((residual**2).T.flatten(), color='m')
            
            criterium = (full_wf**2>residual**2).astype(float) * weigth
            axs[2].plot(criterium.T.flatten(), color='k')
            
            plt.show()
            
        
        #~ #ENDDEBUG
        
        
        return accept_template

