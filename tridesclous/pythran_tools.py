"""
This a pythran version when using sparse template
To use it you need to compile this code with
pythran pythran_tools.py -o pythran_tools.so -fopenmp

"""

import numpy as np

#pythran export pythran_loop_sparse_dist(float32[:,:], float32 [:,:,:], bool[:,:])
def pythran_loop_sparse_dist(waveform, centers,  mask):
    nb_clus = centers.shape[0]
    width = centers.shape[1]
    nb_chan = centers.shape[2]
    
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