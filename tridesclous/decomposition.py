import numpy as np

import sklearn
import sklearn.decomposition



def project_waveforms(peak_waveforms, method='pca', selection=None, catalogueconstructor=None, **params):
    """
    
    
    """
    if method=='pca':
       method='IncrementalPCA' 
    
    if method=='IncrementalPCA':
        flatten_waveforms = peak_waveforms.reshape(peak_waveforms.shape[0], -1)
        pca =  sklearn.decomposition.IncrementalPCA(**params)
        if selection is None:
            pca.fit(flatten_waveforms)
        else:
            pca.fit(flatten_waveforms[selection])
        features = pca.transform(flatten_waveforms)
        projector = pca
    elif method=='peak_max':
        ind_peak = -catalogueconstructor.info['params_waveformextractor']['n_left']+1
        features = peak_waveforms[:, ind_peak, : ].copy()
        projector = None
    else:
        Raise(NotImplementedError)
    
    
    return features, projector