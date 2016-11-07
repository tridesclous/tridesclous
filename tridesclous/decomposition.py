import numpy as np

import sklearn
import sklearn.decomposition



def project_waveforms(flatten_waveforms, method='IncrementalPCA', selection=None, **params):
    """
    
    
    """
    if method=='pca':
       method='IncrementalPCA' 
    
    if method=='IncrementalPCA':
        pca =  sklearn.decomposition.IncrementalPCA(**params)
    else:
        Raise(NotImplementedError)
    
    if selection is None:
        pca.fit(flatten_waveforms)
    else:
        pca.fit(flatten_waveforms[selection])
    
    features = pca.transform(flatten_waveforms)
    
    return features