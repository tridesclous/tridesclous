import numpy as np

import sklearn
import sklearn.decomposition



def project_waveforms(waveforms, method='pca', selection=None,  catalogueconstructor=None, **params):
    """
    
    
    """
    if selection is None:
        waveforms2 = waveforms
    else:
        waveforms2 = waveforms[selection]
    
    
    if method=='pca':
        projector = FullPCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='peak_max':
        projector = PeakMaxOnChannel(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='peakmax_and_pca':
        projector = PeakMax_and_PCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    else:
        Raise(NotImplementedError)
    
    features = projector.transform(waveforms2)
    return features, projector


class FullPCA:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        self.waveforms = waveforms
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.pca =  sklearn.decomposition.IncrementalPCA(**params)
        self.pca.fit(flatten_waveforms)


    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        return self.pca.transform(flatten_waveforms)


class PeakMaxOnChannel:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        self.waveforms = waveforms
        self.ind_peak = -catalogueconstructor.info['params_waveformextractor']['n_left']+1
        print('PeakMaxOnChannel self.ind_peak', self.ind_peak)
        
    def transform(self, waveforms):
        print('ici', waveforms.shape, self.ind_peak)
        features = waveforms[:, self.ind_peak, : ].copy()
        return features


class PeakMax_and_PCA:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        self.waveforms = waveforms
        self.ind_peak = -catalogueconstructor.info['params_waveformextractor']['n_left']+1
        print('PeakMaxOnChannel self.ind_peak', self.ind_peak)

        self.pca =  sklearn.decomposition.IncrementalPCA(**params)
        peaks_val = waveforms[:, self.ind_peak, : ].copy()
        self.pca.fit(peaks_val)
        
        
    def transform(self, waveforms):
        print('ici', waveforms.shape, self.ind_peak)
        peaks_val = waveforms[:, self.ind_peak, : ].copy()
        features = self.pca.transform(peaks_val)
        
        return features
