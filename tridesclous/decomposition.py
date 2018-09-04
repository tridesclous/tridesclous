import numpy as np

import sklearn
import sklearn.decomposition

import sklearn.cluster
import sklearn.manifold

from . import tools


def project_waveforms(waveforms, method='pca', selection=None,  catalogueconstructor=None, **params):
    """
    
    
    """
    if selection is None:
        waveforms2 = waveforms
    else:
        waveforms2 = waveforms[selection]
    
    if waveforms2.shape[0] == 0:
        return None, None, None
    
    
    if method=='global_pca':
        projector = GlobalPCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='peak_max':
        projector = PeakMaxOnChannel(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='pca_by_channel':
        projector = PcaByChannel(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='neighborhood_pca':
        projector = NeighborhoodPca(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    #~ elif method=='peakmax_and_pca':
        #~ projector = PeakMax_and_PCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    #~ elif method=='tsne':
        #~ projector = TSNE(waveforms2, catalogueconstructor=catalogueconstructor, **params)    
    #~ elif method=='pca_by_channel_then_tsne':
        #~ projector = PcaByChannelThenTsne(waveforms2, catalogueconstructor=catalogueconstructor, **params)    
    else:
        Raise(NotImplementedError)
    
    features = projector.transform(waveforms2)
    channel_to_features = projector.channel_to_features
    return features, channel_to_features, projector


class GlobalPCA:
    def __init__(self, waveforms, catalogueconstructor=None, n_components=5, **params):
        cc = catalogueconstructor
        
        self.n_components = n_components
        self.waveforms = waveforms
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components, **params)
        self.pca.fit(flatten_waveforms)
        
        
        #In GlobalPCA all feature represent all channels
        self.channel_to_features = np.ones((cc.nb_channel, self.n_components), dtype='bool')


    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        return self.pca.transform(flatten_waveforms)

class PeakMaxOnChannel:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        cc = catalogueconstructor
        
        self.waveforms = waveforms
        self.ind_peak = -catalogueconstructor.info['waveform_extractor_params']['n_left']
        #~ print('PeakMaxOnChannel self.ind_peak', self.ind_peak)
        
        
        #In full PeakMaxOnChannel one feature is one channel
        self.channel_to_features = np.eye(cc.nb_channel, dtype='bool')
        
    def transform(self, waveforms):
        #~ print('ici', waveforms.shape, self.ind_peak)
        features = waveforms[:, self.ind_peak, : ].copy()
        return features


class PcaByChannel:
    def __init__(self, waveforms, catalogueconstructor=None, n_components_by_channel=3, **params):
        cc = catalogueconstructor
        
        self.waveforms = waveforms
        self.n_components_by_channel = n_components_by_channel
        self.pcas = []
        for c in range(cc.nb_channel):
            #~ print('c', c)
            pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
            pca.fit(waveforms[:,:,c])
            self.pcas.append(pca)

        #In full PcaByChannel n_components_by_channel feature correspond to one channel
        self.channel_to_features = np.zeros((cc.nb_channel, cc.nb_channel*n_components_by_channel), dtype='bool')
        for c in range(cc.nb_channel):
            self.channel_to_features[c, c*n_components_by_channel:(c+1)*n_components_by_channel] = True

    
    def transform(self, waveforms):
        n = self.n_components_by_channel
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        for c, pca in enumerate(self.pcas):
            all[:, c*n:(c+1)*n] = pca.transform(waveforms[:, :, c])
        return all
    


class NeighborhoodPca:
    def __init__(self, waveforms, catalogueconstructor=None, n_components_by_neighborhood=6, radius_um=300., **params):
        
        cc = catalogueconstructor
        
        self.n_components_by_neighborhood = n_components_by_neighborhood
        self.neighborhood = tools.get_neighborhood(cc.geometry, radius_um)
        
        self.pcas = []
        for c in range(cc.nb_channel):
            #~ print('c', c)
            neighbors = self.neighborhood[c, :]
            pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_neighborhood, **params)
            wfs = waveforms[:,:,neighbors]
            wfs = wfs.reshape(wfs.shape[0], -1)
            pca.fit(wfs)
            self.pcas.append(pca)

        #In full NeighborhoodPca n_components_by_neighborhood feature correspond to one channel
        self.channel_to_features = np.zeros((cc.nb_channel, cc.nb_channel*n_components_by_neighborhood), dtype='bool')
        for c in range(cc.nb_channel):
            self.channel_to_features[c, c*n_components_by_neighborhood:(c+1)*n_components_by_neighborhood] = True

    def transform(self, waveforms):
        n = self.n_components_by_neighborhood
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        for c, pca in enumerate(self.pcas):
            neighbors = self.neighborhood[c, :]
            wfs = waveforms[:,:,neighbors]
            wfs = wfs.reshape(wfs.shape[0], -1)
            all[:, c*n:(c+1)*n] = pca.transform(wfs)
        return all


#~ class PeakMax_and_PCA:
    #~ def __init__(self, waveforms, catalogueconstructor=None, **params):
        #~ self.waveforms = waveforms
        #~ self.ind_peak = -catalogueconstructor.info['waveform_extractor_params']['n_left']+1

        #~ self.pca =  sklearn.decomposition.IncrementalPCA(**params)
        #~ peaks_val = waveforms[:, self.ind_peak, : ].copy()
        #~ self.pca.fit(peaks_val)
        
        
    #~ def transform(self, waveforms):
        #~ peaks_val = waveforms[:, self.ind_peak, : ].copy()
        #~ features = self.pca.transform(peaks_val)
        
        #~ return features
    


#~ class TSNE:
    #~ def __init__(self, waveforms, catalogueconstructor=None, **params):
        #~ self.waveforms = waveforms
        #~ flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        #~ self.tsne = sklearn.manifold.TSNE(**params)
    
    #~ def transform(self, waveforms):
        #~ flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        #~ return self.tsne.fit_transform(flatten_waveforms)

    
#~ class PcaByChannelThenTsne:
    #~ def __init__(self, waveforms, catalogueconstructor=None, n_components_by_channel=3, **params):
        #~ self.waveforms = waveforms
        #~ self.n_components_by_channel = n_components_by_channel
        #~ self.pcas = []
        #~ for c in range(self.waveforms.shape[2]):
            #~ pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
            #~ pca.fit(waveforms[:,:,c])
            #~ self.pcas.append(pca)
        #~ self.tsne = sklearn.manifold.TSNE()
    
    #~ def transform(self, waveforms):
        #~ n = self.n_components_by_channel
        #~ all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        #~ for c, pca in enumerate(self.pcas):
            #~ all[:, c*n:(c+1)*n] = pca.transform(waveforms[:, :, c])
        #~ return self.tsne.fit_transform(all)



    