import numpy as np

import sklearn
import sklearn.decomposition

import sklearn.cluster


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
    elif method=='spatial_sliding_pca':
        projector = SpatialSlindingPca(waveforms2, catalogueconstructor=catalogueconstructor, **params)
        
    
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
    


class SpatialSlindingPca:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        
        cc = catalogueconstructor
        
        channel_group = cc.dataio.channel_groups[cc.chan_grp]
        #~ geometry = channel_group['geometry']
        geometry = [ channel_group['geometry'][c] for c in channel_group['channels'] ]
        geometry = np.array(geometry)
        #~ print(channel_group)
        #~ print(geometry)
        
        
        n = cc.nb_channel//8
        print('n', n)
        km = sklearn.cluster.KMeans(n_clusters=n)
        self.geo_labels = km.fit_predict(geometry)
        
        #~ print(labels)
        
        #DEBUG spatial
        #~ import matplotlib.pyplot as plt
        #~ import seaborn as sns
        #~ colors = sns.color_palette('husl', n)
        #~ fig, ax = plt.subplots()
        #~ for l in np.unique(labels):
            #~ g = geometry[l==labels]
            #~ ax.plot(g[:, 0], g[:,1], ls='None', color=colors[l], marker='o')
        #~ plt.show()
        print(waveforms.shape)
        #~ exit()
        
        self.pcas = {}
        
        for l in np.unique(self.geo_labels):
            
            wf = waveforms[:, :, l==self.geo_labels]
            flatten_wf = wf.reshape(wf.shape[0], -1)
            pca =  sklearn.decomposition.IncrementalPCA(n_components=3)
            pca.fit(flatten_wf)
            self.pcas[l] = pca


    def transform(self, waveforms):
        all_comp = []
        for l in np.unique(self.geo_labels):
            wf = waveforms[:, :, l==self.geo_labels]
            flatten_wf = wf.reshape(wf.shape[0], -1)
            all_comp.append(self.pcas[l].transform(flatten_wf))
        all_comp = np.concatenate(all_comp, axis=1)
        print(waveforms.shape)
        print(all_comp.shape)
        return all_comp
