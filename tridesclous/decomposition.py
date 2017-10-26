import numpy as np

import sklearn
import sklearn.decomposition

import sklearn.cluster
import sklearn.manifold


def project_waveforms(waveforms, method='pca', selection=None,  catalogueconstructor=None, **params):
    """
    
    
    """
    if selection is None:
        waveforms2 = waveforms
    else:
        waveforms2 = waveforms[selection]
    
    
    if method=='pca':
        projector = FullPCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='pca_by_channel':
        projector = PcaByChannel(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='peak_max':
        projector = PeakMaxOnChannel(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='peakmax_and_pca':
        projector = PeakMax_and_PCA(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='spatial_sliding_pca':
        projector = SpatialSlindingPca(waveforms2, catalogueconstructor=catalogueconstructor, **params)
    elif method=='tsne':
        projector = TSNE(waveforms2, catalogueconstructor=catalogueconstructor, **params)    
    elif method=='pca_by_channel_then_tsne':
        projector = PcaByChannelThenTsne(waveforms2, catalogueconstructor=catalogueconstructor, **params)    
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


class PcaByChannel:
    def __init__(self, waveforms, catalogueconstructor=None, n_components_by_channel=3, **params):
        self.waveforms = waveforms
        self.n_components_by_channel = n_components_by_channel
        self.pcas = []
        for c in range(self.waveforms.shape[2]):
            #~ print('c', c)
            pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
            pca.fit(waveforms[:,:,c])
            self.pcas.append(pca)
    
    def transform(self, waveforms):
        n = self.n_components_by_channel
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        for c, pca in enumerate(self.pcas):
            #~ print('c', c)
            all[:, c*n:(c+1)*n] = pca.transform(waveforms[:, :, c])
        return all
    

class PeakMaxOnChannel:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        self.waveforms = waveforms
        self.ind_peak = -catalogueconstructor.info['params_waveformextractor']['n_left']+1
        #~ print('PeakMaxOnChannel self.ind_peak', self.ind_peak)
        
    def transform(self, waveforms):
        #~ print('ici', waveforms.shape, self.ind_peak)
        features = waveforms[:, self.ind_peak, : ].copy()
        return features


class PeakMax_and_PCA:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        self.waveforms = waveforms
        self.ind_peak = -catalogueconstructor.info['params_waveformextractor']['n_left']+1
        #~ print('PeakMaxOnChannel self.ind_peak', self.ind_peak)

        self.pca =  sklearn.decomposition.IncrementalPCA(**params)
        peaks_val = waveforms[:, self.ind_peak, : ].copy()
        self.pca.fit(peaks_val)
        
        
    def transform(self, waveforms):
        #~ print('ici', waveforms.shape, self.ind_peak)
        peaks_val = waveforms[:, self.ind_peak, : ].copy()
        features = self.pca.transform(peaks_val)
        
        return features
    


class SpatialSlindingPca:
    def __init__(self, waveforms, catalogueconstructor=None, n_components_by_channel=3,**params):
        
        cc = catalogueconstructor
        
        channel_group = cc.dataio.channel_groups[cc.chan_grp]
        #~ geometry = channel_group['geometry']
        geometry = [ channel_group['geometry'][c] for c in channel_group['channels'] ]
        geometry = np.array(geometry)
        #~ print(channel_group)
        #~ print(geometry)
        
        
        n = cc.nb_channel//8
        #~ print('n', n)
        km = sklearn.cluster.KMeans(n_clusters=n)
        self.geo_labels = km.fit_predict(geometry)
        
        #~ print(labels)
        
        #DEBUG spatial
        import matplotlib.pyplot as plt
        import seaborn as sns
        colors = sns.color_palette('Set1', n_colors=n)
        fig, ax = plt.subplots()
        for l in np.unique(self.geo_labels):
            g = geometry[l==self.geo_labels]
            ax.plot(g[:, 0], g[:,1], ls='None', color=colors[l], marker='o')
        plt.show()
        #~ print(waveforms.shape)
        #~ exit()
        
        self.pcas = {}
        
        for l in np.unique(self.geo_labels):
            #~ print('l', l)
            wf = waveforms[:, :, l==self.geo_labels]
            flatten_wf = wf.reshape(wf.shape[0], -1)
            print(flatten_wf.shape)
            pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel)
            pca.fit(flatten_wf)
            self.pcas[l] = pca


    def transform(self, waveforms):
        all_comp = []
        for l in np.unique(self.geo_labels):
            wf = waveforms[:, :, l==self.geo_labels]
            flatten_wf = wf.reshape(wf.shape[0], -1)
            all_comp.append(self.pcas[l].transform(flatten_wf))
        all_comp = np.concatenate(all_comp, axis=1)
        #~ print(waveforms.shape)
        #~ print(all_comp.shape)
        return all_comp



class TSNE:
    def __init__(self, waveforms, catalogueconstructor=None, **params):
        self.waveforms = waveforms
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        self.tsne = sklearn.manifold.TSNE(**params)
        #~ self.tsne = sklearn.manifold.MDS(**params)
        #~ self.tsne = sklearn.decomposition.KernelPCA(**params)
        #~ self.tsne.fit(flatten_waveforms)


    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        return self.tsne.fit_transform(flatten_waveforms)

    
class PcaByChannelThenTsne:
    def __init__(self, waveforms, catalogueconstructor=None, n_components_by_channel=3, **params):
        self.waveforms = waveforms
        self.n_components_by_channel = n_components_by_channel
        self.pcas = []
        for c in range(self.waveforms.shape[2]):
            pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
            pca.fit(waveforms[:,:,c])
            self.pcas.append(pca)
        self.tsne = sklearn.manifold.TSNE()
    
    def transform(self, waveforms):
        n = self.n_components_by_channel
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        for c, pca in enumerate(self.pcas):
            all[:, c*n:(c+1)*n] = pca.transform(waveforms[:, :, c])
        return self.tsne.fit_transform(all)



    