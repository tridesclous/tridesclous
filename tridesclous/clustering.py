import numpy as np
import pandas as pd
import scipy.signal
import sklearn
import sklearn.decomposition
import sklearn.cluster
import sklearn.mixture

def find_clusters(features, n_clusters,  method='kmeans', **kargs):
    if method == 'kmeans':
        km = sklearn.cluster.KMeans(n_clusters=n_clusters,**kargs)
        labels_ = km.fit_predict(features.values)
    elif method == 'gmm':
        gmm = self._cluster_instance = sklearn.mixture.GMM(n_clusters=n_clusters,**kargs)
        labels_ =gmm.fit_predict(features.values)
    
    labels = pd.Series(labels_, index = features.index, name = 'label')
    return labels

#~ def order_clusters(features, labels):
    
    
    

class Clustering_:
    """
    Clustering class :
        * project waveform with PCA
        * do clustering (kmean or gmm)
        * propose method for merge and split cluster.
    """
    def __init__(self, waveforms):
        self.waveforms = waveforms
        
    
    def project(self, method = 'pca', n_components = 5):
        #TODO remove peak than are out to avoid PCA polution.
        
        if method=='pca':
            self._pca = sklearn.decomposition.PCA(n_components = n_components)
            self.features = pd.DataFrame(self._pca.fit_transform(self.waveforms.values), index = self.waveforms.index,
                        columns = ['pca{}'.format(i) for i in range(n_components)])
        
        return self.features
    
    def find_clusters(self, n_clusters,method='kmeans', **kargs):
        self.labels = find_clusters(self.features, n_clusters, method='kmeans', **kargs)
        self.cluster_labels = np.unique(self.labels)
        return self.labels
    
    def merge_cluster(self, label1, label2):
        self.labels[self.labels==label2] = label1
        return self.labels
    
    def split_cluster(self, label, n, method='kmeans', **kargs):
        mask = self.labels==label
        new_label = find_clusters(self.features, n, method='kmeans', **kargs)
        new_label += max(self.labels)+1
        self.labels[mask] = new_label
        return self.labels
    

    def construct_catalogue(self):
        """
        
        """
        
        
        self.catalogue = {}
        nb_channel = self.waveforms.columns.levels[0].size
        for k in self.cluster_labels:
            # take peak of this cluster
            # and reshaape (nb_peak, nb_channel, nb_csample)
            wf = self.waveforms[self.labels==k].values
            wf = wf.reshape(wf.shape[0], nb_channel, -1)
            
            #compute first and second derivative on dim=2
            kernel = np.array([1,0,-1])/2.
            kernel = kernel[None, None, :]
            wfD =  scipy.signal.fftconvolve(wf,kernel,'same') # first derivative
            wfDD =  scipy.signal.fftconvolve(wfD,kernel,'same') # second derivative
            
            # medians
            center = np.median(wf, axis=0)
            centerD = np.median(wfD, axis=0)
            centerDD = np.median(wfDD, axis=0)
            mad = np.median(np.abs(wf-center),axis=0)*1.4826
            
            #eliminate margin because of border effect of derivative and reshape
            center = center[:, 2:-2].reshape(-1)
            centerD = centerD[:, 2:-2].reshape(-1)
            centerDD = centerDD[:, 2:-2].reshape(-1)
            mad = mad[:, 2:-2].reshape(-1)
            
            self.catalogue[k] = {'center' : center, 'centerD' : centerD, 'centerDD': centerDD,
                                            'mad': mad}
        
        return self.catalogue



from .mpl_plot import ClusteringPlot
class Clustering(Clustering_, ClusteringPlot):
    pass




