import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.cluster
import sklearn.mixture


class Clustering:
    """
    
    
    
    """
    
    def __init__(self, waveforms):
        self.waveforms = waveforms
        
    
    def project(self, method = 'pca', n_components = 5):
        if method=='pca':
            self._pca = sklearn.decomposition.PCA(n_components = n_components)
            
            self.features = pd.DataFrame(self._pca.fit_transform(self.waveforms.values), index = self.waveforms.index,
                        columns = ['pca{}'.format(i) for i in range(n_components)])
            return self.features
    
    def find_clusters(self, n_clusters,method='kmeans', **kargs):
        if method == 'kmeans':
            km = self._cluster_instance = sklearn.cluster.KMeans(n_clusters=n_clusters,**kargs)
            labels = km.fit_predict(self.features.values)
            self.labels = pd.Series(labels, index = self.waveforms.index, name = 'label')
        elif method == 'gmm':
            gmm = self._cluster_instance = sklearn.mixture.GMM(n_clusters=n_clusters,**kargs)
            labels =gmm.fit_predict(self.features.values)
            self.labels = pd.Series(labels, index = self.waveforms.index, name = 'label')
        
        return self.labels
