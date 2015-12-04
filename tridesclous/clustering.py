import numpy as np
import pandas as pd
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
        self.labels = find_clusters(self.features, n_clusters, method='kmeans', **kargs)
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

