import numpy as np

import sklearn
import sklearn.cluster
import sklearn.mixture




def find_clusters(features, method='kmeans', n_clusters=1, **kargs):
    if method == 'kmeans':
        km = sklearn.cluster.KMeans(n_clusters=n_clusters,**kargs)
        labels = km.fit_predict(features)
    elif method == 'gmm':
        gmm = sklearn.mixture.GMM(n_components=n_clusters,**kargs)
        labels =gmm.fit_predict(features)
    else:
        raise(ValueError, 'find_clusters method unlnown')
    
    return labels

