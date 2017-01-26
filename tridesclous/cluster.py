import numpy as np

import sklearn
import sklearn.cluster
import sklearn.mixture




def find_clusters(features, method='kmeans', n_clusters=1, **kargs):
    if method == 'kmeans':
        km = sklearn.cluster.KMeans(n_clusters=n_clusters,**kargs)
        labels = km.fit_predict(features)
    elif method == 'gmm':
        gmm = sklearn.mixture.GaussianMixture(n_components=n_clusters,**kargs)
        #~ labels =gmm.fit_predict(features)
        gmm.fit(features)
        labels =gmm.predict(features)
    elif method == 'agglomerative':
        agg = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, **kargs)
        labels = agg.fit_predict(features)
    elif method == 'dbscan':
        dbscan = sklearn.cluster.DBSCAN(**kargs)
        labels = dbscan.fit_predict(features)
    else:
        raise(ValueError, 'find_clusters method unlnown')
    
    return labels

