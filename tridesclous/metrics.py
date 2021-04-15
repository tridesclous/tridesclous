import numpy as np
import sklearn.metrics.pairwise
import scipy.spatial


def compute_similarity(data, method):
    if method in ('cosine_similarity',  'linear_kernel', 'polynomial_kernel',
                    'sigmoid_kernel', 'rbf_kernel', 'laplacian_kernel'):
        func = getattr(sklearn.metrics.pairwise, method)
        return func(data)
    else:
        raise(NotImplementedError)


def cosine_similarity_with_max(x):
    """
    Similar to cosine_similarity but normed by the max(abs) on each dim.
    
    m = np.maximum(np.abs(u), np.abs(v))
    similarity = np.dot(u, v.T)/np.dot(m, m.T)
    
    """
    def func(u, v):
        #cosine affinity
        #~ sim_final = np.dot(u, v.T)/(np.sqrt(np.dot(u, u.T))*np.sqrt(np.dot(v, v.T)))
        
        #cosine affinity
        sim_final = np.dot(u, v.T)
        #~ sim_final /= (np.sqrt(np.dot(u, u.T))*np.sqrt(np.dot(v, v.T)))
        m = np.maximum(np.abs(u), np.abs(v))
        sim_final /= np.dot(m, m.T)
        return sim_final
    
    cluster_similarity = scipy.spatial.distance.pdist(x, metric=func)
    cluster_similarity = scipy.spatial.distance.squareform(cluster_similarity)
    cluster_similarity += np.eye(cluster_similarity.shape[0])
    return cluster_similarity
    
    


def compute_silhouette(data, labels, metric='euclidean'):
    
    if np.unique(labels).size<2:
        return
    #~ self.silhouette_avg = silhouette_score(data, labels)
    silhouette_values = sklearn.metrics.silhouette_samples(data, labels)
        
    #~ self.silhouette_by_labels = {}
    #~ labels_list = np.unique(labels)
    #~ for k in labels_list:
        #~ v = silhouette_values[k==labels]
        #~ v.sort()
        #~ self.silhouette_by_labels[k] = v
    #~ silhouette_avg = np.mean(silhouette_values)
    
    return silhouette_values#, silhouette_avg
    
