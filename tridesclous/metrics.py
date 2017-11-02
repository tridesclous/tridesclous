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


def inverse_weihgted_distance(x):
    #~ func = lambda u, v: np.sqrt(np.mean((u-v)**2))
    func = lambda u, v: np.mean(np.abs(u-v)/np.maximum(np.abs(u),np.abs(v)))
    d = scipy.spatial.distance.pdist(x, metric=func)
    d = scipy.spatial.distance.squareform(d)
    print(d)
    
    return d
    
    


def compute_silhouette(data, labels, metric='euclidean'):

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
    
