import tridesclous as tdc
import pyqtgraph as pg

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

p = '../example/'
#~ dirname =  p + 'tridesclous_locust'
#~ dirname = p +'tridesclous_olfactory_bulb'
dirname = p +'tridesclous_purkinje'
dataio = tdc.DataIO(dirname=dirname)


cc = catalogueconstructor = tdc.CatalogueConstructor(dataio=dataio)
print(cc)

#~ wf = cc.some_waveforms
#~ data = wf.reshape(wf.shape[0], -1)



def test_silouhette_or_bic():

    data = cc.some_features

    labels = cc.all_peaks['label'][cc.some_peaks_index]
    keep = labels>=0
    labels = labels[keep]
    data = data[keep]        





    range_n_clusters = list(range(2,20))

    kmeans_silhouette_scores = []
    guassianmixture_bic_scores= []

    for n_clusters in range_n_clusters:
        #kmeans
        cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        kmeans_silhouette_scores.append(silhouette_avg)
        
        #GMM
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        cluster_labels = gmm.fit(data)
        guassianmixture_bic_scores.append(gmm.bic(data))

        
        
    fig, ax = plt.subplots()
    ax.plot(range_n_clusters, kmeans_silhouette_scores)

    fig, ax = plt.subplots()
    ax.plot(range_n_clusters, guassianmixture_bic_scores)

    plt.show()
    

def compute_mad(v):
    med = np.median(v)
    return np.median(np.abs(v-med)) * 1.4826
    
def test_range_to_find_residual_minimize():

    labels = cc.all_peaks['label'][cc.some_peaks_index]
    keep = labels>=0
    labels = labels[keep]
    features = cc.some_features[keep]
    waveforms = cc.some_waveforms[keep]
    
    
    range_n_clusters = list(range(1, 25))
    
    all_rms_residuals =[]
    fig, ax = plt.subplots()
    for n_clusters in range_n_clusters:
        #~ cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(features)

        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        cluster_labels = gmm.fit(features).predict(features)
        print(cluster_labels)
        
        
        
        rms_residuals = []
        for k in np.unique(cluster_labels):
            sel = cluster_labels==k
            wfs = waveforms[sel]
            med = np.median(wfs, axis=0)
            residuals = wfs-med
            
            rms_residuals.append(np.mean(residuals**2))
           
        
        print('n_clusters', n_clusters, rms_residuals)
        
        all_rms_residuals.append(rms_residuals)
        
        
        ax.plot(np.ones(len(rms_residuals))*n_clusters, rms_residuals, ls='None', marker='o', color='b')
        
        
    
    
    means = [ np.mean(r) for r in all_rms_residuals]
    stds = [ np.std(r) for r in all_rms_residuals]
    ax.errorbar(range_n_clusters, means, stds, color='r')
    
    meds = [ np.median(r) for r in all_rms_residuals]
    mads = [ compute_mad(r) for r in all_rms_residuals]
    ax.errorbar(range_n_clusters, meds, mads, color='g')
    
    plt.show()
    
    
    
    
            
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(residuals.swapaxes(1,2).reshape(residuals.shape[0], -1).T)
            #~ print(residuals)
            
            #~ plt.show()
            
            
        
        
        
    
    



#~ catalogueconstructor.project(method='pca', n_components=15)

def find_clusters_and_show():

    #~ catalogueconstructor.find_clusters(method='kmeans', n_clusters=12)
    catalogueconstructor.find_clusters(method='gmm', n_clusters=15)

    app = pg.mkQApp()
    win = tdc.CatalogueWindow(catalogueconstructor)
    win.show()

    app.exec_()    



if __name__ =='__main__':
    #~ test_silouhette_or_bic()
    #~ test_range_to_find_residual_minimize()
    find_clusters_and_show()
    
    
    