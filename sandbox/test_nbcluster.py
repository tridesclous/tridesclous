import tridesclous as tdc
import pyqtgraph as pg

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, MeanShift
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
    

def compute_mad(v, axis=0):
    med = np.median(v, axis=axis)
    mad = np.median(np.abs(v-med), axis=axis) * 1.4826
    return mad

def compute_median_mad(v, axis=0):
    med = np.median(v, axis=axis)
    mad = np.median(np.abs(v-med), axis=axis) * 1.4826
    return med, mad



def test_range_to_find_residual_minimize():

    labels = cc.all_peaks['label'][cc.some_peaks_index]
    keep = labels>=0
    labels = labels[keep]
    features = cc.some_features[keep]
    waveforms = cc.some_waveforms[keep]
    
    
    range_n_clusters = list(range(9, 11))
    
    all_rms_residuals =[]
    fig, ax = plt.subplots()
    for n_clusters in range_n_clusters:
        #~ cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(features)

        #~ gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        #~ cluster_labels = gmm.fit(features).predict(features)
        #~ print(cluster_labels)
        
        #~ cluster_labels = MeanShift(cluster_all=False).fit_predict(features)
        
        
        
        rms_residuals = []
        for k in np.unique(cluster_labels):
            if k<0:
                continue
            sel = cluster_labels==k
            wfs = waveforms[sel]
            med = np.median(wfs, axis=0)
            residuals = wfs-med
            
            rms_residuals.append(np.mean(residuals**2))
            
            #~ fig, ax1 = plt.subplots()
            #~ r = residuals.swapaxes(1,2).reshape(residuals.shape[0], -1)
            #~ print(r.shape)
            #~ med = np.median(r, axis=0)
            #~ mad = compute_mad(r,  axis=0)
            #~ print(med.shape)
            #~ ax1.fill_between(np.arange(med.size), med-mad, med+mad)
            #~ print(residuals)
            
            #~ plt.show()
            
           
        
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
    
    
    
def test_split_to_find_residual_minimize():
    
    labels = cc.all_peaks['label'][cc.some_peaks_index]
    #~ keep = labels>=0
    #~ labels = labels[keep]
    #~ features = cc.some_features[keep]
    waveforms = cc.some_waveforms
    
    cluster_labels = np.zeros(waveforms.shape[0], dtype='int64')
    
    
    wf = waveforms.swapaxes(1,2).reshape(waveforms.shape[0], -1)
    
    k = 0
    
    for i in range(1000):
        
        
        left_over = np.sum(cluster_labels>=k)
        
        print('i', i, 'k', k, 'left_over', left_over)
        
        sel = cluster_labels == k
        
        if i!=0 and left_over<30:# or k==40:
            cluster_labels[sel] = -k
            break
        
        wf_sel = wf[sel]
        #~ print(wf_sel .shape[0])
        
        if wf_sel .shape[0]<30:
            cluster_labels[sel] = -k
            k+=1
            continue
        
        med, mad = compute_median_mad(wf_sel)
        
        if np.all(mad<1.6):
            k+=1
            continue
        
        
        #~ fig, axs = plt.subplots(nrows=2)
        #~ axs[0].fill_between(np.arange(med.size), med-mad, med+mad, alpha=.5)
        #~ axs[0].plot(np.arange(med.size), med)
        #~ axs[0].set_title(str(wf_sel.shape[0]))
        #~ axs[1].plot(mad)
        #~ axs[1].axvline(np.argmax(mad), color='r')
        #~ plt.show()
        
        #~ weight = mad-1
        #~ feat = wf_sel * weight
        #~ feat = wf_sel[:, np.argmax(mad), None]
        #~ print(feat.shape)
        #~ print(feat)
        
        ind, = np.nonzero(med<0)
        i = ind[np.argmax(mad[ind])]
        feat = wf_sel[:, i, None]
        
        labels = KMeans(n_clusters=2).fit_predict(feat)
        #~ gmm = GaussianMixture(n_components=2, covariance_type='full')
        #~ labels = gmm.fit(feat).predict(feat)
        
        #~ fig, ax = plt.subplots()
        #~ count, bins = np.histogram(feat, bins=50)
        #~ count0, bins = np.histogram(feat[labels==0], bins=bins)
        #~ count1, bins = np.histogram(feat[labels==1], bins=bins)
        #~ ax.plot(bins[:-1], count, color='b')
        #~ ax.plot(bins[:-1], count0,  color='r')
        #~ ax.plot(bins[:-1], count1, color='g')
        #~ plt.show()
        
        #~ cluster_labels += 1
        ind, = np.nonzero(sel)
        
        med0, mad0 = compute_median_mad(feat[labels==0])
        med1, mad1 = compute_median_mad(feat[labels==1])
        if np.abs(med0)>np.abs(med1):
        #~ if mad0<mad1:
            cluster_labels[ind[labels==1]] += 1
        else:
            cluster_labels[ind[labels==0]] += 1
    
    
    fig, axs = plt.subplots(nrows=2)
    for k in np.unique(cluster_labels):
        sel = cluster_labels == k
        wf_sel = wf[sel]
        
        med, mad = compute_median_mad(wf_sel)
        axs[0].plot(med, label=str(k))
        axs[1].plot(mad)
        
    plt.show()
    
    cc.all_peaks['label'][cc.some_peaks_index] = cluster_labels
    

    app = pg.mkQApp()
    win = tdc.CatalogueWindow(catalogueconstructor)
    win.show()

    app.exec_()    
    



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
    
    test_split_to_find_residual_minimize()
    
    #~ find_clusters_and_show()
    
    
    