import tridesclous as tdc
import pyqtgraph as pg

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture

#~ p = '../example/'
#~ dirname =  p + 'tridesclous_locust'
#~ dirname = p +'tridesclous_olfactory_bulb'
#~ dirname = p +'tridesclous_purkinje'

#~ dirname = '/media/samuel/SamCNRS/DataSpikeSorting/pierre/GT 252/tridesclous_GT256_20160426'
dirname = '/home/samuel/Documents/projet/DataSpikeSorting/david robbe/test_2017_03_24_14_35/tdc_test_2017_03_24_14_35/'


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
    n_left = cc.info['params_waveformextractor']['n_left']
    n_right = cc.info['params_waveformextractor']['n_right']
    width  = n_right - n_left
    
    
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
    
    import scipy.signal
    import diptest
    
    labels = cc.all_peaks['label'][cc.some_peaks_index]
    #~ keep = labels>=0
    #~ labels = labels[keep]
    #~ features = cc.some_features[keep]
    waveforms = cc.some_waveforms

    n_left = cc.info['params_waveformextractor']['n_left']
    n_right = cc.info['params_waveformextractor']['n_right']
    width = n_right - n_left


    
    cluster_labels = np.zeros(waveforms.shape[0], dtype='int64')
    
    
    bins=np.arange(-30,0, 0.1)
    
    def dirty_cut(x, bins):
        labels = np.zeros(x.size, dtype='int64')
        count, bins = np.histogram(x, bins=bins)
        #~ kernel = scipy.signal.get_window(10
        
        #~ kernel = scipy.signal.gaussian(51, 10)
        kernel = scipy.signal.gaussian(51, 5)
        #~ kernel = scipy.signal.gaussian(31, 10)
        kernel/= np.sum(kernel)
        
        #~ fig, ax = plt.subplots()
        #~ ax.plot(kernel)
        #~ plt.show()
        
        
        #~ count[count==1]=0
        count_smooth = np.convolve(count, kernel, mode='same')
        
        
        local_min_indexes, = np.nonzero((count_smooth[1:-1]<count_smooth[:-2])& (count_smooth[1:-1]<=count_smooth[2:]))
        
        if local_min_indexes.size==0:
            lim = 0
        else:
            
            n_on_left = []
            for ind in local_min_indexes:
                lim = bins[ind]
                n = np.sum(count[bins[:-1]<=lim])
                n_on_left.append(n)
                #~ print('lim', lim, n)
                
                #~ if n>30:
                    #~ break
                #~ else:
                    #~ lim = None
            n_on_left = np.array(n_on_left)
            print('n_on_left', n_on_left, 'local_min_indexes', local_min_indexes,  x.size)
            p = np.argmin(np.abs(n_on_left-x.size//2))
            print('p', p)
            lim = bins[local_min_indexes[p]]
            
            
        
        #~ lim = bins[local_min[0]]
        #~ print(local_min, min(x), lim)
        
        #~ if x.size==3296:
        #~ fig, ax = plt.subplots()
        #~ ax.plot(bins[:-1], count, color='b')
        #~ ax.plot(bins[:-1], count_smooth, color='g')
        #~ if lim is not None:
            #~ ax.axvline(lim)
        #~ plt.show()
        

        if lim is None:
            return None, None
        
        labels[x>lim] = 1
        
        return labels, lim
        
    
    wf = waveforms.swapaxes(1,2).reshape(waveforms.shape[0], -1)
    
    k = 0
    dim_visited = []
    for i in range(1000):
                
        left_over = np.sum(cluster_labels>=k)
        print()
        print('i', i, 'k', k, 'left_over', left_over)
        
        sel = cluster_labels == k
        
        if i!=0 and left_over<30:# or k==40:
            cluster_labels[sel] = -k
            print('BREAK left_over<30')
            break
        
        wf_sel = wf[sel]
        n_with_label = wf_sel .shape[0]
        print('n_with_label', n_with_label)
        
        if wf_sel .shape[0]<30:
            print('too few')
            cluster_labels[sel] = -k
            k+=1
            dim_visited = []
            continue
        
        med, mad = compute_median_mad(wf_sel)
        
        if np.all(mad<1.6):
            print('mad<1.6')
            k+=1
            dim_visited = []
            continue

        if np.all(wf_sel.shape[0]<100):
            print('Too small cluster')
            k+=1
            dim_visited = []
            continue
        
        
        #~ weight = mad-1
        #~ feat = wf_sel * weight
        #~ feat = wf_sel[:, np.argmax(mad), None]
        #~ print(feat.shape)
        #~ print(feat)
        
        
        #~ while True:
        
        possible_dim, = np.nonzero(med<0)
        possible_dim = possible_dim[~np.in1d(possible_dim, dim_visited)]
        if len(possible_dim)==0:
            print('BREAK len(possible_dim)==0')
            #~ dim = None
            break
        dim = possible_dim[np.argmax(mad[possible_dim])]
        print('dim', dim)
        #~ dim_visited.append(dim)
        #~ print('dim', dim, 'dim_visited',dim_visited)
        #~ feat = wf_sel[:, dim]
        
        #~ dip_values = np.zeros(possible_dim.size)
        #~ for j, dim in enumerate(possible_dim):
            #~ print('j', j)
            #~ dip, f = diptest.dip(wf_sel[:, dim], full_output=True, x_is_sorted=False)
            #~ dip_values[j] = dip
        
        #~ dip_values[j] = dip

        #~ dim = possible_dim[np.argmin(dip_values)]
        #~ dip = dip_values[np.argmin(dip_values)]
        
        
        #~ dip, f = diptest.dip(wf_sel[:, dim], full_output=True, x_is_sorted=False)
        #~ dip, pval = diptest.diptest(wf_sel[:, dim])
        
        #~ print('dim', dim, 'dip', dip, 'pval', pval)

        #~ fig, axs = plt.subplots(nrows=2, sharex=True)
        #~ axs[0].fill_between(np.arange(med.size), med-mad, med+mad, alpha=.5)
        #~ axs[0].plot(np.arange(med.size), med)
        #~ axs[0].set_title(str(wf_sel.shape[0]))
        #~ axs[1].plot(mad)
        #~ axs[1].axvline(dim, color='r')
        #~ plt.show()

            
            #~ if dip<0.01:
                #~ break
        
        
        
        feat = wf_sel[:, dim]
        
        labels, lim = dirty_cut(feat, bins)
        if labels is None:
            channel_index = dim // width
            #~ dim_visited.append(dim)
            dim_visited.extend(range(channel_index*width, (channel_index+1)*width))
            print('loop', dim_visited)
            continue
        
        
            
        
        print(feat[labels==0].size, feat[labels==1].size)
        
        #~ fig, ax = plt.subplots()
        #~ count, bins = np.histogram(feat, bins=bins)
        #~ count0, bins = np.histogram(feat[labels==0], bins=bins)
        #~ count1, bins = np.histogram(feat[labels==1], bins=bins)
        #~ ax.axvline(lim)
        #~ ax.plot(bins[:-1], count, color='b')
        #~ ax.plot(bins[:-1], count0,  color='r')
        #~ ax.plot(bins[:-1], count1, color='g')
        #~ ax.set_title('dim {} dip {:.4f} pval {:.4f}'.format(dim, dip, pval))
        #~ plt.show()
        

        #~ if pval>0.1:
            #~ print('BREAK pval>0.05')
            #~ break
        

        ind, = np.nonzero(sel)
        
        #~ med0, mad0 = compute_median_mad(feat[labels==0])
        #~ med1, mad1 = compute_median_mad(feat[labels==1])
        #~ if np.abs(med0)>np.abs(med1):
        #~ if mad0<mad1:
            #~ cluster_labels[ind[labels==1]] += 1
        #~ else:
            #~ cluster_labels[ind[labels==0]] += 1
        
        #~ if dip>0.05 and pval<0.01:
        
        
        
        print('nb1', np.sum(labels==1), 'nb0', np.sum(labels==0))
        
        if np.sum(labels==0)==0:
            channel_index = dim // width
            #~ dim_visited.append(dim)
            dim_visited.extend(range(channel_index*width, (channel_index+1)*width))
            print('nb0==0', dim_visited)
            
            continue
        
        #~ cluster_labels[cluster_labels>k] += 1#TODO reflechir la dessus!!!
        cluster_labels[ind[labels==1]] += 1
        
        if np.sum(labels==1)==0:
            k+=1
            dim_visited = []
        

        #~ med0, mad0 = compute_median_mad(feat[labels==0])
        

        
    


    
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
    
 
 