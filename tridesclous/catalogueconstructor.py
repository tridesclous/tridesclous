import numpy as np

import sklearn
import sklearn.decomposition
import sklearn.cluster
import sklearn.mixture

from . import signalpreprocessor
from . import  peakdetector
from . import waveformextractor

class CatalogueConstructor:
    """
    CatalogueConstructor scan a smal part of the dataset to construct the catalogue.
    Catalogue are the centroid of each cluster.
    
    
    
    There 4 steps:
      1. initialize
      2. loop over chunks:
        * detect peaks
        * extract waveforms
        
      3. finalize:
        * concatenate waveforms
        * find good waveforms limit
        * learn partial PCA
        * apply PCA
        * find clusters
        * put labels on peaks
      4. save
    
    When done you can opntionaly open the GUI CatalogueWindow for doing manual
    clustering.
    
    Usage:
        cc = CatalogueConstructor(dataio)
        cc.initialize()
        for chunk, pos in iter_on_preprocessed_signals:
            cc.process_one_chunk(pos, chunk)
        cc.finalize()
        cc.save()
    
    
    """
    def __init__(self, dataio):
        self.dataio = dataio
        
    
    def initialize(self, chunksize=1024,
            memory_mode='ram',
            
            #signal preprocessor
            signalpreprocessor_engine='signalpreprocessor_numpy',
            highpass_freq=300, backward_chunksize=1280,
            
            #peak detector
            peakdetector_engine='peakdetector_numpy',
            peak_sign='-', relative_threshold=5, peak_span=0.0005,
            
            #waveformextractor
            n_left=-20, n_right=30, 
            
            #features
            pca_batch_size=16384,
            ):
                
        self.chunksize = chunksize

        self.n_left = n_left
        self.n_right = n_right
        
        
        self.memory_mode = memory_mode
        
        if self.memory_mode=='ram':
            self.all_peak_pos = []
            self.all_peak_waveforms = []
        
        elif self.memory_mode=='memmap':
            raise(NotImplementedError)
        
        self.pca_batch_size = pca_batch_size
        
        self.peak_sign = peak_sign
        
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[signalpreprocessor_engine]
        self.signalpreprocessor = SignalPreprocessor_class(self.dataio.sample_rate, self.dataio.nb_channel, chunksize, self.dataio.dtype)
        self.signalpreprocessor.change_params(highpass_freq=highpass_freq, backward_chunksize=backward_chunksize)
        
        PeakDetector_class = peakdetector.peakdetector_engines[peakdetector_engine]
        self.peakdetector = PeakDetector_class(self.dataio.sample_rate, self.dataio.nb_channel,
                                                        self.chunksize, self.dataio.dtype)
        self.peakdetector.change_params(peak_sign=peak_sign,
                                    relative_threshold=relative_threshold, peak_span=peak_span)
        
        self.waveformextractor = waveformextractor.WaveformExtractor(self.dataio.nb_channel, self.chunksize)
        self.waveformextractor.change_params(n_left=self.n_left, n_right=self.n_right)
    
        self.nb_peak = 0
        
    def process_one_chunk(self, pos, sigs_chunk):

        pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        if preprocessed_chunk is  None:
            return
        
        n_peaks, chunk_peaks = self.peakdetector.process_data(pos2, preprocessed_chunk)
        if chunk_peaks is  None:
            return
        
        for peak_pos, waveforms in self.waveformextractor.new_peaks(pos2, preprocessed_chunk, chunk_peaks):
            #TODO for debug only: remove it:
            assert peak_pos.sahpe[0] == waveforms.shape[0]
            # #
            
            if self.memory_mode=='ram':
                self.all_peak_pos.append(peak_pos)
                self.all_peak_waveforms.append(waveforms)
            elif self.memory_mode=='memmap':
                raise(NotImplementedError)
            
            self.nb_peak += peak_pos.size
    
    
    def finalize(self):
    
        if self.memory_mode=='ram':
            self.all_peak_pos = np.concatenate(self.all_peak_pos, axis=0)
            self.all_peak_waveforms = np.concatenate(self.all_peak_waveforms, axis=0)
            
        elif self.memory_mode=='memmap':
            raise(NotImplementedError)
        
        #TODO inject good_events logic
        
        #PCA
        self.pca = sklearn.decomposition.IncrementalPCA(batch_size=self.pca_batch_size)
        self.pca.fit(self.all_peak_waveforms)
        self.features = self.pca.transform(self.all_peak_waveforms)
        
        self.labels = np.ones(self.nb_peak, dtype='int64')
        
        
    #~ def project(self, method = 'pca', n_components = 5, selection = None):
        #~ #TODO remove peak than are out to avoid PCA polution.
        #~ wf = self.waveforms.values
        #~ if selection is None:
            #~ selection = self.good_events
        
        #~ if method=='pca':
            #~ self._pca = sklearn.decomposition.PCA(n_components = n_components)
            
            #~ self._pca.fit(wf[selection])
            #~ feat = self._pca.transform(wf)
            
            #~ self.features = pd.DataFrame(feat, index = self.waveforms.index, columns = ['pca{}'.format(i) for i in range(n_components)])
            
        #~ return self.features
    
    #~ def reset(self):
        #~ self.cluster_labels = np.unique(self.labels.values)
        #~ self.catalogue = {}
        


    #~ def find_clusters(self, n_clusters,method='kmeans', order_clusters = True,**kargs):
        #~ self.labels = find_clusters(self.features, n_clusters, method='kmeans', **kargs)
        #~ self.labels[~self.good_events] = -1
        
        #~ self.reset()
        #~ if order_clusters:
            #~ self.order_clusters()
        #~ return self.labels

    #~ def merge_cluster(self, label1, label2, order_clusters = True,):
        #~ self.labels[self.labels==label2] = label1
        #~ self.reset()
        #~ if order_clusters:
            #~ self.order_clusters()
        #~ return self.labels
    
    #~ def split_cluster(self, label, n, method='kmeans', order_clusters = True, **kargs):
        #~ mask = self.labels==label
        #~ new_label = find_clusters(self.features[mask], n, method=method, **kargs)
        #~ new_label += max(self.cluster_labels)+1
        #~ self.labels[mask] = new_label
        #~ self.reset()
        #~ if order_clusters:
            #~ self.order_clusters()
        #~ return self.labels
    
    #~ def order_clusters(self):
        #~ """
        #~ This reorder labels from highest power to lower power.
        #~ The higher power the smaller label.
        #~ Negative labels are not reassigned.
        #~ """
        #~ cluster_labels = self.cluster_labels.copy()
        #~ cluster_labels.sort()
        #~ cluster_labels =  cluster_labels[cluster_labels>=0]
        #~ powers = [ ]
        #~ for k in cluster_labels:
            #~ wf = self.waveforms[self.labels==k].values
            #~ #power = np.sum(np.median(wf, axis=0)**2)
            #~ power = np.sum(np.abs(np.median(wf, axis=0)))
            #~ powers.append(power)
        #~ sorted_labels = cluster_labels[np.argsort(powers)[::-1]]
        
        #~ #reassign labels
        #~ N = int(max(cluster_labels)*10)
        #~ self.labels[self.labels>=0] += N
        #~ for new, old in enumerate(sorted_labels+N):
            #~ #self.labels[self.labels==old] = new
            #~ self.labels[self.labels==old] = cluster_labels[new]
        #~ self.reset()
    
    #~ def construct_catalogue(self):
        #~ """
        
        #~ """
        
        #~ self.catalogue = OrderedDict()
        #~ nb_channel = self.waveforms.columns.levels[0].size
        #~ for k in self.cluster_labels:
            #~ # take peak of this cluster
            #~ # and reshaape (nb_peak, nb_channel, nb_csample)
            #~ wf = self.waveforms[self.labels==k].values
            #~ wf = wf.reshape(wf.shape[0], nb_channel, -1)
            
            #~ #compute first and second derivative on dim=2
            #~ kernel = np.array([1,0,-1])/2.
            #~ kernel = kernel[None, None, :]
            #~ wfD =  scipy.signal.fftconvolve(wf,kernel,'same') # first derivative
            #~ wfDD =  scipy.signal.fftconvolve(wfD,kernel,'same') # second derivative
            
            #~ # medians
            #~ center = np.median(wf, axis=0)
            #~ centerD = np.median(wfD, axis=0)
            #~ centerDD = np.median(wfDD, axis=0)
            #~ mad = np.median(np.abs(wf-center),axis=0)*1.4826
            
            #~ #eliminate margin because of border effect of derivative and reshape
            #~ center = center[:, 2:-2]
            #~ centerD = centerD[:, 2:-2]
            #~ centerDD = centerDD[:, 2:-2]
            
            #~ self.catalogue[k] = {'center' : center.reshape(-1), 'centerD' : centerD.reshape(-1), 'centerDD': centerDD.reshape(-1) }
            
            #~ #this is for plotting pupose
            #~ mad = mad[:, 2:-2].reshape(-1)
            #~ self.catalogue[k]['mad'] = mad
            #~ self.catalogue[k]['channel_peak_max'] = np.argmax(np.max(center, axis=1))

        
        #~ return self.catalogue    
    
    #~ def save(self):
        #~ pass




#~ def find_clusters(features, n_clusters, method='kmeans', **kargs):
    #~ if method == 'kmeans':
        #~ km = sklearn.cluster.KMeans(n_clusters=n_clusters,**kargs)
        #~ labels = km.fit_predict(features)
    #~ elif method == 'gmm':
        #~ gmm = sklearn.mixture.GMM(n_components=n_clusters,**kargs)
        #~ labels =gmm.fit_predict(features)
    
    #~ return labels

