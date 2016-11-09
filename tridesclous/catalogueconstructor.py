import os
import json
from collections import OrderedDict
import time

import numpy as np
import scipy.signal
import seaborn as sns

#~ import sklearn
#~ import sklearn.decomposition
#~ import sklearn.cluster
#~ import sklearn.mixture

from . import signalpreprocessor
from . import  peakdetector
from . import waveformextractor
from . import decomposition
from . import cluster 

from .tools import median_mad

from pyqtgraph.Qt import QtCore, QtGui

from .iotools import ArrayCollection

class CatalogueConstructor:
    """
    CatalogueConstructor scan a smal part of the dataset to construct the catalogue.
    Catalogue are the centroid (median+mad) of each cluster.
    
    
    """
    def __init__(self, dataio, name='initial_catalogue'):
        self.dataio = dataio
        
        self.catalogue_path = os.path.join(self.dataio.dirname, name)
        if not os.path.exists(self.catalogue_path):
            os.mkdir(self.catalogue_path)
        #~ print('self.catalogue_path', self.catalogue_path)
        self.arrays = ArrayCollection(parent=self, dirname=self.catalogue_path)
        
        self.info_filename = os.path.join(self.catalogue_path, 'info.json')
        if not os.path.exists(self.info_filename):
            #first init
            self.info = {}
            self.flush_info()
        else:
            with open(self.info_filename, 'r', encoding='utf8') as f:
                self.info = json.load(f)
        
        for name in ['peak_pos', 'peak_segment', 'peak_label',
                    'peak_waveforms', 'peak_waveforms_index', 'features']:
            # this set attribute to class if exsits
            self.arrays.load_if_exists(name)
        #~ print(' __init__ self.peak_pos', self.peak_pos)
        if self.peak_pos is not None:
            self.nb_peak = self.peak_pos.size
            self.memory_mode='memmap'
    
    def flush_info(self):
        with open(self.info_filename, 'w', encoding='utf8') as f:
            json.dump(self.info, f, indent=4)
    
    
    def initialize_signalprocessor_loop(self, chunksize=1024,
            memory_mode='ram',
            
            internal_dtype = 'float32',
            
            #signal preprocessor
            signalpreprocessor_engine='signalpreprocessor_numpy',
            highpass_freq=300, backward_chunksize=1280,
            common_ref_removal=True,
            
            #peak detector
            peakdetector_engine='peakdetector_numpy',
            peak_sign='-', relative_threshold=5, peak_span=0.0005,
            
            ):
        
        #TODO remove stuff if already computed
        
        
        self.chunksize = chunksize
        self.memory_mode = memory_mode
        
        for name, dtype in [('peak_pos', 'int64'),
                                        ('peak_segment', 'int64'),
                                        ('peak_label', 'int32')]:
            self.arrays.initialize_array(name, self.memory_mode,  dtype, (-1, ))
        
        
        self.params_signalpreprocessor = dict(highpass_freq=highpass_freq, backward_chunksize=backward_chunksize,
                    common_ref_removal=common_ref_removal, output_dtype=internal_dtype)
        SignalPreprocessor_class = signalpreprocessor.signalpreprocessor_engines[signalpreprocessor_engine]
        self.signalpreprocessor = SignalPreprocessor_class(self.dataio.sample_rate, self.dataio.nb_channel, chunksize, self.dataio.dtype)
        
        
        self.params_peakdetector = dict(peak_sign=peak_sign, relative_threshold=relative_threshold, peak_span=peak_span)
        PeakDetector_class = peakdetector.peakdetector_engines[peakdetector_engine]
        self.peakdetector = PeakDetector_class(self.dataio.sample_rate, self.dataio.nb_channel,
                                                        self.chunksize, internal_dtype)
        
        #TODO make processed data as int32 ???
        self.dataio.reset_signals(signal_type='processed', dtype='float32')
        
        self.nb_peak = 0
        
        #TODO put all params in info
        self.info['internal_dtype'] = internal_dtype
        self.info['params_signalpreprocessor'] = self.params_signalpreprocessor
        self.info['params_peakdetector'] = self.params_peakdetector
        self.flush_info()
    
    
    def estimate_signals_noise(self, seg_num=0, duration=10.):
        
        length = int(duration*self.dataio.sample_rate)
        length -= length%self.chunksize
        
        name = 'filetered_sigs_for_noise_estimation_seg_{}'.format(seg_num)
        shape=(length, self.dataio.nb_channel)
        filtered_sigs = self.arrays.create_array(name, self.info['internal_dtype'], shape, 'memmap')
        
        params2 = dict(self.params_signalpreprocessor)
        params2['normalize'] = False
        self.signalpreprocessor.change_params(**params2)
        
        
        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chunksize=self.chunksize, i_stop=length,
                                                    signal_type='initial',  return_type='raw_numpy')
        for pos, sigs_chunk in iterator:
            pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
            if preprocessed_chunk is not None:
                filtered_sigs[pos2-preprocessed_chunk.shape[0]:pos2, :] = preprocessed_chunk
        
        
        self.signals_medians = medians= np.median(filtered_sigs, axis=0)
        self.signals_mads = np.median(np.abs(filtered_sigs-medians),axis=0)*1.4826
        
        self.arrays.detach_array(name)

        
    def signalprocessor_one_chunk(self, pos, sigs_chunk, seg_num):

        pos2, preprocessed_chunk = self.signalpreprocessor.process_data(pos, sigs_chunk)
        if preprocessed_chunk is  None:
            return
        
        self.dataio.set_signals_chunk(preprocessed_chunk, seg_num=seg_num, i_start=pos2-preprocessed_chunk.shape[0],
                        i_stop=pos2, signal_type='processed')
        
        n_peaks, chunk_peaks = self.peakdetector.process_data(pos2, preprocessed_chunk)
        if chunk_peaks is  None:
            chunk_peaks = np.array([], dtype='int64')
        
        peak_pos = chunk_peaks
        peak_segment = np.ones(peak_pos.size, dtype='int64') * seg_num
        peak_label = np.zeros(peak_pos.size, dtype='int32')
        
        self.arrays.append_chunk('peak_pos',  peak_pos)
        self.arrays.append_chunk('peak_label',  peak_label)
        self.arrays.append_chunk('peak_segment',  peak_segment)

        self.nb_peak += peak_pos.size


        
    def run_signalprocessor_loop(self, seg_num=0, duration=60.):
        
        length = int(duration*self.dataio.sample_rate)
        length -= length%self.chunksize
        #initialize engines
        
        p = dict(self.params_signalpreprocessor)
        p['normalize'] = True
        p['medians'] = self.signals_medians
        p['mads'] = self.signals_mads
        self.signalpreprocessor.change_params(**p)
        
        self.peakdetector.change_params(**self.params_peakdetector)
        
        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chunksize=self.chunksize, i_stop=length,
                                                    signal_type='initial', return_type='raw_numpy')
        for pos, sigs_chunk in iterator:
            #~ print(seg_num, pos, sigs_chunk.shape)
            self.signalprocessor_one_chunk(pos, sigs_chunk, seg_num)
        
        self.dataio.flush_signals(seg_num=seg_num)
        
        
    
    def finalize_signalprocessor_loop(self):
        
        self.arrays.finalize_array('peak_pos')
        self.arrays.finalize_array('peak_label')
        self.arrays.finalize_array('peak_segment')
        #~ self.on_new_cluster()
    
    
    def extract_some_waveforms(self, n_left=-20, n_right=30,  nb_max=10000):
        
        peak_width = - n_left + n_right

        if self.nb_peak>nb_max:
            take_mask = np.zeros(self.nb_peak, dtype='bool')
            take_index = np.random.choice(self.nb_peak, size=nb_max).astype('int64')
            take_mask[take_index] = True
        else:
            take_index = np.arange(self.nb_peak, dtype='int64')
            take_mask = np.ones(self.nb_peak, dtype='bool')
        
        nb_peak_waveforms = take_index.size
        self.arrays.create_array('peak_waveforms_index', 'int64', (nb_peak_waveforms,), self.memory_mode)
        self.peak_waveforms_index[:] = take_index
        
        shape=(nb_peak_waveforms, peak_width, self.dataio.nb_channel)
        self.arrays.create_array('peak_waveforms', self.info['internal_dtype'], shape, self.memory_mode)

        seg_nums = np.unique(self.peak_segment)
        n = 0
        for seg_num in seg_nums:
            insegmen_peak_pos = self.peak_pos[take_mask & (self.peak_segment==seg_num)]
            for pos in insegmen_peak_pos:
                i_start = pos+n_left
                i_stop = i_start+peak_width
                wf = self.dataio.get_signals_chunk(seg_num=seg_num, i_start=i_start, i_stop=i_stop, signal_type='processed')
                self.peak_waveforms[n, :, :] = wf
                n +=1
                #~ print(n, seg_num)
        
        self.info['params_waveformextractor'] = dict(n_left=n_left, n_right=n_right,  nb_max=nb_max)
        self.flush_info()
        

    def project(self, method='IncrementalPCA', selection = None, n_components=5, **params):
        """
        params:
        n_components
        
        """
        wf = self.peak_waveforms.reshape(self.peak_waveforms.shape[0], -1)
        params['n_components'] = n_components
        features = decomposition.project_waveforms(self.peak_waveforms, method=method, selection=None,
                    catalogueconstructor=self, **params)
        
        #trick to make it persistant
        self.arrays.create_array('features', self.info['internal_dtype'], features.shape, self.memory_mode)
        self.features[:] = features
    
    
    def find_clusters(self, method='kmeans', n_clusters=1, order_clusters=True, selection=None, **kargs):
        #TODO clustering on all peaks
        
        if selection is None:
            labels = cluster.find_clusters(self.features, method=method, n_clusters=n_clusters, **kargs)
            self.peak_label[self.peak_waveforms_index] = labels
        else:
            
            sel = selection[self.peak_waveforms_index]
            features = self.features[sel]
            labels = cluster.find_clusters(features, method=method, n_clusters=n_clusters, **kargs)
            labels += max(self.cluster_labels)+1
            self.peak_label[sel] = labels
        
        self.on_new_cluster(label_changed=None)
        
        #~ if order_clusters:
            #~ self.order_clusters()
    
    
    def on_new_cluster(self, label_changed=None):
        """
        label_changed can be remove/add/modify
        """
        if self.peak_pos==[]: return
        
        self.cluster_labels = np.unique(self.peak_label)
        
        if label_changed is None:
            #re count evry clusters
            self.cluster_count = { k:np.sum(self.peak_label==k) for k in self.cluster_labels}
        else:
            for k in label_changed:
                if k in self.cluster_labels:
                    self.cluster_count[k] = np.sum(self.peak_label==k)
                else:
                    self.cluster_count.pop(k)
        
        #TODO
        self.compute_centroid(label_changed=label_changed)
        
        self._check_plot_attributes()

    def _check_plot_attributes(self):
        if not hasattr(self, 'peak_selection'):
            self.peak_selection = np.zeros(self.nb_peak, dtype='bool')
        
        if not hasattr(self, 'cluster_visible'):
            self.cluster_visible = {}
        
        for k in self.cluster_labels:
            if k not in self.cluster_visible:
                self.cluster_visible[k] = True
        for k in list(self.cluster_visible.keys()):
            if k not in self.cluster_labels:
                self.cluster_visible.pop(k)
        
        if not hasattr(self, 'cluster_colors'):
            self.refresh_colors(reset=True)
        else:
            self.refresh_colors(reset=False)

    
    def compute_centroid(self, label_changed=None):
        if label_changed is None:
            # recompute all clusters
            self.centroids = {}
            label_changed = self.cluster_labels
        
        t1 = time.perf_counter()
        for k in label_changed:
            if k not in self.cluster_labels:
                self.centroids.pop(k)
                continue
            wf = self.peak_waveforms[self.peak_label[self.peak_waveforms_index]==k]
            median, mad = median_mad(wf, axis = 0)
            mean, std = np.mean(wf, axis=0), np.std(wf, axis=0)
            max_on_channel = np.argmax(np.max(np.abs(mean), axis=0))
            
            self.centroids[k] = {'median':median, 'mad':mad, 'max_on_channel' : max_on_channel, 
                        'mean': mean, 'std': std}
        
        t2 = time.perf_counter()
        print('compute_centroid', t2-t1)
        
    
    def refresh_colors(self, reset=True, palette = 'husl'):
        if reset:
            self.colors = {}
        
        n = self.cluster_labels.size
        color_table = sns.color_palette(palette, n)
        for i, k in enumerate(self.cluster_labels):
            if k not in self.colors:
                self.colors[k] = color_table[i]
        
        self.colors[-1] = (.4, .4, .4)
        
        self.qcolors = {}
        for k, color in self.colors.items():
            r, g, b = color
            self.qcolors[k] = QtGui.QColor(r*255, g*255, b*255)

    def merge_cluster(self, labels_to_merge, order_clusters =False,):
        #TODO: maybe take the first cluster label instead of new one (except -1)
        new_label = max(self.cluster_labels)+1
        for k in labels_to_merge:
            take = self.peak_label == k
            self.peak_label[take] = new_label

        if order_clusters:
            self.order_clusters()
        else:
            self.on_new_cluster(label_changed=labels_to_merge+[new_label])
    
    
    def split_cluster(self, label, n, method='kmeans', order_clusters=True, **kargs):
        mask = self.peak_label==label
        self.find_clusters(method=method, n_clusters=n, order_clusters=order_clusters, selection=mask, **kargs)
    
    def order_clusters(self):
        """
        This reorder labels from highest power to lower power.
        The higher power the smaller label.
        Negative labels are not reassigned.
        """
        cluster_labels = self.cluster_labels.copy()
        cluster_labels.sort()
        cluster_labels =  cluster_labels[cluster_labels>=0]
        powers = [ ]
        for k in cluster_labels:
            power = np.sum(self.centroids[k]['median'].flatten()**2)
            powers.append(power)
        sorted_labels = cluster_labels[np.argsort(powers)[::-1]]
        
        #reassign labels
        N = int(max(cluster_labels)*10)
        self.peak_label[self.peak_label>=0] += N
        for new, old in enumerate(sorted_labels+N):
            #~ self.peak_label[self.peak_label==old] = cluster_labels[new]
            self.peak_label[self.peak_label==old] = new
        
        self.on_new_cluster()
    
    def construct_catalogue(self):
        #TODO
        pass
        #~ t1 = time.perf_counter()
        #~ self.catalogue = OrderedDict()
        #~ nb_channel = self.dataio.nb_channel
        #~ for k in self.cluster_labels:
            #~ #print('construct_catalogue', k)
            #~ # take peak of this cluster
            #~ # and reshaape (nb_peak, nb_channel, nb_csample)
            #~ wf = self.peak_waveforms[self.peak_label==k]
            
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
        
        #~ t2 = time.perf_counter()
        #~ print('construct_catalogue', t2-t1)
        
        #~ return self.catalogue


