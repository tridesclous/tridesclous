import os
import json
from collections import OrderedDict
import time
import pickle

import numpy as np
import scipy.signal
import scipy.interpolate
import seaborn as sns

from . import signalpreprocessor
from . import  peakdetector
from . import waveformextractor
from . import decomposition
from . import cluster 

from .tools import median_mad


from .iotools import ArrayCollection

import matplotlib.pyplot as plt

# TODO auto cut left rigth after first cut
# TODO improve different of sample waveform
# TODO make a clear distinct between all peak and sample peak/waveform/features (because different sizes) CF ndscatter!!!!!
# TODO make some label for insorted type


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
                    'peak_waveforms', 'peak_waveforms_index', 'features',
                    'signals_medians','signals_mads', ]:
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
            peak_sign='-', relative_threshold=7, peak_span=0.0005,
            
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
        self.dataio.reset_processed_signals(dtype=internal_dtype)
        
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
        
        
        signals_medians = np.median(filtered_sigs, axis=0)
        signals_mads = np.median(np.abs(filtered_sigs-signals_medians),axis=0)*1.4826
        
        self.arrays.detach_array(name)
        
        #make theses persistant
        self.arrays.create_array('signals_medians', self.info['internal_dtype'], signals_medians.shape, 'memmap')
        self.arrays.create_array('signals_mads', self.info['internal_dtype'], signals_mads.shape, 'memmap')
        self.signals_medians[:] = signals_medians
        self.signals_mads[:] = signals_mads


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
        p['signals_medians'] = self.signals_medians
        p['signals_mads'] = self.signals_mads
        self.signalpreprocessor.change_params(**p)
        
        self.peakdetector.change_params(**self.params_peakdetector)
        
        iterator = self.dataio.iter_over_chunk(seg_num=seg_num, chunksize=self.chunksize, i_stop=length,
                                                    signal_type='initial', return_type='raw_numpy')
        for pos, sigs_chunk in iterator:
            #~ print(seg_num, pos, sigs_chunk.shape)
            self.signalprocessor_one_chunk(pos, sigs_chunk, seg_num)
        
        
        
        
    
    def finalize_signalprocessor_loop(self):
        self.dataio.flush_processed_signals()
        
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
        

    def project(self, method='pca', selection = None, n_components=5, **params):
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
        
        self.on_new_cluster()
        
        #~ if order_clusters:
            #~ self.order_clusters()
    
    def on_new_cluster(self):
        self.cluster_labels = np.unique(self.peak_label)
    
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
        
    #~ def merge_cluster(self, labels_to_merge):
        #~ #TODO: maybe take the first cluster label instead of new one (except -1)
        #~ new_label = max(self.cluster_labels)+1
        #~ for k in labels_to_merge:
            #~ take = self.peak_label == k
            #~ self.peak_label[take] = new_label
        
        #~ self.on_new_cluster()
        
        #~ self.on_new_cluster(label_changed=labels_to_merge+[new_label])
    
    def split_cluster(self, label, n, method='kmeans', order_clusters=True, **kargs):
        mask = self.peak_label==label
        self.find_clusters(method=method, n_clusters=n, order_clusters=order_clusters, selection=mask, **kargs)
    
    def trash_small_cluster(self, n=10):
        for k in self.cluster_labels:
            mask = self.peak_label==k
            if np.sum(mask)<=n:
                self.peak_label[mask] = -1
        self.on_new_cluster()
        
        
    
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
    
    def make_catalogue(self):
        #TODO: offer possibility to resample some waveforms or choose the number
        
        t1 = time.perf_counter()
        self.catalogue = {}
        
        self.catalogue = {}
        self.catalogue['n_left'] = self.info['params_waveformextractor']['n_left'] +2
        self.catalogue['n_right'] = self.info['params_waveformextractor']['n_right'] -2
        self.catalogue['peak_width'] = self.catalogue['n_right'] - self.catalogue['n_left']
        
        cluster_labels = self.cluster_labels[self.cluster_labels>=0]
        self.catalogue['cluster_labels'] = cluster_labels
        
        n, full_width, nchan = self.peak_waveforms.shape
        centers0 = np.zeros((cluster_labels.size, full_width - 4, nchan), dtype=self.info['internal_dtype'])
        centers1 = np.zeros_like(centers0)
        centers2 = np.zeros_like(centers0)
        self.catalogue['centers0'] = centers0 # median of wavforms
        self.catalogue['centers1'] = centers1 # median of first derivative of wavforms
        self.catalogue['centers2'] = centers2 # median of second derivative of wavforms
        
        subsample = np.arange(1.5, full_width-2.5, 1/20.)
        self.catalogue['subsample_ratio'] = 20
        interp_centers0 = np.zeros((cluster_labels.size, subsample.size, nchan), dtype=self.info['internal_dtype'])
        self.catalogue['interp_centers0'] = interp_centers0
        
        print('peak_width', self.catalogue['peak_width'])
        
        for i, k in enumerate(cluster_labels):
            #print('construct_catalogue', k)
            # take peak of this cluster
            # and reshaape (nb_peak, nb_channel, nb_csample)
            #~ wf = self.peak_waveforms[self.peak_label==k]
            wf0 = self.peak_waveforms[self.peak_label[self.peak_waveforms_index]==k]
            
            
            #compute first and second derivative on dim=1 (time)
            kernel = np.array([1,0,-1])/2.
            kernel = kernel[None, :, None]
            wf1 =  scipy.signal.fftconvolve(wf0,kernel,'same') # first derivative
            wf2 =  scipy.signal.fftconvolve(wf1,kernel,'same') # second derivative
            
            #median and
            #eliminate margin because of border effect of derivative and reshape
            #~ center0 = np.median(wf0, axis=0)
            #~ centers0[i,:,:] = center0[2:-2, :]
            #~ centers1[i,:,:] = np.median(wf1, axis=0)[2:-2, :]
            #~ centers2[i,:,:] = np.median(wf2, axis=0)[2:-2, :]
            center0 = np.mean(wf0, axis=0)
            centers0[i,:,:] = center0[2:-2, :]
            centers1[i,:,:] = np.mean(wf1, axis=0)[2:-2, :]
            centers2[i,:,:] = np.mean(wf2, axis=0)[2:-2, :]

            #interpolate centers0 for reconstruction inbetween bsample when jitter is estimated
            f = scipy.interpolate.interp1d(np.arange(full_width), center0, axis=0, kind='cubic')
            oversampled_center = f(subsample)
            interp_centers0[i, :, :] = oversampled_center
            
            #~ fig, ax = plt.subplots()
            #~ ax.plot(np.arange(full_width-4), center0[2:-2, :], color='b', marker='o')
            #~ ax.plot(subsample-2.,oversampled_center, color='c')
            #~ plt.show()
        

        #find max  channel for each cluster for peak alignement
        self.catalogue['max_on_channel'] = np.zeros_like(self.catalogue['cluster_labels'])
        for i, k in enumerate(cluster_labels):
            center = self.catalogue['centers0'][i]
            self.catalogue['max_on_channel'][i] = np.argmax(np.max(np.abs(center), axis=0))
        
        #colors
        if not hasattr(self, 'colors'):
            self.refresh_colors()
        self.catalogue['cluster_colors'] = {}
        self.catalogue['cluster_colors'].update(self.colors)
        
        #params
        self.catalogue['params_signalpreprocessor'] = dict(self.info['params_signalpreprocessor'])
        self.catalogue['params_peakdetector'] = dict(self.info['params_peakdetector'])
        self.catalogue['signals_medians'] = self.signals_medians
        self.catalogue['signals_mads'] = self.signals_mads        

        
        
        t2 = time.perf_counter()
        print('construct_catalogue', t2-t1)
        
        return self.catalogue
    
    def save_catalogue(self):
        self.make_catalogue()
        
        filename = os.path.join(self.catalogue_path, 'initial_catalogue.pickle')
        with open(filename, mode='wb') as f:
            pickle.dump(self.catalogue, f)
    
    def load_catalogue(self):
        filename = os.path.join(self.catalogue_path, 'initial_catalogue.pickle')
        with open(filename, mode='rb') as f:
            self.catalogue = pickle.load(f)
        return self.catalogue

