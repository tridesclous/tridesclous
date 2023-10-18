import numpy as np

import sklearn
import sklearn.decomposition
import sklearn.cluster
import sklearn.manifold
import sklearn.discriminant_analysis

from . import tools

import joblib

import time

def project_waveforms(method='pca_by_channel', catalogueconstructor=None, selection=None, **params):
    """
    If slection is None then the fit of projector is done on all some_peaks_index
    
    Otherwise the fit is done on the a subset of waveform force by slection bool mask
    
    selection is mask bool size all_peaks
    """
    cc = catalogueconstructor
    
    
    if method=='global_pca':
        projector = GlobalPCA(catalogueconstructor=cc, selection=selection, **params)
    elif method=='peak_max':
        projector = PeakMaxOnChannel(catalogueconstructor=cc, selection=selection, **params)
    elif method=='pca_by_channel':
        projector = PcaByChannel(catalogueconstructor=cc, selection=selection, **params)
    #~ elif method=='neighborhood_pca':
        #~ projector = NeighborhoodPca(waveforms, catalogueconstructor=catalogueconstructor, **params)
    elif method=='global_lda':
        projector = GlobalLDA(catalogueconstructor=cc, selection=selection, **params)
    else:
        raise NotImplementedError
    
    #~ features = projector.transform(waveforms2)
    features = projector.get_features(catalogueconstructor)
    channel_to_features = projector.channel_to_features
    return features, channel_to_features, projector


class GlobalPCA:
    def __init__(self, catalogueconstructor=None, selection=None, n_components=5, **params):

        cc = catalogueconstructor
        
        self.n_components = n_components
        
        self.waveforms = cc.get_some_waveforms()
        
        if selection is None:
            waveforms = self.waveforms
            #~ print('all selection', waveforms.shape[0])
        else:
            peaks_index, = np.nonzero(selection)
            waveforms = cc.get_some_waveforms(peaks_index=peaks_index)
            #~ print('subset selection', waveforms.shape[0])
        
        #~ print(waveforms.shape)
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        #~ self.pca =  sklearn.decomposition.IncrementalPCA(n_components=n_components, **params)
        self.pca =  sklearn.decomposition.TruncatedSVD(n_components=n_components, **params)
        self.pca.fit(flatten_waveforms)
        
        
        #In GlobalPCA all feature represent all channels
        self.channel_to_features = np.ones((cc.nb_channel, self.n_components), dtype='bool')

    def get_features(self, catalogueconstructor):
        features = self.transform(self.waveforms)
        del self.waveforms
        return features

    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        return self.pca.transform(flatten_waveforms)

class PeakMaxOnChannel:
    def __init__(self,  catalogueconstructor=None, selection=None, **params):
        if selection is not None:
            print('selection with PeakMaxOnChannel is a non sens')
            
        cc = catalogueconstructor
        
        #~ self.waveforms = waveforms
        # TODO something faster with only the max!!!!!
        self.waveforms = cc.get_some_waveforms()
        
        self.ind_peak = -catalogueconstructor.info['extract_waveforms']['n_left']
        #~ print('PeakMaxOnChannel self.ind_peak', self.ind_peak)
        
        
        #In full PeakMaxOnChannel one feature is one channel
        self.channel_to_features = np.eye(cc.nb_channel, dtype='bool')
    
    def get_features(self, catalogueconstructor):
        features = self.transform(self.waveforms)
        del self.waveforms
        return features
    
        
    def transform(self, waveforms):
        #~ print('ici', waveforms.shape, self.ind_peak)
        features = waveforms[:, self.ind_peak, : ].copy()
        return features



#~ Parallel(n_jobs=n_jobs)(delayed(count_match_spikes)(sorting1.get_unit_spike_train(u1),
                                                                                  #~ s2_spiketrains, delta_frames) for
                                                      #~ i1, u1 in enumerate(unit1_ids))

#~ def get_pca_one_channel(wf_chan, chan, thresh, n_left, n_components_by_channel, params):
    #~ print(chan)
    #~ pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
    #~ wf_chan = waveforms[:,:,chan]
    #~ print(wf_chan.shape)
    #~ print(wf_chan[:, -n_left].shape)
    #~ keep = np.any((wf_chan>thresh) | (wf_chan<-thresh))
    #~ keep = (wf_chan[:, -n_left]>thresh) | (wf_chan[:, -n_left]<-thresh)

    #~ if keep.sum() >=n_components_by_channel:
        #~ pca.fit(wf_chan[keep, :])
        #~ return pca
    #~ else:
        #~ return None


class PcaByChannel:
    def __init__(self, catalogueconstructor=None, selection=None, n_components_by_channel=3, adjacency_radius_um=200, **params):
        
        cc = catalogueconstructor
        
        thresh = cc.info['peak_detector']['relative_threshold']
        n_left = cc.info['extract_waveforms']['n_left']
        self.dtype = cc.info['internal_dtype']
        
        
        #~ self.waveforms = waveforms
        self.n_components_by_channel = n_components_by_channel
        self.adjacency_radius_um = adjacency_radius_um
        
        
        #~ t1 = time.perf_counter()
        if selection is None:
            peaks_index = cc.some_peaks_index
        else:
            peaks_index,  = np.nonzero(selection)
        
        some_peaks = cc.all_peaks[peaks_index]
        
        self.pcas = []
        for chan in range(cc.nb_channel):
        #~ for chan in range(20):
            #~ print('fit', chan)
            sel = some_peaks['channel'] == chan
            wf_chan = cc.get_some_waveforms(peaks_index=peaks_index[sel], channel_indexes=[chan])
            wf_chan = wf_chan[:, :, 0]
            #~ print(wf_chan.shape)
            
            if wf_chan.shape[0] - 1 > n_components_by_channel:
                #~ pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_channel, **params)
                #~ print('PcaByChannel SVD')
                pca = sklearn.decomposition.TruncatedSVD(n_components=n_components_by_channel, **params)
                #~ print(wf_chan.shape)
                #~ print(np.sum(np.isnan(wf_chan)))
                #~ import matplotlib.pyplot as plt
                #~ fig, ax = plt.subplots()
                #~ ax.plot(wf_chan.T)
                #~ plt.show()
                try:
                    pca.fit(wf_chan)
                except ValueError:
                    pca = None
                    print('Error in PcaByChannel for channel {} maybe too noisy (Nan, Inf,)'.format(chan))
            else:
                pca = None
            self.pcas.append(pca)

        #~ t2 = time.perf_counter()
        #~ print('pca fit', t2-t1)
            


            
            #~ pca = get_pca_one_channel(waveforms, chan, thresh, n_left, n_components_by_channel, params)
            
        #~ n_jobs = -1
        #~ self.pcas = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_pca_one_channel)(waveforms, chan, thresh, n_components_by_channel, params) for chan in range(cc.nb_channel))
        

        #In full PcaByChannel n_components_by_channel feature correspond to one channel
        self.channel_to_features = np.zeros((cc.nb_channel, cc.nb_channel*n_components_by_channel), dtype='bool')
        for c in range(cc.nb_channel):
            self.channel_to_features[c, c*n_components_by_channel:(c+1)*n_components_by_channel] = True

    def get_features(self, catalogueconstructor):
        cc = catalogueconstructor
        
        nb = cc.some_peaks_index.size
        n = self.n_components_by_channel
        
        features = np.zeros((nb, cc.nb_channel*self.n_components_by_channel), dtype=self.dtype)
        
        some_peaks = cc.all_peaks[cc.some_peaks_index]

        if cc.mode == 'sparse':
            assert cc.info['peak_detector']['method'] == 'geometrical'
            #~ adjacency_radius_um = cc.info['peak_detector']['adjacency_radius_um']
            channel_adjacency = cc.dataio.get_channel_adjacency(chan_grp=cc.chan_grp, adjacency_radius_um=self.adjacency_radius_um)
        
        #~ t1 = time.perf_counter()
        for chan, pca in enumerate(self.pcas):
            if pca is None:
                continue
            #~ print('transform', chan)
            #~ sel = some_peaks['channel'] == chan
            
            if cc.mode == 'dense':
                wf_chan = cc.get_some_waveforms(peaks_index=cc.some_peaks_index, channel_indexes=[chan])
                wf_chan = wf_chan[:, :, 0]
                #~ print('dense', wf_chan.shape)
                features[:, chan*n:(chan+1)*n] = pca.transform(wf_chan)
            elif cc.mode == 'sparse':
                sel = np.isin(some_peaks['channel'], channel_adjacency[chan])
                #~ print(chan, np.sum(sel))
                wf_chan = cc.get_some_waveforms(peaks_index=cc.some_peaks_index[sel], channel_indexes=[chan])
                wf_chan = wf_chan[:, :, 0]
                #~ print('sparse', wf_chan.shape)
                features[:, chan*n:(chan+1)*n][sel, :] = pca.transform(wf_chan)

        #~ t2 = time.perf_counter()
        #~ print('pca transform', t2-t1)
            
            
        return features
        
    
    def transform(self, waveforms):
        n = self.n_components_by_channel
        all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=self.dtype)
        for c, pca in enumerate(self.pcas):
            if pca is None:
                continue
            #~ print(c)
            all[:, c*n:(c+1)*n] = pca.transform(waveforms[:, :, c])
        return all



class GlobalLDA:
    def __init__(self, catalogueconstructor=None, selection=None, **params):

        cc = catalogueconstructor
        
        self.waveforms = cc.get_some_waveforms()
        
        if selection is None:
            #~ waveforms = self.waveforms
            raise NotImplementedError
        else:
            peaks_index, = np.nonzero(selection)
            waveforms = cc.get_some_waveforms(peaks_index=peaks_index)
            labels = cc.all_peaks[peaks_index]['cluster_label']
        
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        
        self.lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        self.lda.fit(flatten_waveforms, labels)
        
        
        #In GlobalPCA all feature represent all channels
        self.channel_to_features = np.ones((cc.nb_channel, self.lda._max_components), dtype='bool')

    def get_features(self, catalogueconstructor):
        features = self.transform(self.waveforms)
        del self.waveforms
        return features


    def transform(self, waveforms):
        flatten_waveforms = waveforms.reshape(waveforms.shape[0], -1)
        return self.lda.transform(flatten_waveforms)



#~ class NeighborhoodPca:
    #~ def __init__(self, waveforms, catalogueconstructor=None, n_components_by_neighborhood=6, radius_um=300., **params):
        #~ cc = catalogueconstructor
        
        #~ self.n_components_by_neighborhood = n_components_by_neighborhood
        #~ self.neighborhood = tools.get_neighborhood(cc.geometry, radius_um)
        
        #~ self.pcas = []
        #~ for c in range(cc.nb_channel):
            #~ neighbors = self.neighborhood[c, :]
            #~ pca = sklearn.decomposition.IncrementalPCA(n_components=n_components_by_neighborhood, **params)
            #~ wfs = waveforms[:,:,neighbors]
            #~ wfs = wfs.reshape(wfs.shape[0], -1)
            #~ pca.fit(wfs)
            #~ self.pcas.append(pca)

        #~ #In full NeighborhoodPca n_components_by_neighborhood feature correspond to one channel
        #~ self.channel_to_features = np.zeros((cc.nb_channel, cc.nb_channel*n_components_by_neighborhood), dtype='bool')
        #~ for c in range(cc.nb_channel):
            #~ self.channel_to_features[c, c*n_components_by_neighborhood:(c+1)*n_components_by_neighborhood] = True

    #~ def transform(self, waveforms):
        #~ n = self.n_components_by_neighborhood
        #~ all = np.zeros((waveforms.shape[0], waveforms.shape[2]*n), dtype=waveforms.dtype)
        #~ for c, pca in enumerate(self.pcas):
            #~ neighbors = self.neighborhood[c, :]
            #~ wfs = waveforms[:,:,neighbors]
            #~ wfs = wfs.reshape(wfs.shape[0], -1)
            #~ all[:, c*n:(c+1)*n] = pca.transform(wfs)
        #~ return all

