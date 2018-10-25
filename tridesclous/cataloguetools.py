"""
Some helping function that apply on catalogueconstructor (apply steps, sumamry, ...)

"""
import time

import matplotlib.pyplot as plt


from .matplotlibplot import plot_centroids


#TODO debug this when no peak at all
def apply_all_catalogue_steps(catalogueconstructor, fullchain_kargs, 
                feat_method, feat_kargs,clust_method, clust_kargs, verbose=True):
    """
    Helper function to call seuquentialy all catalogue steps with dict of params as input.
    Used by offline mainwinwod and OnlineWindow
    
    
    Usage:
      
      catalogueconstructor = CatalogueConstructor(dataio, chan_grp=0)
      
      fullchain_kargs = {
        'duration' : 10,
        'preprocessor' : {
            'highpass_freq' : 400.,
            'lowpass_freq' : 5000.,
            'smooth_size' : 0,
            'chunksize' : 1024,
            'lostfront_chunksize' : 128,
            'signalpreprocessor_engine' : 'numpy',
        },
        'peak_detector' : {
            'peakdetector_engine' : 'numpy',
            'peak_sign' : '-',
            'relative_threshold' : 5.,
            'peak_span' : 0.0002,
        },
        'noise_snippet' : {
            'nb_snippet' : 300,
        },
        'extract_waveforms' : {
            'n_left' : -20,
            'n_right' : 30,
            'mode' : 'rand',
            'nb_max' : 20000,
            'align_waveform' : False,
        },
        'clean_waveforms' : {
            'alien_value_threshold' : 100.,
        },
      }
      feat_method = 'global_pca'
      feat_kargs = {'n_components': 5}
      clust_method = 'sawchaincut'
      clust_kargs = {}
      
      apply_all_catalogue_steps(catalogueconstructor, fullchain_kargs, 
                feat_method, feat_kargs,clust_method, clust_kargs)
      
      
    
    
    """
    
    cc = catalogueconstructor
    
    p = {}
    p.update(fullchain_kargs['preprocessor'])
    p.update(fullchain_kargs['peak_detector'])
    cc.set_preprocessor_params(**p)
    dataio = cc.dataio
    
    #TODO offer noise esatimation duration somewhere
    noise_duration = min(10., fullchain_kargs['duration'], dataio.get_segment_length(seg_num=0)/dataio.sample_rate*.99)
    #~ print('noise_duration', noise_duration)
    t1 = time.perf_counter()
    cc.estimate_signals_noise(seg_num=0, duration=noise_duration)
    t2 = time.perf_counter()
    if verbose:
        print('estimate_signals_noise', t2-t1)
    
    t1 = time.perf_counter()
    cc.run_signalprocessor(duration=fullchain_kargs['duration'])
    t2 = time.perf_counter()
    if verbose:
        print('run_signalprocessor', t2-t1)

    t1 = time.perf_counter()
    cc.extract_some_waveforms(**fullchain_kargs['extract_waveforms'])
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_waveforms', t2-t1)
    
    
    
    t1 = time.perf_counter()
    #~ duration = d['duration'] if d['limit_duration'] else None
    #~ d['clean_waveforms']
    cc.clean_waveforms(**fullchain_kargs['clean_waveforms'])
    t2 = time.perf_counter()
    if verbose:
        print('clean_waveforms', t2-t1)
    
    #~ t1 = time.perf_counter()
    #~ n_left, n_right = cc.find_good_limits(mad_threshold = 1.1,)
    #~ t2 = time.perf_counter()
    #~ print('find_good_limits', t2-t1)

    t1 = time.perf_counter()
    cc.extract_some_noise(**fullchain_kargs['noise_snippet'])
    t2 = time.perf_counter()
    if verbose:
        print('extract_some_noise', t2-t1)
    
    #~ print(cc)
    
    t1 = time.perf_counter()
    cc.extract_some_features(method=feat_method, **feat_kargs)
    t2 = time.perf_counter()
    if verbose:
        print('project', t2-t1)
    
    t1 = time.perf_counter()
    cc.find_clusters(method=clust_method, **clust_kargs)
    t2 = time.perf_counter()
    if verbose:
        print('find_clusters', t2-t1)



summay_template = """
Cluster {label}
Max on channel (abs): {max_on_abs_channel}
Max on channel (local to group): {max_on_channel}
Peak amplitude MAD: {max_peak_amplitude}
Peak amplitude (µV): {max_peak_amplitude_uV}

"""
def summary_clusters(catalogueconstructor, labels=None, label=None):
    cc =catalogueconstructor
    
    if labels is None and label is None:
        labels = cc.positive_cluster_labels
    if label is not None:
        labels = [label]
    
    
    channels = cc.dataio.channel_groups[cc.chan_grp]['channels']
    
    for l, label in enumerate(labels):

        ind = cc.index_of_label(label)
        cluster = cc.clusters[ind]

        max_on_channel = cluster['max_on_channel']
        
        if max_on_channel>=0:
            max_on_abs_channel = channels[max_on_channel]
        else:
            max_on_channel = None
            max_on_abs_channel = None
        
        max_peak_amplitude = cluster['max_peak_amplitude']
        max_peak_amplitude_uV = 'No conversion available'
        if cc.dataio.datasource.bit_to_microVolt is not None and max_on_channel is not None:
            max_peak_amplitude_uV = max_peak_amplitude * cc.signals_mads[max_on_channel] * cc.dataio.datasource.bit_to_microVolt
            
            
        
        d = dict(label=label,
                    max_on_channel=max_on_channel,
                    max_on_abs_channel=max_on_abs_channel,
                    max_peak_amplitude=max_peak_amplitude,
                    max_peak_amplitude_uV=max_peak_amplitude_uV,
                    )
        text = summay_template.format(**d)
        
        print(text)
        
        fig, axs = plt.subplots(ncols=2)
        plot_centroids(cc, labels=[label,], ax=axs[0])
        axs[0].set_title('cluster {}'.format(label))
        
        

    
    
    
    
    


