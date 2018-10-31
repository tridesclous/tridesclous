

import matplotlib.pyplot as plt
import numpy as np

from .catalogueconstructor import CatalogueConstructor
from .matplotlibplot import plot_centroids, plot_waveforms_histogram, plot_isi




_summary_catalogue_clusters_template = """
Cluster {label}
Max on channel (abs): {max_on_abs_channel}
Max on channel (local to group): {max_on_channel}
Peak amplitude MAD: {max_peak_amplitude}
Peak amplitude (µV): {max_peak_amplitude_uV}

"""
def summary_catalogue_clusters(dataio, chan_grp=None, labels=None, label=None):
    cc =CatalogueConstructor(dataio, chan_grp=chan_grp)
    
    if labels is None and label is None:
        labels = cc.positive_cluster_labels
    if label is not None:
        labels = [label]
    
    channels = dataio.channel_groups[chan_grp]['channels']
    
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
        text = _summary_catalogue_clusters_template.format(**d)
        
        print(text)
        
        fig, axs = plt.subplots(ncols=2)
        plot_centroids(cc, labels=[label,], ax=axs[0])
        axs[0].set_title('cluster {}'.format(label))
        
        if dataio.datasource.bit_to_microVolt is None:
            units = 'MAD'
            title = 'Amplitude in MAD (STD) ratio'
        else:
            units = 'uV'
            title = 'Amplitude μV'
        
        plot_waveforms_histogram(cc, label=label, ax=axs[1], channels=[max_on_channel], units=units)
        axs[1].set_title('Hist on channel {}'.format(max_on_abs_channel))
        axs[1].set_ylabel(title)



_summary_noise_template = """
Noise mesured with mesured with MAD (=robust STD)
  Channel group {chan_grp}
  Nb_channel:  {nb_chan}
  Noise range: {min_mad:.2f} - {max_mad:.2f} μV
  By channel noise: {reduce_noise_vector} μV
  Average noise along channel: {noise_mean}  μV
  Threshold:  {relative_threshold} *MAD = {thresh_uV} μV
"""

def summary_noise(dataio, chan_grp=None):
    cc =CatalogueConstructor(dataio, chan_grp=chan_grp)
    
    channels = dataio.channel_groups[chan_grp]['channels']
    
    assert cc.dataio.datasource.bit_to_microVolt is not None, 'bit_to_microVolt is not provided for this data source'
    
    noise_uV = cc.signals_mads * cc.dataio.datasource.bit_to_microVolt
    
    noise_txt = [ '{:.1f}'.format(e) for e in noise_uV]
    
    if len(channels)>8:
        reduce_noise_vector = "[{} ... {}]".format(' '.join(noise_txt[:4]),' '.join(noise_txt[-4:]))
    else:
        reduce_noise_vector = "[{}]".format(' '.join(noise_txt))
    
    
    relative_threshold = cc.info['peak_detector_params']['relative_threshold']
    
    d =dict(chan_grp=chan_grp,
        nb_chan=len(channels),
        min_mad=min(noise_uV),
        max_mad=max(noise_uV),
        reduce_noise_vector=reduce_noise_vector,
        noise_mean=np.mean(noise_uV),
        relative_threshold=relative_threshold,
        thresh_uV=relative_threshold * np.mean(noise_uV),
        
    )
    
    text = _summary_noise_template.format(**d)
    
    print(text)




_summary_after_peeler_clusters_template = """
Cluster {label}
Max on channel (abs): {max_on_abs_channel}
Max on channel (local to group): {max_on_channel}
Peak amplitude MAD: {max_peak_amplitude:.1f}
Peak amplitude (µV): {max_peak_amplitude_uV:.1f}
Nb spikes : {nb_spike}

"""

def summary_after_peeler_clusters(dataio, catalogue=None, chan_grp=None, labels=None, label=None):
    if chan_grp is None:
        chan_grp = min(dataio.channel_groups.keys())
    
    #~ cc =CatalogueConstructor(dataio, chan_grp=chan_grp)
    if catalogue is None:
        catalogue = dataio.load_catalogue(chan_grp=chan_grp)
    
    clusters = catalogue['clusters']
    
    if labels is None and label is None:
        labels = clusters['cluster_label']
    if label is not None:
        labels = [label]
    
    channels = dataio.channel_groups[chan_grp]['channels']
    
    for l, label in enumerate(labels):
        ind = np.nonzero(clusters['cluster_label']==label)[0][0]
        
        cluster = clusters[ind]
        
        max_on_channel = cluster['max_on_channel']
        if max_on_channel>=0:
            max_on_abs_channel = channels[max_on_channel]
        else:
            max_on_channel = None
            max_on_abs_channel = None

        max_peak_amplitude = cluster['max_peak_amplitude']
        max_peak_amplitude_uV = 'No conversion available'
        if dataio.datasource.bit_to_microVolt is not None and max_on_channel is not None:
            max_peak_amplitude_uV = max_peak_amplitude * catalogue['signals_mads'][max_on_channel] * dataio.datasource.bit_to_microVolt

        
        nb_spike = 0
        for seg_num in range(dataio.nb_segment):
            all_spikes = dataio.get_spikes(seg_num=seg_num, chan_grp=chan_grp)
            nb_spike += np.sum(all_spikes['cluster_label'] == label)
            
        
        fig, axs = plt.subplots(ncols=2, nrows=2)
        axs[1, 1].remove()
        # centroids
        ax = axs[0, 0]
        plot_centroids(catalogue, dataio=dataio, labels=[label,], ax=ax)
        ax.set_title('cluster {}'.format(label))
        
        
        # waveform density
        ax = axs[0, 1]
        if dataio.datasource.bit_to_microVolt is None:
            units = 'MAD'
            title = 'Amplitude in MAD (STD) ratio'
        else:
            units = 'uV'
            title = 'Amplitude μV'
        # TODO plot waveforms in a neighborhood
        plot_waveforms_histogram(catalogue, dataio=dataio, label=label, ax=ax, channels=[max_on_channel], units=units)
        ax.set_ylabel(title)
        
        # ISI
        ax = axs[1, 0]
        plot_isi(dataio, catalogue=catalogue, chan_grp=chan_grp, label=label, ax=ax, bin_min=0, bin_max=100, bin_size=1.)
        

        d =dict(chan_grp=chan_grp,
                    label=label,
                    max_on_channel=max_on_channel,
                    max_on_abs_channel=max_on_abs_channel,
                    max_peak_amplitude=max_peak_amplitude,
                    max_peak_amplitude_uV=max_peak_amplitude_uV,
                    nb_spike=nb_spike,
            
        )
        
        text = _summary_after_peeler_clusters_template.format(**d)
        
        print(text)
        
        ax.figure.text(.55, .25, text,  va='center') #, ha='center')
        
        
        
        
        
        
