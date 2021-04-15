
import os

import numpy as np

from .catalogueconstructor import CatalogueConstructor
from .matplotlibplot import plot_centroids, plot_waveforms_histogram, plot_isi
from .tools import get_neighborhood



_summary_catalogue_clusters_template = """
Cluster {label}
Peak channel (abs): {extremum_channel_abs}
Peak channel (local to group): {extremum_channel}
Peak amplitude MAD: {extremum_amplitude}
Peak amplitude (µV): {extremum_amplitude_uV}

"""
def summary_catalogue_clusters(dataio, chan_grp=None, labels=None, label=None,
        show_channels=None, neighborhood_radius=None):
    
    
    cc =CatalogueConstructor(dataio, chan_grp=chan_grp)
    
    if labels is None and label is None:
        labels = cc.positive_cluster_labels
    if label is not None:
        labels = [label]
    
    # channel index absolut to datasource
    channel_abs = dataio.channel_groups[chan_grp]['channels']

    if show_channels is None:
        if neighborhood_radius is None:
            if len(channel_abs)>10:
                show_channels = False
            else:
                show_channels = True
        else:
            show_channels = True
    
    for l, label in enumerate(labels):

        ind = cc.index_of_label(label)
        cluster = cc.clusters[ind]

        extremum_channel = cluster['extremum_channel']
        
        if extremum_channel>=0:
            extremum_channel_abs = channel_abs[extremum_channel]
        else:
            extremum_channel = None
            extremum_channel_abs = None
        
        extremum_amplitude = cluster['extremum_amplitude']
        extremum_amplitude_uV = np.nan
        if cc.dataio.datasource.bit_to_microVolt is not None and extremum_channel is not None:
            extremum_amplitude_uV = extremum_amplitude * cc.signals_mads[extremum_channel] * cc.dataio.datasource.bit_to_microVolt
            
            
        
        d = dict(label=label,
                    extremum_channel=extremum_channel,
                    extremum_channel_abs=extremum_channel_abs,
                    extremum_amplitude=extremum_amplitude,
                    extremum_amplitude_uV=extremum_amplitude_uV,
                    )
        text = _summary_catalogue_clusters_template.format(**d)
        
        print(text)
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(ncols=2)
        plot_centroids(cc, labels=[label,], ax=axs[0], show_channels=show_channels, neighborhood_radius=neighborhood_radius)
        axs[0].set_title('cluster {}'.format(label))
        
        if dataio.datasource.bit_to_microVolt is None:
            units = 'MAD'
            title = 'Amplitude in MAD (STD) ratio'
        else:
            units = 'uV'
            title = 'Amplitude μV'
        
        plot_waveforms_histogram(cc, label=label, ax=axs[1], channels=[extremum_channel], units=units)
        axs[1].set_title('Hist on channel {}'.format(extremum_channel_abs))
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
    
    
    relative_threshold = cc.info['peak_detector']['relative_threshold']
    
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
Max on channel (abs): {extremum_channel_abs}
Max on channel (local to group): {extremum_channel}
Peak amplitude MAD: {extremum_amplitude:.1f}
Peak amplitude (µV): {extremum_amplitude_uV:.1f}
Nb spikes : {nb_spike}

"""

def summary_after_peeler_clusters(dataio, catalogue=None, chan_grp=None, labels=None, label=None,
            show_channels=None, neighborhood_radius=None):
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
    
    channel_abs = dataio.channel_groups[chan_grp]['channels']
 
    if show_channels is None:
        if neighborhood_radius is None:
            if len(channel_abs)>10:
                show_channels = False
            else:
                show_channels = True
        else:
            show_channels = True
    
    figs = []
    for l, label in enumerate(labels):
        ind = np.nonzero(clusters['cluster_label']==label)[0][0]
        
        cluster = clusters[ind]
        
        extremum_channel = cluster['extremum_channel']
        if extremum_channel>=0:
            extremum_channel_abs = channel_abs[extremum_channel]
        else:
            extremum_channel = None
            extremum_channel_abs = None

        extremum_amplitude = cluster['extremum_amplitude']
        extremum_amplitude_uV = np.nan
        if dataio.datasource.bit_to_microVolt is not None and extremum_channel is not None:
            extremum_amplitude_uV = extremum_amplitude * catalogue['signals_mads'][extremum_channel] * dataio.datasource.bit_to_microVolt

        
        nb_spike = 0
        for seg_num in range(dataio.nb_segment):
            all_spikes = dataio.get_spikes(seg_num=seg_num, chan_grp=chan_grp)
            nb_spike += np.sum(all_spikes['cluster_label'] == label)
            

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(ncols=2, nrows=2)
        figs.append(fig)
        axs[0, 0].remove()
        # centroids
        ax = axs[0, 1]
        plot_centroids(catalogue, dataio=dataio, labels=[label,], ax=ax, show_channels=show_channels, neighborhood_radius=neighborhood_radius)
        ax.set_title('cluster {}'.format(label))
        
        
        # waveform density
        ax = axs[1, 1]
        if dataio.datasource.bit_to_microVolt is None:
            units = 'MAD'
            title = 'Amplitude in MAD (STD) ratio'
        else:
            units = 'uV'
            title = 'Amplitude μV'
        
            
        if neighborhood_radius is None:
            plot_channels_hist = [extremum_channel]
        else:
            neihb = get_neighborhood(dataio.get_geometry(chan_grp=chan_grp) , neighborhood_radius)
            plot_channels_hist,  = np.nonzero(neihb[extremum_channel])    # local index (to group)
            
        plot_waveforms_histogram(catalogue, dataio=dataio, label=label, ax=ax, channels=plot_channels_hist,
                        units=units)
        ax.set_title(title)
        
        # ISI
        ax = axs[1, 0]
        plot_isi(dataio, catalogue=catalogue, chan_grp=chan_grp, label=label, ax=ax, bin_min=0, bin_max=100, bin_size=1.)
        ax.set_title('ISI (ms)')

        d =dict(chan_grp=chan_grp,
                    label=label,
                    extremum_channel=extremum_channel,
                    extremum_channel_abs=extremum_channel_abs,
                    extremum_amplitude=extremum_amplitude,
                    extremum_amplitude_uV=extremum_amplitude_uV,
                    nb_spike=nb_spike,
            
        )
        
        text = _summary_after_peeler_clusters_template.format(**d)
        
        #~ print(text)
        
        ax.figure.text(.05, .75, text,  va='center') #, ha='center')
    
    return figs


def generate_report(dataio, export_path=None, neighborhood_radius=None) :
    if export_path is None:
        export_path = os.path.join(dataio.dirname, 'report')
    
    
    for chan_grp in dataio.channel_groups.keys():
        if not dataio.is_spike_computed(chan_grp=chan_grp):
            continue
        
        catalogue = dataio.load_catalogue(chan_grp=chan_grp)
        
        if catalogue is None:
            continue
        
        clusters = catalogue['clusters']
        
        # TODO with cell_label
        labels = clusters['cluster_label']
        
        path = os.path.join(export_path, 'chan_grp {}'.format(chan_grp), 'figures')
        if not os.path.exists(path):
            os.makedirs(path)
        
        for label in labels:
            figs = summary_after_peeler_clusters(dataio, chan_grp=chan_grp, label=label, neighborhood_radius=neighborhood_radius)
            figs[0].savefig(os.path.join(path, 'summary_cluster_{}.png'.format(label)))
    