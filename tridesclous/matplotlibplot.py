import numpy as np


from .tools import median_mad
from .dataio import DataIO
from .catalogueconstructor import CatalogueConstructor
from .tools import make_color_dict, get_neighborhood


try:
    import matplotlib.pyplot as plt
    # print('Problem import matplotlib.pyplot as plt')
except:
    pass
    
    
def plot_probe_geometry(dataio, chan_grp=0,  margin=150, channel_number_mode='absolut'):
    channel_group = dataio.channel_groups[chan_grp]
    channels = channel_group['channels']
    geometry = dataio.get_geometry(chan_grp=chan_grp)
    
    fig, ax = plt.subplots()
    for c, chan in enumerate(channels):
        x, y = geometry[c]
        ax.plot([x], [y], marker='o', color='r')
        if channel_number_mode == 'absolut':
            t = '{}'.format(chan)
        elif channel_number_mode == 'local':
            t = '{}'.format(c)
        elif channel_number_mode == 'both':
            t = '{}: {}'.format(c, chan)
        
        ax.text(x, y, t,  size=14, ha='center')

    ax.set_xlim(np.min(geometry[:, 0])-margin, np.max(geometry[:, 0])+margin)
    ax.set_ylim(np.min(geometry[:, 1])-margin, np.max(geometry[:, 1])+margin)
    
    return fig




def plot_signals(dataio_or_cataloguecconstructor, chan_grp=0, seg_num=0, time_slice=(0., 5.), 
            signal_type='initial', with_span=False, with_peaks=False):
    
    if isinstance(dataio_or_cataloguecconstructor, CatalogueConstructor):
        cataloguecconstructor = dataio_or_cataloguecconstructor
        dataio = cataloguecconstructor.dataio
        chan_grp = cataloguecconstructor.chan_grp
    elif isinstance(dataio_or_cataloguecconstructor, DataIO):
        cataloguecconstructor = None
        dataio = dataio_or_cataloguecconstructor
        
    channel_group = dataio.channel_groups[chan_grp]
    channels = channel_group['channels']
    
    i_start = int(time_slice[0]*dataio.sample_rate)
    i_stop = int(time_slice[1]*dataio.sample_rate)
    
    raw_sigs = dataio.get_signals_chunk(seg_num=seg_num, chan_grp=chan_grp,
                i_start=i_start, i_stop=i_stop, signal_type=signal_type)
    
    if signal_type=='initial':
        med, mad = median_mad(raw_sigs)
        sigs = (raw_sigs-med)/mad
        ratioY = 0.3
    elif signal_type=='processed':
        sigs = raw_sigs.copy()
        ratioY = 0.05

    #spread signals
    sigs *= ratioY
    sigs += np.arange(0, len(channels))[np.newaxis, :]
    
    
    
    times = np.arange(sigs.shape[0])/dataio.sample_rate
    fig, ax = plt.subplots()
    ax.plot(times, sigs)
    
    if with_peaks or with_span:
        assert cataloguecconstructor is not None
        peaks = cataloguecconstructor.all_peaks
        
        keep = (peaks['segment']==seg_num) & (peaks['index']>=i_start) & (peaks['index']<i_stop)
        peak_indexes = peaks[keep]['index'].copy()
        peak_indexes -= i_start
        
        if with_peaks:
            for i in range(len(channels)):
                ax.plot(times[peak_indexes], sigs[peak_indexes, i], ls='None', marker='o', color='k')
        
        if with_span:
            d = cataloguecconstructor.info['peak_detector']
            s = d['peak_span_ms'] / 1000.
            for ind in peak_indexes:
                ax.axvspan(times[ind]-s, times[ind]+s, color='b', alpha = .3)
    
    ax.set_yticks([])
    
    return fig


def _prepare_waveform_fig(ax=None, waveforms=None,
                                        flip_bottom_up=False, 
                                        ratioY='auto', deltaX='auto', margin='auto',
                                        **kargs):
    out_kargs = {}
    out_kargs.update(kargs)
    
    out_kargs['waveforms'] = waveforms
    out_kargs['flip_bottom_up'] = flip_bottom_up
    
    
    if ax is None:
        fig, ax = plt.subplots()
    out_kargs['ax'] = ax
    
    if flip_bottom_up:
        geometry = kargs['geometry'].copy()
        geometry[:, 1] *= -1.
        out_kargs['geometry'] = geometry
    #~ print(out_kargs.keys())
    
    geometry = out_kargs['geometry']
    # compute the smallest distance on x and y axis in between electrode
    if np.unique(geometry[:, 0]).size>1:
        xdist = np.min(np.diff(np.sort(np.unique(geometry[:, 0]))))
    else:
        xdist = None
    
    if np.unique(geometry[:, 1]).size>1:
        ydist = np.min(np.diff(np.sort(np.unique(geometry[:, 1]))))
    else:
        ydist = None
    
    if xdist is None:
        if ydist is not None:
            xdist = ydist
        else:
            xdist = 10.
    if ydist is None:
        if xdist is not None:
            ydist = xdist
        else:
            ydist = 10.

    #deltaX is tha width along x axis
    if deltaX =='auto':
        deltaX = xdist/2.5
    
    if ratioY == 'auto':
        m = np.max(np.abs(waveforms))
        ratioY = ydist / m * 0.7
    if margin =='auto':
        margin = 1.5*min(xdist, ydist)
    
    out_kargs['deltaX'] = deltaX
    out_kargs['ratioY'] = ratioY
    out_kargs['margin'] = margin
    
    
    return out_kargs
    


def _plot_wfs(ax=None, waveforms=None, channels=None, geometry=None, 
                        ratioY=None, deltaX=None, linewidth=2., alpha=1.0, color='k', **unused):
        
    wf = waveforms.copy()
    if wf.ndim ==2:
        wf = wf[None, : ,:]
    width = wf.shape[1]

    vect =np.zeros(wf.shape[1]*wf.shape[2])
    wf *= ratioY
    for i, chan in enumerate(channels):
        x, y = geometry[i]
        vect[i*width:(i+1)*width] = np.linspace(x-deltaX, x+deltaX, num=width)
        wf[:, :, i] += y
    
    wf[:, 0,:] = np.nan
    wf = wf.swapaxes(1,2).reshape(wf.shape[0], -1).T
    
    ax.plot(vect, wf, color=color, lw=linewidth, alpha=alpha)


def _enhance_waveform_fig(ax=None, channels=None, geometry=None,
            show_channels=True, channel_number_mode='absolut',
            show_amplitude=False, ratio_mad=5,
            show_ticks=False, flip_bottom_up=True,
            aspect_ratio='equal', xlims=None, ylims=None,
            margin=None, ratioY=None,
            text_size=10, text_color='k',
            **unused):

    if show_channels:
        for c, chan in enumerate(channels):
            x, y = geometry[c, :]
            if channel_number_mode == 'absolut':
                t = '{}'.format(chan)
            elif channel_number_mode == 'local':
                t = '{}'.format(c)
            elif channel_number_mode == 'both':
                t = '{}: {}'.format(c, chan)
            ax.text(x, y, t, color=text_color, size=text_size,ha='center')

    if xlims is None:
        xlim0, xlim1 = np.min(geometry[:, 0])-margin, np.max(geometry[:, 0])+margin
    else:
        xlim0, xlim1 = xlims
    if ylims is None:
        ylim0, ylim1 = np.min(geometry[:, 1])-margin, np.max(geometry[:, 1])+margin
    else:
        ylim0, ylim1 = ylims
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(ylim0, ylim1)
    
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    
    if show_amplitude:
        #~ x = xlim0 + margin/10
        x = xlim0 + (xlim1-xlim0)/50
        y = (ylim1+ylim0)/2
        ax.plot([x, x], [y, y+ratioY*ratio_mad], color='k', linewidth=2)
        ax.text(x,y, '{}*MAD'.format(ratio_mad), ha='left')
        

    ax.set_aspect(aspect_ratio)    


def plot_waveforms_with_geometry(waveforms, channels, geometry, **kargs):
    kargs['waveforms'] = waveforms
    kargs['channels'] = channels
    kargs['geometry'] = geometry
    kargs = _prepare_waveform_fig(**kargs)
    _plot_wfs(**kargs)
    _enhance_waveform_fig(**kargs)

    
def plot_waveforms(cataloguecconstructor, labels=None, nb_max=50, alpha=0.4,
                neighborhood_radius=None, **kargs):
    cc = cataloguecconstructor
    channels = np.array(cc.dataio.channel_groups[cc.chan_grp]['channels'])
    geometry = cc.dataio.get_geometry(chan_grp=cc.chan_grp)
    
    kargs['alpha'] = alpha
    
    all_wfs = cc.get_some_waveforms()
    kargs['waveforms'] = all_wfs
    kargs['channels'] = channels
    kargs['geometry'] = geometry
    kargs = _prepare_waveform_fig(**kargs)
    
    if labels is None:
        wfs = all_wfs[:nb_max,:, :]
        kargs['waveforms'] = wfs
        _plot_wfs(**kargs)
        _enhance_waveform_fig(**kargs)
    else:
        if not hasattr(cc, 'colors'):
            cc.refresh_colors()
        
        if isinstance(labels, int):
            labels = [labels]
        
        for label in labels:
            peaks = cc.all_peaks[cc.some_peaks_index]
            keep = peaks['cluster_label'] == label
            wfs = all_wfs[keep][:nb_max]
            
            kargs['color'] = cc.colors.get(label, 'k')
            kargs['waveforms'] = wfs
            _plot_wfs(**kargs)
        
        _enhance_waveform_fig(**kargs)


def plot_centroids(arg0, labels=[], alpha=1, neighborhood_radius=None, **kargs):
    """
    arg0 can be cataloguecconstructor or catalogue (a dict)
    """
    
    if isinstance(labels, int):
        labels = [labels]
    
    if isinstance(arg0, CatalogueConstructor):
        cc = arg0
        dataio = cc.dataio       
        if not hasattr(cc, 'colors'):
            cc.refresh_colors()
        colors = cc.colors
        chan_grp = cc.chan_grp
        
        centroids_wfs = cc.centroids_median
        
        label_inds = []
        for label in labels:
            ind = cc.index_of_label(label)
            label_inds.append(ind)
            
        ratio_mad = cc.info['peak_detector']['relative_threshold']
        
    elif isinstance(arg0, dict) and 'clusters' in arg0:
        catalogue = arg0
        dataio = kargs['dataio']
        chan_grp = catalogue['chan_grp']
        clusters = catalogue['clusters']
        colors = make_color_dict(clusters)
        
        centroids_wfs = catalogue['centers0']
        
        label_inds = []
        for label in labels:
            ind = np.nonzero(clusters['cluster_label']==label)[0][0]
            label_inds.append(ind)
        
        ratio_mad = catalogue['peak_detector_params']['relative_threshold']
        
    else:
        raise(Exception('arg0 must a catalogue constructor or a catalogue dict'))
    
    channels = dataio.channel_groups[chan_grp]['channels']
    geometry =dataio.get_geometry(chan_grp=chan_grp)

    if neighborhood_radius is not None:
        assert len(labels) == 1
        neighborhood = get_neighborhood(geometry, neighborhood_radius)
        extremum_channel = clusters[label_inds[0]]['extremum_channel']
        keep = neighborhood[extremum_channel, :]
        centroids_wfs = centroids_wfs[:,  :, keep]
        channels = np.array(channels)[keep]
        geometry = geometry[keep, :]
    
    kargs['ratio_mad'] = ratio_mad
    kargs['waveforms'] = centroids_wfs[label_inds, :, :]
    kargs['channels'] = channels
    kargs['geometry'] = geometry
    kargs = _prepare_waveform_fig(**kargs)

    for ind, label in zip(label_inds, labels):
        wf = centroids_wfs[ind, :, :]
        kargs['waveforms'] = wf
        kargs['color'] = colors.get(label, 'k')
        _plot_wfs(**kargs)
    
    _enhance_waveform_fig(**kargs)





def plot_waveforms_density(waveforms, bin_min, bin_max, bin_size, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    wf = waveforms
    data = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
    
    bins = np.arange(bin_min, bin_max, bin_size)
    
    hist2d = np.zeros((data.shape[1], bins.size))
    indexes0 = np.arange(data.shape[1])
    
    data_bined = np.floor((data-bin_min)/bin_size).astype('int32')
    data_bined = data_bined.clip(0, bins.size-1)
    
    for d in data_bined:
        hist2d[indexes0, d] += 1
    
    im = ax.imshow(hist2d.T, interpolation='nearest', 
                    origin='lower', aspect='auto', extent=(0, data.shape[1], bin_min, bin_max), cmap='hot')
    
    return im
    
def plot_waveforms_histogram(arg0, label=None, ax=None, channels=None,
            bin_min=None, bin_max=None, bin_size=0.1, units='MAD',
            dataio=None,# usefull when arg0 is catalogue
            ):
    """
    arg0 can be cataloguecconstructor or catalogue (a dict)
    """

    if ax is None:
        fig, ax = plt.subplots()
    
    if isinstance(arg0, CatalogueConstructor):
        cc = arg0
        dataio = cc.dataio
        chan_grp = cc.chan_grp
        
        # take waveforms
        #~ ind = cc.index_of_label(label)
        spike_labels = cc.all_peaks['cluster_label'][cc.some_peaks_index]
        #~ wf = cc.some_waveforms[spike_labels==label]
        ind, = np.nonzero(cc.all_peaks['cluster_label'] == label)
        wf = cc.get_some_waveforms(peaks_index=ind)
        wf = wf[:, :, channels]
        if units in ('uV', 'μV'):
            wf = wf * cc.signals_mads[channels][None, None, :] * dataio.datasource.bit_to_microVolt

        n_left = cc.info['extract_waveforms']['n_left']
        n_right = cc.info['extract_waveforms']['n_right']
    
    elif isinstance(arg0, dict) and 'clusters' in arg0:
        catalogue = arg0
        chan_grp = catalogue['chan_grp']
        clusters = catalogue['clusters']

        n_left = catalogue['n_left']
        n_right = catalogue['n_right']
        
        all_wf = []
        for seg_num in range(dataio.nb_segment):
            # TODO loop over segments
            spikes = dataio.get_spikes(seg_num=seg_num, chan_grp=chan_grp,)
        
            # take waveforms
            #~ spike_labels = spikes['cluster_label']
            spikes = spikes[spikes['cluster_label'] == label]
            sample_indexes = spikes['index']
            
            if sample_indexes.size>1000:
                # limit to 1000 spike by segment
                sample_indexes = np.random.choice(sample_indexes, size=1000)
            sample_indexes = sample_indexes[(sample_indexes>-n_left)]
            
            wf_ = dataio.get_some_waveforms(seg_num=seg_num, chan_grp=chan_grp,
                        peak_sample_indexes=sample_indexes, n_left=n_left, n_right=n_right)
            wf_ = wf_[:, :, channels]
            all_wf.append(wf_)
        wf = np.concatenate(all_wf, axis=0)
        
        if units in ('uV', 'μV'):
            wf = wf * catalogue['signals_mads'][channels][None, None, :] * dataio.datasource.bit_to_microVolt
        
        
        
    else:
        raise(Exception('arg0 must a catalogue constructor or a catalogue dict'))


    if bin_min is None:
        bin_min = np.min(wf) - 1.
    if bin_max is None:
        bin_max = np.max(wf) +1
    if bin_size is None:
        if units=='MAD':
            bin_size = 0.1
        elif units in ('uV', 'μV'):
            bin_size = (bin_max - bin_min) / 500.
    
    #~ data = wf.swapaxes(1,2).reshape(wf.shape[0], -1)
    
    #~ bins = np.arange(bin_min, bin_max, bin_size)
    
    #~ hist2d = np.zeros((data.shape[1], bins.size))
    #~ indexes0 = np.arange(data.shape[1])
    
    #~ data_bined = np.floor((data-bin_min)/bin_size).astype('int32')
    #~ data_bined = data_bined.clip(0, bins.size-1)
    
    #~ for d in data_bined:
        #~ hist2d[indexes0, d] += 1
    
    #~ im = ax.imshow(hist2d.T, interpolation='nearest', 
                    #~ origin='lower', aspect='auto', extent=(0, data.shape[1], bin_min, bin_max), cmap='hot')
                    
    im = plot_waveforms_density(wf, bin_min, bin_max, bin_size, ax=ax)

    peak_width = n_right - n_left
    for c, chan in enumerate(channels):
        abs_chan = dataio.channel_groups[chan_grp]['channels'][chan]
        ax.text(c*peak_width-n_left, 0, '{}'.format(abs_chan),  size=10, ha='center', color='w')
        if c>0:
            ax.axvline((c) * peak_width, color='w')
    
    ax.set_xticks([])

def plot_features_scatter_2d(cataloguecconstructor, labels=None, nb_max=500):
    cc = cataloguecconstructor
    
    all_feat = cc.some_features
    n = all_feat.shape[1]
    
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    
    l = []
    if labels is None:
        l.append( (cc.some_features[:nb_max], 'k') )
    else:
        if not hasattr(cc, 'colors'):
            cc.refresh_colors()
        
        if isinstance(labels, int):
            labels = [labels]
        for label in labels:
            peaks = cc.all_peaks[cc.some_peaks_index]
            keep = peaks['cluster_label'] == label
            feat = cc.some_features[keep][:nb_max]
            color = cc.colors.get(label, 'k')
            l.append((feat, color))
    
    for c in range(n):
        for r in range(n):
            ax = axs[r, c]
            
            if c==r:
                for feat, color in l:
                    y, x = np.histogram(feat[:, r], bins=100)
                    ax.plot(x[:-1], y, color=color)
            elif c<r:
                for feat, color in l:
                    ax.plot(feat[:, c], feat[:, r], color=color, markersize=2, ls='None', marker='o')
            else:
                fig.delaxes(ax)



def plot_isi(dataio, catalogue=None, chan_grp=None, label=None, ax=None, bin_min=0, bin_max=100, bin_size=1.):
    """
    bin are in ms
    """
    if ax is None:
        fig, ax = plt.subplots()

    if catalogue is None:
        catalogue = dataio.load_catalogue(chan_grp=chan_grp)
    
    sr = dataio.sample_rate
    
    bins = np.arange(bin_min, bin_max, bin_size)
    
    count = None
    for seg_num in range(dataio.nb_segment):
        spikes = dataio.get_spikes(seg_num=seg_num, chan_grp=chan_grp,)
        spikes = spikes[spikes['cluster_label'] == label]
        sample_indexes = spikes['index']
        
        isi = np.diff(sample_indexes)/ (sr/1000.)
        
        count_, bins = np.histogram(isi, bins=bins)
        if count is None:
            count = count_
        else:
            count += count_
    
    ax.plot(bins[:-1], count, color='k') # TODO color
    


    
