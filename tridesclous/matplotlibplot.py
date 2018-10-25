import numpy as np
import matplotlib.pyplot as plt

from .tools import median_mad
from .dataio import DataIO
from .catalogueconstructor import CatalogueConstructor


    
    
def plot_probe_geometry(dataio, chan_grp=0,  margin=150,):
    channel_group = dataio.channel_groups[chan_grp]
    channels = channel_group['channels']
    geometry = dataio.get_geometry(chan_grp=chan_grp)
    
    fig, ax = plt.subplots()
    for c, chan in enumerate(channels):
        x, y = geometry[c]
        ax.plot([x], [y], marker='o', color='r')
        ax.text(x, y, '{}: {}'.format(c, chan),  size=20)

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
            d = cataloguecconstructor.info['peak_detector_params']
            s = d['peak_span']
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

    
def plot_waveforms(cataloguecconstructor, labels=None, nb_max=50, alpha=0.4, **kargs):
    cc = cataloguecconstructor
    channels = cc.dataio.channel_groups[cc.chan_grp]['channels']
    geometry = cc.dataio.get_geometry(chan_grp=cc.chan_grp)
    
    kargs['alpha'] = alpha
    
    all_wfs = cc.some_waveforms
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


def plot_centroids(cataloguecconstructor, labels=[], alpha=1, **kargs):
    cc = cataloguecconstructor
    channels = cc.dataio.channel_groups[cc.chan_grp]['channels']
    geometry = cc.dataio.get_geometry(chan_grp=cc.chan_grp)

    if not hasattr(cc, 'colors'):
        cc.refresh_colors()

    if isinstance(labels, int):
        labels = [labels]
    
    inds = []
    for label in labels:
        ind = cc.index_of_label(label)
        inds.append(ind)
    
    kargs['ratio_mad'] = cc.info['peak_detector_params']['relative_threshold']
    kargs['waveforms'] = cc.centroids_median[inds, :, :]
    kargs['channels'] = channels
    kargs['geometry'] = geometry
    kargs = _prepare_waveform_fig(**kargs)

    for label in labels:
        ind = cc.index_of_label(label)
        wf = cc.centroids_median[ind, :, :]
        kargs['waveforms'] = wf
        kargs['color'] = cc.colors.get(label, 'k')
        _plot_wfs(**kargs)
    
    _enhance_waveform_fig(**kargs)


def plot_waveforms_histogram(cataloguecconstructor, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    ind = cc.index_of_label(label)
    
    



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
    
    
