"""
Find a good example of collision in striatum rat dataset.

"""
import os,shutil


from tridesclous import DataIO, CatalogueConstructor, Peeler
from tridesclous import download_dataset
from tridesclous.cataloguetools import apply_all_catalogue_steps
from tridesclous.peeler import make_prediction_signals
from tridesclous.tools import int32_to_rgba
from tridesclous.matplotlibplot import plot_waveforms_with_geometry, plot_features_scatter_2d


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


dirname = 'tdc_olfactory_bulb'
channels = [5,6,7,8]


def make_catalogue():
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
            
    dataio = DataIO(dirname=dirname)
    localdir, filenames, params = download_dataset(name='olfactory_bulb')
    dataio.set_data_source(type='RawData', filenames=filenames, **params)
    dataio.add_one_channel_group(channels = channels)
        
        
    cc = CatalogueConstructor(dataio=dataio)


    params = {
        'duration' : 300.,
        'preprocessor' : {
            'highpass_freq' : 300.,
            'chunksize' : 1024,
            'pad_width' : 100,
        },
        'peak_detector' : {
            'peak_sign' : '-',
            'relative_threshold' : 7.,
            'peak_span' : 0.0005,
            #~ 'peak_span' : 0.000,
        },
        'extract_waveforms' : {
            'n_left' : -25,
            'n_right' : 40,
            'nb_max' : 10000,
        },
        'clean_waveforms' : {
            'alien_value_threshold' : 60.,
        },
        'noise_snippet' : {
            'nb_snippet' : 300,
        }, 

    'feature_method': 'global_pca',
    'feature_kargs': {'n_components': 20},
    
    'cluster_method': 'kmeans',
    'cluster_kargs': {'n_clusters': 5},
    
    'clean_cluster' : False,
    'clean_cluster_kargs' : {},
        

    }

    apply_all_catalogue_steps(cc, params, verbose=True)


    cc.order_clusters(by='waveforms_rms')
    cc.move_cluster_to_trash(4)
    cc.make_catalogue_for_peeler()

def apply_peeler():
    dataio = DataIO(dirname=dirname)
    catalogue = dataio.load_catalogue(chan_grp=0)
    peeler = Peeler(dataio)
    peeler.change_params(catalogue=catalogue,chunksize=1024)

    peeler.run(progressbar=True)


def make_animation():
    """
    Good example between 1.272 1.302
    because collision
    """
    
    
    dataio = DataIO(dirname=dirname)
    catalogue = dataio.load_catalogue(chan_grp=0)
    
    clusters = catalogue['clusters']
    
    sr = dataio.sample_rate
    
    # also a good one a  11.356 - 11.366
    
    t1, t2 = 1.272, 1.295
    i1, i2 = int(t1*sr), int(t2*sr)
    
    
    spikes = dataio.get_spikes()
    spike_times = spikes['index'] / sr
    keep = (spike_times>=t1) & (spike_times<=t2)
    
    spikes = spikes[keep]
    print(spikes)
    
    
    sigs = dataio.get_signals_chunk(i_start=i1, i_stop=i2,
                signal_type='processed')
    sigs = sigs.copy()
    times = np.arange(sigs.shape[0])/dataio.sample_rate
    
    def plot_spread_sigs(sigs, ax, ratioY = 0.02, **kargs):
        #spread signals
        sigs2 = sigs * ratioY
        sigs2 += np.arange(0, len(channels))[np.newaxis, :]
        ax.plot(times, sigs2, **kargs)
        
        ax.set_ylim(-0.5, len(channels)-.5)
        ax.set_xticks([])
        ax.set_yticks([])
    
    residuals = sigs.copy()
    
    local_spikes = spikes.copy()
    local_spikes['index'] -= i1
    
    #~ fig, ax = plt.subplots()
    #~ plot_spread_sigs(sigs, ax, color='k')
    
    num_fig = 0
    
    fig_pred, ax_predictions = plt.subplots()
    ax_predictions.set_title('All detected templates from catalogue')
    

    fig, ax = plt.subplots()
    plot_spread_sigs(residuals, ax, color='k', lw=2)
    ax.set_title('Initial filtered signals with spikes')
    
    fig.savefig('../img/peeler_animation_sigs.png')
    
    fig.savefig('png/fig{}.png'.format(num_fig))
    num_fig += 1
    
    
    for i in range(local_spikes.size):
        label = local_spikes['cluster_label'][i]
        
        color = clusters[clusters['cluster_label']==label]['color'][0]
        color = int32_to_rgba(color, mode='float')
        
        pred = make_prediction_signals(local_spikes[i:i+1], 'float32', (i2-i1, len(channels)), catalogue)
        
        fig, ax = plt.subplots()
        plot_spread_sigs(residuals, ax, color='k', lw=2)
        plot_spread_sigs(pred, ax, color=color, lw=1.5)
        ax.set_title('Dected spike label {}'.format(label))

        fig.savefig('png/fig{}.png'.format(num_fig))
        num_fig += 1
        
        residuals -= pred
    
        plot_spread_sigs(pred, ax_predictions, color=color, lw=1.5)
        
        fig, ax = plt.subplots()
        plot_spread_sigs(residuals, ax, color='k', lw=2)
        plot_spread_sigs(pred, ax, color=color, lw=1, ls='--')
        ax.set_title('New residual after substraction')

        fig.savefig('png/fig{}.png'.format(num_fig))
        num_fig += 1
        

    fig_pred.savefig('png/fig{}.png'.format(num_fig))
    num_fig += 1
        

    
    #~ plt.show()


def make_catalogue_figure():


    dataio = DataIO(dirname=dirname)
    catalogue = dataio.load_catalogue(chan_grp=0)
    
    clusters = catalogue['clusters']
    
    geometry = dataio.get_geometry(chan_grp=0)
    
    fig, ax = plt.subplots()
    ax.set_title('Catalogue have 4 templates')
    for i in range(clusters.size):
        color = clusters[i]['color']
        color = int32_to_rgba(color, mode='float')
        
        waveforms = catalogue['centers0' ][i:i+1]
    
        plot_waveforms_with_geometry(waveforms, channels, geometry,
                ax=ax, ratioY=3, deltaX= 50, margin=50, color=color,
                linewidth=3, alpha=1, show_amplitude=True, ratio_mad=8)
    
    fig.savefig('../img/peeler_templates_for_animation.png')
    
    #~ plt.show()

def make_pca_collision_figure():

    dataio = DataIO(dirname=dirname)
    cc = CatalogueConstructor(dataio=dataio)
    
    clusters = cc.clusters
    #~ plot_features_scatter_2d(cc, labels=None, nb_max=500)
    
    #~ plot_features_scatter_2d
    
    fig, ax = plt.subplots()
    ax.set_title('Collision problem')
    ax.set_aspect('equal')
    features = cc.some_features
    
    labels = cc.all_peaks[cc.some_peaks_index]['cluster_label']
    
    for k in [0,1,2,3]:
        color = clusters[clusters['cluster_label']==k]['color'][0]
        color = int32_to_rgba(color, mode='float')
        
        keep = labels==k
        feat = features[keep]
    
        print(np.unique(labels))
        
        
        ax.plot(feat[:,0], feat[:,1], ls='None', marker='o', color=color, markersize=3, alpha=.5)
    
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    
    ax.set_xlabel('pca0')
    ax.set_ylabel('pca1')
    
    ax.annotate('Collision', xy=(17.6, -16.4), xytext=(30, -30),
            arrowprops=dict(facecolor='black', shrink=0.05))
    
    #~ 

    
    fig.savefig('../img/collision_proble_pca.png')
    
    #~ plt.show()

if __name__ == '__main__':
    #~ make_catalogue()
    #~ apply_peeler()
    
    make_animation()
    
    make_catalogue_figure()
    
    make_pca_collision_figure()
    
    #convert -delay 250 -loop 0 png/*.png  ../img/peeler_animation.gif






