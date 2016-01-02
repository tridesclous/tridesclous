from matplotlib import pyplot
import seaborn as sns
import numpy as np
import pandas as pd

from .tools import median_mad

sns.set(style="white")

"""
Function for plotting signals, signals+peak, waveform, ... with matplotlib.


PeakDetector, WaveformExtractor, Clustering, Peeler have there method implemented here.
Splitting compute code and plot code should simplify reading.


"""

def add_vspan(ax, n_left, n_right, nb_channel):
    width = n_right - n_left
    for i in range(nb_channel):
        if i%2==1:
            ax.axvspan(width*i, width*(i+1)-1, alpha = .1, color = 'k')
        ax.axvline(-n_left + width*i, alpha = .05, color = 'k')
        

class PeakDetectorPlot:
    pass


class WaveformExtractorPlot:
    
    def plot_good_limit(self):
        fig, axs = pyplot.subplots(nrows = 2)
        
        long_wf = self.long_waveforms
        short_wf = self.get_ajusted_waveforms()
        
        medL, madL = median_mad(long_wf)
        medS, madS = median_mad(short_wf)
        
        all_med = pd.concat([medL, medS], axis=1)
        all_med.columns = ['long', 'short']
        all_med.plot(ax=axs[0])

        all_mad = pd.concat([madL, madS], axis=1)
        all_mad.columns = ['long', 'short']
        all_mad.plot(ax=axs[1])
        
        axs[0].set_ylabel('median')
        axs[1].set_ylabel('mad')
        axs[1].axhline(1.1)
        
        
        width = self.n_right - self.n_left
        for ax in axs:
            for i in range(self.nb_channel):
                l1 = self.limit_left - self.n_left
                l2 = width - (self.n_right-self.limit_right)
                ax.axvline(l1+width*i, color = 'k', lw = 1, ls = '--')
                ax.axvline(l2+width*i, color = 'k', lw = 1, ls = '--')
            
            add_vspan(ax, self.n_left, self.n_right, self.nb_channel)
    


class ClusteringPlot:
    def plot_explained_variance_ratio(self):
        fix, ax = pyplot.subplots()
        ax.plot(np.cumsum(self._pca.explained_variance_ratio_))
        ax.set_ylim(0,1)
    
    def plot_waveform_variance(self, factor = 5.):
        med, mad = median_mad(self.waveforms)
        
        n = self._pca.n_components
        fix, axs = pyplot.subplots(nrows = n)
        for i in range(n):
            ax = axs[i]
            comp= self._pca.components_[i,:]
            ax.plot(med, color = 'm', lw = 2)
            ax.fill_between(np.arange(med.size),med+factor*comp,med-factor*comp, alpha = .2, color = 'm')
            ax.set_ylabel('pca{} {:.1f}%'.format(i, self._pca.explained_variance_ratio_[i]*100.))


            nb_channel = self.waveforms.columns.levels[0].size
            samples = self.waveforms.columns.levels[1]
            n_left, n_right = min(samples), max(samples)+1
            
            add_vspan(ax, n_left, n_right, nb_channel)
    
    
    def plot_projection(self, colors = None, palette = 'husl', plot_density = False):
        if not hasattr(self, 'cluster_labels'):
            
            g = sns.PairGrid(self.features, diag_sharey=False)
            g.map_upper(pyplot.scatter)
            g.map_diag(sns.kdeplot, lw=3)
            if plot_density:
                g.map_lower(sns.kdeplot, cmap="Blues_d")
                
        else:
        
            if colors is None:
                colors = sns.color_palette(palette, len(self.cluster_labels))
            
            df = pd.concat([self.features, self.labels], axis=1)
            hue_kws = {'cmap':[sns.light_palette(color, as_cmap=True) for color in colors]}
            g = sns.PairGrid(df, diag_sharey=False,hue='label', vars = self.features.columns, hue_kws=hue_kws)
            g.map_diag(sns.kdeplot, lw=3)
            g.map_upper(pyplot.scatter)
            if plot_density:
                g.map_lower(sns.kdeplot)
        
    
    def plot_catalogue(self, colors = None, palette = 'husl', sameax = True):
        if colors is None:
            colors = sns.color_palette(palette, len(self.catalogue))
        
        if sameax:
            fix, ax = pyplot.subplots()
        else:
            fix, axs = pyplot.subplots(nrows = len(self.catalogue), sharey = True, sharex = True)
            
        for i,k in enumerate(self.catalogue):
            wf0 = self.catalogue[k]['center']
            mad = self.catalogue[k]['mad']
            if not sameax:
                ax = axs[i]
            ax.plot(wf0, color = colors[i], label = '#{}'.format(k))
            ax.fill_between(np.arange(wf0.size), wf0-mad, wf0+mad, color = colors[i], alpha = .4)

        nb_channel = self.waveforms.columns.levels[0].size
        samples = self.waveforms.columns.levels[1]
        n_left, n_right = min(samples)+2, max(samples)-1
        #~ print(wf0.shape)
        #~ print(n_left, n_right, n_right - n_left, wf0.shape[0]/nb_channel)
        if sameax:
            axs = [ax]
        
        for ax in axs:
            add_vspan(ax, n_left, n_right, nb_channel)
            ax.legend()
        
    
    def plot_derivatives(self,  colors = None, palette = 'husl'):
        if colors is None:
            colors = sns.color_palette(palette, len(self.catalogue))

        fix, axs = pyplot.subplots(nrows = 3)
        for i,k in enumerate(self.catalogue):
            axs[0].plot(self.catalogue[k]['center'], color = colors[i], label = '#{}'.format(k))
            axs[1].plot(self.catalogue[k]['centerD'], color = colors[i])
            axs[2].plot(self.catalogue[k]['centerDD'], color = colors[i])    
        
        axs[0].set_ylabel("waveform")
        axs[1].set_ylabel("waveform '")
        axs[2].set_ylabel("waveform ''")
        axs[0].legend()
        
        nb_channel = self.waveforms.columns.levels[0].size
        samples = self.waveforms.columns.levels[1]
        n_left, n_right = min(samples)+2, max(samples)-1
        for ax in axs:
            add_vspan(ax, n_left, n_right, nb_channel)


class PeelerPlot:
    pass


