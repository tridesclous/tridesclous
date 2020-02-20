.. _important_details:

Important details
=================

A list a miscellaneous details, some of them are related to data handling
and other to details of methods at different steps of the chain.


Multi segments
--------------

Many recording are split into several "Segments". Segment here refer to the
`neo meaning <http://neo.readthedocs.io>`_.

This is because the protocol has been recorded in several pieces.

There are several cases:
  * each file corresponds to one segment
  * on file has internally several segments. For instance Blackrock. Because you can
    make a pause during recording the same file.
  * several files that themselves contain several segment. Outch!!

Tridesclous use neo.rawio for reading signals. In neo, reading signals
from segments is natural so this notion of segment is the same in tridesclous.

This means that you can feed tridesclous with a list a files (with same channels maps of course)
and tridesclous will generate them as a list of segment naturally.

Note that the raw file should store the data in this order:
*t0c0 t0c1 t0c2 t0c3... t1c0 t1c1 t1c2 t1c3* where *t{i}c{j}* 
is sample no. *i* on channel no. *j*. Thus if the data is loaded in a
2D numpy array `x` where each row is the time series data from one
channel, you can save it in a raw file with this code:
`x.T.tofile(filename)`.



Channel groups, geometry and PRB file
-------------------------------------

There several situation when you want use **channel_group**:
  * The probe have several shank and you want do do
    the spike sorting indenpendently on each shank
  * Only a subset a channels are used.
  * There are dead channels
  * The dataset is several tetrode (N-trode) and so several channel group.

Tridesclous manage theses **channel_group** globally, this means that with same DataIO.

But you will need to construct a catalogue for each channel_group and run the Peeler for
each **channel_group**. Of course, you can compute each group in parallel if you have enough
resources to do it (CPU, RAM, GPU).

If you use probe you naturally know the "geometry" of the probe. For handmade tetrode, you don't
know the geometry.


PRB files are simple files easy to edit with a text editor and where introduced 
in `klusta <http://klusta.readthedocs.io>`_. There are also used in 
`spyking-circus <http://spyking-circus.readthedocs.io/en/latest/code/probe.html>`_
and certainly in other toolbox.

This files describe both **"channel_group"** and there **"geomtery"**. Klusta also needs **"graph"**
but it is ignored in tridesclous.



A typical PRB file look like this, here 8 channels (2 tetrodes)::

    channel_groups = {
        0: {
            'channels': [0, 1, 2, 3],
            'geometry': {
                0 : [-50, 0],
                1 : [0, 50],
                2 : [50, 0],
                3 : [0, -50],
        },
        1: {
            'channels': [4, 5, 6, 7],
            'geometry': {
                4 : [-50, 0],
                5 : [0, 50],
                6 : [50, 0],
                7 : [0, -50],
        },
    }
    


If some of the channels were dead or picked up excessive noise, they can be skipped::

  channel_groups = {
      0: {
       #  'channels': [0, 1, 2, 3],   # if all channels were to be used in spike-sorting
          'channels': [1, 2, 3],   # do not use data from channel at index 0
          'graph':  [],  # Used by klusta but we don't care, SpikeInterface fills this up automatically
          'geometry':  {
             # 0: [0, 20],
              1: [0, -20],
              2: [20, 0],
              3: [-20, 0],
          }
       },
  }

**Units of geometry must be micrometers**



Collections of PRB files exists here:
  * https://github.com/kwikteam/probes/blob/master/neuronexus
  * https://github.com/spyking-circus/spyking-circus/tree/master/probes

tridesclous can automatically download them with the DataIO::

    dataio.download_probe('kampff_128', origin='spyking-circus')
    dataio.download_probe('1x32_buzsaki', origin='kwikteam')


    
Pre-processing
----------------

The pre-processing chain is done as follows:
  1. **Filter**:  high pass (remove low flucatution) + low pass (remove very high freqs) + kernel smooth
  2. **common_ref_removal**:  this substracts sample by sample the median across channels
     When there is a strong noise that appears on all channels (sometimes due to reference) you
     can substract it. This is as if all channels would re referenced numerically to there medians.
  3. **Normalisisation**:  this is more or less a robust z-score channel by channel

  
Important:
  * For spike sorting it is important to compute the filter with forward-backward method.
    This is filtfilt in `matlab <https://fr.mathworks.com/help/signal/ref/filtfilt.html?requestedDomain=www.mathworks.com>`_
    or `in scipy <https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.filtfilt.html>`_.
    The method prevents phase problems. If we apply only a forward filter the peak would be reduced and harder to detect.
    There is a counter part of this filtfilt: if we want to avoid reading the whole dataset forward and then
    the whole dataset backward, we need a margin at the edge of each signal chunk to avoid bad side effect due to filter.
    This parameters is controlled by **lostfront_chunksize**. This leads to more computation and potentially, to small errors.
    Fortunately, when filtering in high frequency (case in spike sorting) 128 sample at 20kHz is sufficent to not make
    mistakes.
  * The smooth is in fact also a filter with a short kernel applied also forward-backward.
  * Filters are applied with a `biquad method <https://en.wikipedia.org/wiki/Digital_biquad_filter>`_.
    High pass + low pass + smooth is computed within the same filter with several sections (cascade).
  * The normalisation is a robust z-score. This means that for each channel **sig_normed = (sig - median) / mad.**
    Mad is `median absolute deviation <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_
    So after pre processing chain, the units of each is **signal to noise ratio**. So as the Gaussian low:
    
      * magnitude 1 = 1 mad = 1 robust sd = 68% of the noise
      * magnitude 2 = 2 mad = 2 robust sd = 95% of the noise
      * magnitude 3 = 3 mad = 3 robust sd = 99.7% of the noise
    
    This is crucial to have this in minds for settings the good threshold.
  * Many software also include a `whitening <https://en.wikipedia.org/wiki/Whitening_transformation>`_ stage.
    Basically this consists of applying to signals the factorized and inversed covariance matrix.
    This is intentionally not done in tridesclous for theses reasons:
    
      * Contrary to what some user think: this does not denoise signals.
      * This must be computed on chunks where there are no spikes. This is hard to do it cleanly.
      * Matrix inversion can lead to numerical error and so some pathological tricks are often added.
  
  
Peak detection and threshold
----------------------------

If one understands that the preprocessed signals units are MAD, the threshold become very intuitive.

The best is to have spikes that have the big signal to noise ratio so that all spikes from a cluster
do not overlap with noise. This is important because if the threshold is too close to the noise
some of the spikes will not be detected and so the cluster will be partial and so the centroids of the
cluster will be wrong. Bad practice!!

There is 2 algorithms to detect spike:
  * **global** : every local extrema above the threshold on at least one channel is considered as peak.
  * **geometrical** same local extrema detection but only on local part given the geometry of the probe.

With high frequency noise the true peak can be noisy and become a double local extremum. When 
you want to avoid that to not having twice the same peak extracted with some sample delayed. 
This is the role of the **peak_span_ms** parameters: when 2 local extrema are in the same span, only the
best is kept.


Waveforms extraction
------------------------

For contruction of catalogue, we need to extract **waveforms**. It is a snipet around each
peak. The feature and cluster will be based on this array.

  * Not all waveforms are extracted in tridesclous only a subset of them. If the **duration**
    choosen for the cataloque is too long them we could have too much peaks. Gathering them
    all of them can be too long and lead too memory overflow. So a random subset is choosen.
    If clusters are clear and dense enough, it is not a problem because it will lead to same
    centroid if we have took waveforms from all peak. For low firing rate neurons having 
    low dense cluster can be a problem and the user need to keep eye open on this.
    In the catalogue peaks that have label -11 means:  they don't have been chossen,
    so they have no waveforms.
  * catalogueconstructor.some_waveforms shape is (nb_peak, nb_sample, nb_channel). On the
    sample axis waveforms are aligned of the extrema.
  * sample width of each waveforms is controlled by **n_left** and **n_right** parameters.
    They must choosen carrefully. If it is too long the total dimenssion will be high and 
    there will be too much noise for clustering. If it is too short, the Peeler (template 
    matching) will fail when substracting leading to noisy residual due to borders.
    A good rule is:
    
       * the median of each cluster need to be back to 0 on each side
       * AND the mad of a cluster need to be back to 1 (noise) on each side.


Noise snippets extraction
----------------------------

A good practice is to extract also some noise snipet that do not overlap peaks.
This will be usefull to compare waveforms of peaks and snippet of noise statistically.

If everything OK, this noise must median=0 and mad=1 because the preprocessed signal
is normalized this way. Checking this is important.

Noise snippet can be also projected in the same sub space than waveforms.
With this, we can compare distance noise to waveforms in the feature space.


Feature extraction
---------------------

On that part we enter in the quarrel zone. It is a subject were people
having introduce new methods in context of spike sorting stand up to defend
religously new ideas.

The problem is pretty simple: the dimenssion of waveforms (nb_sampleXnb_channel)
is too big for clustering algorithm, so we need to reduce this "space" to a smaller "space".
And course we want to reduce this dimentionality while keeping difference between cluster.
The step is so called **feature extraction**.

The most obvious methods is PCA but also SVD, ICA, or wavelet tricks have been proposed.

 For instance., PCA will keep the sample were the variance is the biggest in full space.

Keep in mind, that choosing between PCA or SVD or ICA do not mater so much.

The real problem in fact is how can we do this when we have lot of channels ?
Many tools apply a dimenssion reduction by channel (often PCA) and concatenate them all.
This a well establish mistake because each channel will have the same weight in the feature
space even if it contain noise. A better approach proposed by some tools is to take in a neighborhood 
some channels, concatenate there waveform and to apply a PCA on it. Doing this will automatically
elimate channel with few variance.

Note that when a spike have a clear spatial signature, (for example in dense array a spike can be
seen on 10 channels), taking only the amplitude by channel of spike at the extrema is very naive
but lead to good result. This is called **meak_max** in tridesclous. This is the fastest method
and do not imply alegebric formula.

To not upset anybody we implement several methods, so the user can choose and compare:
  * **global_pca** concatenate all waveform and apply a PCA on it. The best for tetrode (and low channel count)!!
  * **peak_max** get only the peak by channel. Very fast for dense array and not so bad.
  * **pca_by_channel** the most widespread method. Apply a PCA by channel and concatenate them after.
  * **neighborhood_pca** the most sofisticated. For each channel we concatenate the waveforms of the
    neighborhood and apply a PCA on it.
  


Clustering
-----------

Likewise feature extraction, for cluster, imagination and creativity have been large
to introduce in the context of spike sorting some well establish or new fashionable methods
of clustering. While the field of machine learning is exploding todays the number of
sorting algotrithm is naturally become bigger.

Unfortunatly there is a central dilemma : the end user want that the algorithm tell him how many
cluster we have but robust clustering algorithm also want that the end user tell him how much 
cluster there are. Outch!

Of course for very big dataset with tens (or hundreds!) of neurons nobody wants want to try all
**n_cluster** possibilities for discovering the best. There is a strong need of automagic cluster number
finding. This is possible with many methods, for instance density based approaches. But keep in mind
that there are always some parameters somewhere (often hidden to user) that can dramatically
change the cluster number. So don't be credulous when some toolbox propose full automatic spike
sorting, some (hiden) parameters can lead to over clusterised or over merged results.

The approach in tridesclous is to let the user choose the method but validate manually the choice with
the CatalogueWindow. The user eye and intuition is better a weapon than a pre parametrised algotihm.

As we are lazy, we did not implement any of theses methods but use them from `sklean <http://scikit-learn.org>`_ package.
However, two home made methods are implemented here: **sawchaincut** and **prunningshears**, be curious and test then. 
**sawchaincut** and **pruninsheass** is more or less what all we want : a full automated clustering, this works rather well on dense multi-electrode
array when there is a high spatial redundancy (a spike is seen by several channels) but need some manual curation
(like every automated clustering algorithm). **pruninsheass** is the most recent and give better results.

The actual method list is:
  * **kmeans** super classic, super fast but we need to decide **n_cluster**
  * **gmm** gaussian mixture model another classic, we need to decide **n_cluster**
  * **agglomerative** for trying, we need to decide **n_cluster**
  * **dbscan** density based algorithm n_cluster should be automatic. But **eps** parameters
    play a role in the results.
  * **hdbscan** identic with without **eps**. Very usefull for low channel count.
  * **isosplit** : develop by Jeremy Maglang for moutainsort, very impressive on tetrode datasets.
     Unfortunatly works only on linux and sometimes unstable. Must installed separately.
  * **sawchaincut** this is a home made, full automatic, not so good, not so bad, dirty, secret algorithm.
    It is density based. Most beautiful and well isolated clusters should be captured by this one.
  * **pruningshears** this is also a home made stuff. Internal it use hdbscan but have a slow
    and but efficient strategy to explore sub space based on spatial (geometry) information.
    If you don't known which one to choose and you are in a hurry, take this one.


In between sample interpolation
-------------------------------

A non intuitive but strong source of noise is the sampling jitter.
Signals are digitaly sampled between 10kHz and 30kHz so the inter sample interval
is between 33 and 100 μs. A spike been a very short event, the true of the signal peak 
before digitalisation have few chance to be at the same time than the sample. It is in fact
in between 2 samples with a random uniform low.

You can observe it easly: you compute the centroid with the median of a cluster
and  you can see a big overshoot of the variance (done with the mad) around the peak.
This is due to high first derivative et poor alignement.

At the Peeler level, we need to compensate this jitter before substract the centroid from
the signal otherwise the residual will show strong fake peak around the true peak (like the mad).
This is due to jitter. The remain noise amplitude if no jitter compensation can be in order
of magnitude of 2 or 3 mad. The phenonem is really clear with spike with big amplitude at 10kHz.
At upper sample rate with small peak this is less important.

The method used for jitter estimation and cancellation is describe 
`here <https://github.com/christophe-pouzat/PouzatDetorakisEuroScipy2014/blob/master/PouzatDetorakis2014.org>`_.
In short this method based on taylor expanssion is fast and acurate enough.


