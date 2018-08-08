.. _catalogue_window:

Catalogue window
================

CatalogueWindow is important, it helps users to manually validate and correct some
mistakes made by the automatic steps (filtering, threshold, feature extraction, clustering, ...)

All views are linked, this means that some actions interact with other views.

All views can be customised by settings (often with a double in the view).

Here some detail on each view.

.. NOTE::
  
    A lot of efforts have been put in making this UI as smooth as possible but for big datasets (>100 channels)
    the CatalogueWindow can be slow for some actions because it trigs a full refresh on other views
    and no computation can be done (re compute centroid for instance). So be patient and smart.


.. NOTE::

    For some manual actions on catalogue, CatalogueWindow can suddenly crash. While this is annoying,
    you should not lose any data. Just open again the same dataset and you should be in a previous
    situation.
    
    In cases of crashes please send an issue on github `<https://github.com/tridesclous/tridesclous/issues>`_
    It takes only minutes and helps a lot to make tridesclous more stable. Please copy/paste the error
    message in the console in the issue and describe very briefly the actions that triggered the crash.


Toolbar on the left
-------------------

You can:

  * **Make catalogue for Peeler**: this construct all waveform centroid + first and second derivative + interpolated 
    waveforms. This is needed by the Peeler. This initial catalogue is valid for one channel group only.
    This is saved inside the working directory of the DataIO in the appropriate channel_group sub dir.
    Do this at the end when the clusters seem OK before launching the Peeler on the entire dataset.
  * **refresh**: this reloads data from DataIO and refreshes all views. This is extremly useful when you
    externally in jupyter for instance change or test some methods that transform data.
    This avoids to restart the UI.
  * **New peaks**: re detect peaks, you change the threshold for a new detection here.
  * **New waveforms**: take some other (or all) waveforms with other parameters.
  * **Clean waveforms**: detect bad (allien) waveform.
  * **New noise snippet**: extract new noise snippet.
  * **New features**: choose other method or parameters for features extraction.
  * **New cluster**: choose other method or parameters for clustering.
  * **Compute metrics**: the compute metric on clusters : similarity, ratio_similarity, silhouet, ...
  * **Help**: a button that magically telestranport you in that this current page.
  * **Savepoint**: duplicate the working dir, that you can manually go back in the past 
    with a funny game of folder renaming.


Peak list
----------

.. autoclass:: tridesclous.gui.peaklists.PeakList

Cluster list
------------

.. autoclass:: tridesclous.gui.peaklists.ClusterPeakList

Trace viewer
------------

.. autoclass:: tridesclous.gui.traceviewer.CatalogueTraceViewer


NDScatter
---------

.. autoclass:: tridesclous.gui.ndscatter.NDScatter

Pair list
---------

.. autoclass:: tridesclous.gui.pairlist.PairList

Waveform viewer
---------------

.. autoclass:: tridesclous.gui.waveformviewer.WaveformViewer

Waveform hist viewer
--------------------

.. autoclass:: tridesclous.gui.waveformhistviewer.WaveformHistViewer

Feature on time viewer
----------------------

.. autoclass:: tridesclous.gui.waveformhistviewer.WaveformHistViewer

Spike similarity
----------------

.. autoclass:: tridesclous.gui.similarity.SpikeSimilarityView

Cluster similarity
------------------

.. autoclass:: tridesclous.gui.similarity.ClusterSimilarityView

Cluster ratio similarity
------------------------

.. autoclass:: tridesclous.gui.similarity.ClusterRatioSimilarityView

Silhouette
----------

.. autoclass:: tridesclous.gui.silhouette.Silhouette



