.. _catalogue_window

Catalogue window
================

CatalogueWindow is important, it helps users to manually validate and correct some
mistake by automatic step (filtering, threshold, feature extraction, clustering, ...)

All view are linked, this means that some actions interact with other views.

All view can be customised by settings.

Here some detail on each view.

.. NOTES::
  
    Many effort have been done to make this UI as smooth as possible but for big datasets (>100 channels)
    the CatalogueWindow can be slow for some actions because, it trig a full refresh on other view
    and no computation must be done (re compute centroid for instance). So be patient and smart.


Toolbar on the left
-------------------

You can:

  * **Save the catalogue**: this construct all waveform centroid + first and second derivative + interpolated 
    waveforms. This is need by the Peeler. This initial catalogue is valid for one channel group only.
    This is save inside the working directoty of the DataIO in the appropriate channel_group sub dir.
    Do this at the end when the clusters seems OK before lanching the Peeler on the entire dataset.
  * **refresh**: this reload data from DataIO and refresh all views. This is extremly usefull when you
    externally in jupyter for instance change or test some methods that transform data.
    This avoid to restart the UI.
  * **New peaks**: re detect peaks, you change the threshold for a new detection here.
  * **New waveforms**: take some other (or all) waveforms with other parameters.
  * **New noise snippet**: extract new noise snippet.
  * **New features**: choose other method or or parameters for features extraction.
  * **New cluster**: choose other method or or parameters for clusering.
  * **Compute metrics**: the compute metric on clusters : similarity, ratio_similarity, silhouet, ...


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



