.. _step_by_step:

Step by step quickstart
=======================

Here a step by step hands on with the user interface.
Since writting a jupyter notebook is a better methods to keep trace of the spike sorting process.
The *click and play* approach is prefered by beginners.


Launch
------

Ina console::

    tdc


You have a minimalist window with some icons on the left.

Step 1 - Initialize dataset
---------------------------

This step consist of initializing the datasets and configure everything: nb of channels,
probe geometry, nombre of channel groups.

To do that, normally, you have to click on "Initialize dataset", but for trying you can simply use menu
**"File > download dataser > striatum_rat"**.
This will locally download datasets from https://github.com/tridesclous/tridesclous_datasets and configure everything for you.

After that step some information are display about the:
  * DataIO: the deal with with the dataset (nb channel, nb channel group, n segment, smaple rate, durations...)
  * CatalogueConstructor: empty for the moment

Here we have a dataset given by David Robbe and Mostafa Safai recorded with a tetrode in sriatum of a rat.
The signal is sample at 20kHz.


Step 2 - Initialize Catalogue
-----------------------------

This step contain several sub steps to create the catalogue.
This is automatique but need some parameters.

Click on "Initialize" on parameters dialog popup.
Parameters are organized in sections:
  * duration
  * preprocessor
  * peak_detector
  * noise_snippets
  * extract_waveforms

Keep default parameters.
For complete details on parameters see :ref:`parameters` and :ref:`important_details`.

Them you have to choose a **feature method**. For tetrode **"global_pca"** with 5 component sounds good.
You will be able to change this in catalogueWindow later on.

Them you have to choose a **cluster method**. Here let choose **"gmm"** (gaussian mixture model) with 3 clusters.

depending on dataset and choosen method this can take a while.
Here should take around 5 seconds for 300s of signal.

After that step CatalogueConstructor info is updated:
   * nb of detected peak.
   * waveforms shape
   * features shape
   * cluster labels

Step 3 - CatalogueWindow
------------------------------

Click on CatalogueWindow. A window with multi view will help for manual correction of the auto-catalogue (step 2)

This window contain docks than can be arranged as you want. Some of them are organized in tabs, but you can change.
with drag and drop. You can event close or move some view on another screen.
Righ click on the left toolbar to make them appear again.

.. image:: img/snapshot_cataloguewindow.png

On the right toolbar you can manually re run some sub step of the previous chain : detect peak, extract waveforoms,
extract noise snippets, extract features, cluster.

Main view are:
  * spike list
  * cluster list
  * trace view
  * ndscatter
  * waveform view

For more detail see :ref:`catalogue_window`

All view are linked, this means that when click somewhere it will change other view.
For instance, if you select a spike, the trace view will zoom on taht spike and the ndscatter
will highligh the spike.

In the trace view you can zoom Y with the mouse wheel and zoom X with right click.

Make visible one by one each cluster [0, 1, 2]. Play with the noise (label -2) and see what happen in each view.

Click on **"random tour"** in ndscatter. It is a dynamic projection that include many dimension like in GGobi.
It help a lot to anderstand how many cluster we have.

Many view can be custumized with a settings dialog. Some time you have to double click on the view, sometimes a a button.
For instance in **waveformhistviewer** you can choose the *colormap* and the *binsize* with a double lcik in the black area.

In **pairlist**, select each pair and see what happen on  **waveformhistviewer**.
use the mouse wheel to zoom the colormap and right click to zoom X/Y.


Them click on **Compute metric**, this will enable some views: **spikesimilarity**, **clustersimilarity**,
**silhouette**.


Go to **waveformviewer**, select "geometry" or "flatten".


**Cluster list** contain a context menu that propose lot of atcion: merge, split, trash.
Click on "re label cluster by rms".


Now you can see that cluster  0 and 1 are very well isolated but cluster 2 is very close from our choosen threshold.
To simplify we will send it to "trash". This mans that the "peeler" (template matching) will not try to get it.


Now do "save catalogue". We have 2 cluster in our catalogue.

Close the window.

Step 4 - Run peeler
----------------------

Click "run peeler" and keep parameters.

This should take about 10 seconds (for 500s of signal).
The speedup 50x over real tim is due to low number of channel and low number of cluster.



Step 5 - PeelerWindow
-------------------------
Click on "open PeelerWindow"

.. image:: img/snapshot_peelerwindow.png

This windows is for checking, if peeler have corectly do its job, in other words if the catalogue were OK.

You can click on the spike list and the trace auto zoom on the spike.

On the trace view y ou can click on "residual".

The main improtan things to understand here is:
  * the green trace is the **preprocessed** signal (filter and normalized)
  * the magenta trace is the **prediction** = zero + waveform interpolated in between samples.
  * the yellow one is the **residual = preprocess - prediction**

If the catalogue is good and the peeler not buggy, the residual must always under the threhold (white line) for all channels.

You can see that some spike are not labelled (-10) this means that:
   * we forgot a cluster in the catalogue
   * we deliberatly remove this cluster because it is too close for threshold or noise.
   * the interpolation between sample is wrong and the remaining noise due to sampling jitter is bigger
     than standard noise (too bad).



