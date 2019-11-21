.. _overview:

Overview
========

**tris des clous** is a very dishonest translation of **spike sorting** to French.

Pronouce it [tree day clue] in English.

The primary goal of tridesclous was to provide a toolkit to teach good practices in spike sorting techniques.
Trideslcous is now mature and can be used to sort spike on tetrode up to neuropixel probe recorded dataset.

Tridesclous is both:

  * an offline spike sorter tools able to read many file format
  * a realtime spike sorting combined with `pyacq <https://github.com/pyacq/pyacq>`_

  
Main features
-------------

  * template matching based mathod
  * offer several alternative methods at several processing steps of the chain
  * offer a UI written in Qt for interactive exploration.
  * use `neo <https://github.com/NeuralEnsemble/python-neo>`_ for reading dataset. So many format are available (Raw, Blackrock, Neuralynx, Plexon, Spike2, Tdt, OpenEphys...)
  * use hardware acceleration with opencl : both gpu and multicore cpu
  * use few memory
  * have built in dataset to try it
  * quite fast For a tetrode dataset, you can expect X30 speedup over real time on a simple laptop.
  * have an simple python API. Easy to write notebook or build custom pipeline.
  * multi-platform
  * open source based on a true open source stack

The forest of spike sorting tools is dense and *tridesclous* is a new tree.
Be curious and try it.


General workflow
-------------------

Many tools claim to be "automatic" for spike sorting.
In tridesclous we don't, the workflow is:

  1. Construct catalog. This part is automatic but needs carefully chosen parameters/methods.
     This is more or less the legacy chain of spike sorting = preprocessing+waveform+feature+clustering
     This can be done with a small subset of the whole dataset as long as it is stationary.
  2. Check and correct the catalog. **This part is manual.** It is done through a user interface.
     Multiple views in the interface help the end user make good decisions: change the threshold, enlarge waveform shape,  change feature method, change clustering algorithm and of course merge and split clusters.
     This part is crucial and must be performed to clean clusters.
  3. "Peel spikes". This is the real spike sorting. It is a template matching approach that subtracts spikes for signals as long as some spike matches the catalog. This part can be run offline as fast as possible or online (*soft* real time) if fast enough.
  4. Check the final result with a dedicated user interface. No manual action is required here.

Manual checking part **2** and **4** can be optional if you like to use black-box style spike sorting tools.


Why is it different from other tools:

  * Today, some other tools propose more or less the same workflow except the central step: check the catalog before the template matching! This is critical. These tools often over split some clusters and this leads to long, useless  and uncontrolled manual merges (or split sometimes) at the end of the process.
  * The catalog is built only from a small part of the dataset, let say some minutes. Some other tools try to cluster spike on long recording but there are many chances that signal, noise, amplitude will not be stationary.
    Clustering on a small part is faster and leads to more stable results.
  * The user interface (for catalogue check/correct and peeler check) is part of the same package.
    So viewers are closely linked to methods and everything is done to alleviate the pathologies of these methods.


Online spike sorting
--------------------

If you have a `pyacq <https://github.com/pyacq/pyacq>`_ compatible device (Blackrock, Multi channel system, NiDaqMx, Measurement computing, ...) you can also test tridesclous online during the experiment. See `online_demo.py <https://github.com/tridesclous/tridesclous/blob/master/example/online_demo.py>`_

In pyacq, you can build your own viewers in a custom "Node", so you should be able to monitor during the recording what you need (receptive field, ...)




..
    Comparison with other tools
    -------------------------------

      * klusta
      * kilosort + phy
      * spyking circus + phy
      * mountain sort
      * yass


