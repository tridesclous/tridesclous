.. _overview:

Overview
======

Pronouce it [tree day clue] in English.

The primary goal of tridesclous is to provide a toolkit to teach good practices in spike sorting techniques.
This tools is now mature and can be used for experimental data.

Authors: Christophe Pouzat and Samuel Garcia

General workflow
-------------------

Many tools claim to be "automatic" for spike sorting.
In tridesclous we don't, the workflow is:

  1. Construct catalogue. This part is automatic but needs carrefully chosen parameters.
     This is more or less the legagy chain of spike sorting = preprocessing+waveform+feature+clustering
     This can be done with a small subset of the whole dataset as long as it stationary.
  2. Check and correct the catalogue. **This part is manual.** It is done through a user interface.
     Many viewers help the end user make good decisions: change the threshold, enlarge waveform shape,
     change feature method, change clustering algorithm and of course merge and split clusters.
     This part is crucial and must be performed to clean clusters.
  3. "Peel spikes" this is the real spike sorting. It is a template matching approach that substracts spikes
     for signals as long as some spike matches the catalogue. This part is can be run offline as fast as
     possible or online (*soft* real time) if fast enought.
  4. Check the final result with a dedicated user interface. No manual action is required here.


Why is it different from other tools:

  * Today, some other tools propose more or less the same workflow except the central step: check the catalogue before
    the template matching! This is critical. Theses tools often over split some clusters and this leads to long, useless
    and uncontrolled manual merges (or split sometimes) at the end of the process.
  * The catalogue is done only on a small part of the dataset, let say some minutes. Some other tools try to cluster
    spike on long recording but there are many chances that signal, noise, amplitude will not be stationnary.
    Clusering on a small part is faster and leads to more stable results.
  * The user interface (for catalogue check/correct and peeler check) is part of the same package.
    So viewer are closely linked to methods and everything is done to alleviate the patologies of these methods.


..
    Comparison with other tools
    -------------------------------

      * klusta
      * kilosort + phy
      * spyking circus + phy
      * montain sort
      * yass


