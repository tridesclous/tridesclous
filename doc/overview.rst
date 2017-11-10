Overview
======

Pronouce it [tree day clue] for english speaker.

The primary goal of tridesclous is to provide some material to teach good practices in spike sorting technics.
This tools is now more mature and can be used for experimental data.


General workflow
-------------------

Many tools claim to be "automatic" for spike sorting.
In tridesclous we don't, the workflow is:

  1. Construct catalogue. This part automatic but need carrefully chossen parameters.
     This is more or less the legagy chain of spike sorting = preprocessing+waveform+feature+clustering
     This can be done with a small subset of the all dataset as long as it stationary.
  2. Check and correct the catalogue. **This part is manual.** It is done through a user interface.
     Many viewer help the end user to take good decisions : change the threshold, enlarge waveform shape,
     change feature method, change clustering algorithm and of course merge and split clusters.
     This part is crutial an dmust to clean clusters.
  3. "Peel spikes" this is the real spike sorting. It is a template matching approach that substarct spikes
     for signals as long as some spike match the catalogue.
  4. Check the final result with a dedicated user interface. No manual action are require here.


Why is it different from other tools:

  * Today, some other tools propose more or less the same workflow except the central step: check the catalogue before
    the template matching! This is crutial. Theses tools often over split some cluster and this lead to long, useless
    and uncontrolled manual merge (or split sometimes) at the end of the process.
  * The catalogue is done only on small part of the dataset, let say some minuts. Some other tools try to cluster
    spike on long recording but there many chance that signal, noise, amplitude will not be stationnary.
    Clusering on a small part is faster and lead to more stable results.
  * user interface (for catalogue check/correct and peerler check) is part of the same package.
    So viewer are closely link to methods and everything is done to zoom of patology of theses methods.


..
    Comparison with other tools
    -------------------------------

      * klusta
      * kilosort + phy
      * spyking circus + phy
      * montain sort
      * yass


