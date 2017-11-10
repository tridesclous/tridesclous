Important details
=================

A list a miscellaneous details, some of them are related to data handling
and other to detail of methods at different steps of the chain.


Multi segments
--------------

Many recording are split in several "Segments". Segment here refer to 
`neo meaning <http://neo.readthedocs.io>`_.

This is because in protocol have been recorded in several pieces.

There are several cases:
  * each file correspond to one segment
  * on file have internally several segment. For instance Blackrock. Because you can
    make a pause during recording.
  * several files that themself contains several segment. Outch!!

Tridesclous use neo.rawio for reading signals. In neo, reading signals
from segment is natural so this notion of segment is the same in tridesclous.

This means that you can feed tridesclous with a list a files (with same channels maps of course)
and tridesclous will generate them as a list of segment naturally.



Channel groups, geometry and PRB file
-------------------------------------

There several situation when you want use **channel_group**:
  * The probe have several shank and you want you want do do
    the spike sorting indenpendently on each shank
  * Only a subset a channel are used.
  * There dead channels
  * The dataset is several tetrode and so several channel group.

Tridesclous manage theses channel_group globaly, this means that withe same DataIO.

But you will need to construct a catalogue for each channel_group and run the Peeler for
each channel_group. Of course, you can compute each group in parallel if you have enough
ressource to do it (CPU, RAM, GPU).

If you use probe you naturally known the "geometry" the probe. For hand made tetrode, you don't
known the geometry.


PRB file are simple files easy to edit with text editor and where introduced 
in `klusta <http://klusta.readthedocs.io>`_. There are also used in 
`spyking-circus <http://spyking-circus.readthedocs.io/en/latest/code/probe.html>`_
and certainly in other toolbox.

This files describe both **"channel_group"** and there **"geomtery"**. Klusta also need **"graph"**
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
    

**Units of geometry must be micrometers**



Collections of PRB files exists here:
  * https://github.com/kwikteam/probes/blob/master/neuronexus
  * https://github.com/spyking-circus/spyking-circus/tree/master/probes

tridesclous can automatically download them with the DataIO::

    dataio.download_probe('kampff_128', origin='spyking-circus')
    dataio.download_probe('1x32_buzsaki', origin='kwikteam')


    
Pre processing
----------------


Peak detection and threshold
--------------------------------


Waveforms extraction
------------------------


Noise snippets extraction
----------------------------


Feature extraction
---------------------


Clustering
-----------


In between sample interpolation
------------------------------------
