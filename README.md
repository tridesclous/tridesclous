# trisdesclous: spike sorting with a [French touch](https://fr.wikipedia.org/wiki/French_touch_(informatique)).

![icon](tridesclous/gui/icons/png/main_icon.png)

Authors: Christophe Pouzat and Samuel Garcia


**tris des clous** is a very dishonest translation of **spike sorting** to French.

It is pronounced [tree day clue] for English speakers.

The documentation is here: http://tridesclous.readthedocs.io/

This module provides some functions and examples of spike sorting.
It is our purpose to teach good practices in spike sorting techniques.

You can also use it for real data of course. 
The forest of spike sorting tools is dense and *tridesclous* is a new tree.
Be curious and try it.

*tridesclous* is a complete rewrite of our old (and unsuccessful) tools (SpikeOmatic, OpenElectrophy, ...)
with up-to-date (in 2017) python modules to simplify our codes.

tridesclous:
  * should make it easy to keep a trace of your spike sorting process in
    a jupyter notebook.
  * offer a simple UI written in Qt for interactive exploration.
  * can be used for online spikesorting with pyacq.

In short, *tridesclous* forces you to write a script for spike sorting but 
you also benefit from a simple UI for offline spike sorting (and online soon).

Bonus:
  * tridesclous is quite fast. For a tetrode dataset, you can expect X30 speedup over real time on a simple laptop.
  * some pieces of algorithm are written both in pure python (numpy/scipy/...) and OpenCL (filter, peak detetion, ...). So *tridesclous* should be efficient for large array (>=64 channel) soon.
  * each piece of the algorithm is written with chunk by chunk in mind. So the offline *tridesclous* is not agressive for memory.
  * tridesclous used [neo](https://github.com/NeuralEnsemble/python-neo) for reading dataset. So many format are available (Raw, Blackrock, Neuralynx, Plexon, Spike2, Tdt, ...)
  * tridesclous is open source and based on true opensource stack.
  * some datasets are available for testing it now here https://github.com/tridesclous/tridesclous_datasets


# Installation

http://tridesclous.readthedocs.io/en/latest/installation.html

# Launch

http://tridesclous.readthedocs.io/en/latest/launch.html


# Screenshots

## Offline Catalogue Window
![snapshot](doc/img/snapshot_cataloguewindow.png)

## offline Peeler Window
![snapshot](doc/img/snapshot_peelerwindow.png)

## Online Peeler in a pyacq.Node
![snapshot](doc/img/online_tridesclous.gif)
