# trisdesclous: spike sorting with a [French touch](https://fr.wikipedia.org/wiki/French_touch_(informatique)).

![icon](tridesclous/gui/icons/png/main_icon.png)

The documentation is here: http://tridesclous.readthedocs.io/

Authors: Christophe Pouzat and Samuel Garcia

**tris des clous** is a very dishonest translation of **spike sorting** to French.

Pronouce it [tree day clue] in English.

The primary goal of tridesclous is to provide a toolkit to teach good practices in spike sorting techniques.
This tools is now mature and can be used for experimental data.

The forest of spike sorting tools is dense and *tridesclous* is a new tree.
Be curious and try it.

*tridesclous* is a complete rewrite of our old (and unsuccessful) tools (SpikeOmatic, OpenElectrophy, ...)
with up-to-date (in 2017) python modules to simplify our codes.

tridesclous:
  * should make it easy to keep a trace of your spike sorting process in
    a jupyter notebook.
  * offer a UI written in Qt for interactive exploration.
  * can be used for online spikesorting with [pyacq](http://pyacq.readthedocs.io). See [online_demo.py](https://github.com/tridesclous/tridesclous/blob/master/example/online_demo.py)


Bonus:
  * tridesclous is quite fast. For a tetrode dataset, you can expect X30 speedup over real time on a simple laptop.
  * some pieces of algorithm are written both in pure python (numpy/scipy/...) and OpenCL (filter, peak detetion, ...). So *tridesclous* is efficient for large array (>=64 channel).
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


# Status

[![readthedocs](https://readthedocs.org/projects/tridesclous/badge/?version=latest&style=flat)]( http://tridesclous.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/tridesclous/tridesclous.svg?style=svg)](https://circleci.com/gh/tridesclous/tridesclous)
[![Appveyor](https://ci.appveyor.com/api/projects/status/7cqmevwu0r3sq87e?svg=true)](https://ci.appveyor.com/project/samuelgarcia/tridesclous)
