.. _api:

API for scripting
=========

Scripting with tridesclous can avoid unnecessary click and play with the GUI
in parameter dialogs if you need to process many files.
Almost everything can be done 3 classes:
  * :class:`DataIO` for configuration of the dataset, format selection, PRB file assignement, ...
  * :class:`CatalogueConstructor` run all steps to construct the catalogue : signal processing, fetaure, clustering, ...
  * :class:`Peeler` run the template matching engine

Of course everything done by script can still be check and modify with the GUI (MainWindow, CatalogueWindow and PeelerWindow).


**The best is to read** `examples in the git repo <https://github.com/tridesclous/tridesclous/tree/master/example>`_


Classes API
----------------

.. automodule:: tridesclous


 
