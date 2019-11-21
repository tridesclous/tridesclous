Launch
======


There are several ways to launch **tridesclous**:
  * Inside a **jupyter notebook** good option to make example in your lab
  * Inside python script, if you construct automatic pipeline
  * With a Graphical User Interface (GUI): the least frightening for beginners

  
Please read carrefully, the how and the why for each methods.


Method 1: Launching tridesclous inside a jupyter notebook or a python script
----------------------------------------------------------------------------

This is the best method that authors recommend for users.
People that never code with python or any language can be scared about this but this is easy!

Keep in mind that for reproducible science you need to keep track of what you are doing and this is the best way.

Also note that magic all-in-one command line and GUI keep you away from deep understanding of your spike sorting tool chain.
Writting simple code block can help you a lot to realize and overcome difficulties.



So for this method:
  1. Launch jupyter notebook (easy if you have anaconda)
  2. Copy/paste:

    * `this notebook <https://github.com/tridesclous/tridesclous/blob/master/example/example_locust_dataset.ipynb>`_
    * `or this one <https://github.com/tridesclous/tridesclous/blob/master/example/example_olfactory_bulb_dataset.ipynb>`_
    
  3. Read it carrefully.
  4. Modify it and do your spike sorting.
  
  
Please also explore the [examples folder](https://github.com/tridesclous/tridesclous/tree/master/example) that contains
some example on some dataset.



Method 2: Launching tridesclous GUI
------------------------------------

Here's the method for lazy people (or people in a hurry).

For demagogical reasons, we wrote a GUI in Qt for launching tridesclous.



Do:
  * open a terminal::
  
      workon tdc  (or source activate tdc for windows)
      tdc
  
  * In the GUI you must:
      1. File>Initialize example_locust_dataset
      2. Select a channel group in **chan_grp**
      3. **Initialize catalogue**
      4. **Open catalogue Window**
      5. save catalogue when happy
      6. **run Peeler**
      7. **open PeelerWindow**

See :ref:`step_by_step` for complete explanation.




