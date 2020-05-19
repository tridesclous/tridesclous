.. _realtime:

Real time
=========

Overview
--------

The Peeler have been design to do the processing chunk by chunk in mind.
So the offline Peeler is also adapted to OnlinePeeler to be used in real time.
OnlinePeeler is a pyacq Node.

`pyacq <https://github.com/pyacq/pyacq>`_ is a system for distributed
data acquisition and stream processing. It support some device use in
electrophysiology (Blackrock, Multichannel system, Measurement computing,
National Instrument, OpenEphys...). Pyacq offer the possibility to dtsribute the
computing on several machines. So it is particulary usefull in online 
spike sorting context when high channel count. The user will be able
to distribute on several machines: the acquisition itself, the OnelinePeeler
and some display.


pyacq and Tridesclous do not offer a strict real real time engine but
an online engine which latency whihch can be controlled by the chunksize.



Installation for quick demo
---------------------------

You need to install pyacq (from source)::

    pip install --upgrade https://github.com/pyacq/pyacq/archive/master.zip


And then you should be able to do::

    tdc rt_demo



Integration with open-ephys GUI
-------------------------------

  * **Step 1 ** install pyacq::
  
      pip install --upgrade https://github.com/pyacq/pyacq/archive/master.zip

  * **Step 2**
  
    Installation openephys from source (with compilation)
    `See <https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491544/Installation)>`_

    
  * **Step 3**
  
    Download and compile the `PythonPlugin <https://github.com/NeuroNetMem/PythonPlugin>`_ for openephys-GUI
  
    Select the branch **cmake_build** (mai 2020)

    
  * **Step 4**
  
    Compile tridesclous python plugin.
    
    Please adapt path to your isntallation
    
    Copy source to PythonPlugin::
    
        cp -R /home/samuel/Documents/tridesclous/utils/OpenephysTridesclousPyPlugin /home/samuel/Documents/open-ephys/PythonPlugin/python_modules
    
    Select conda (or virtualenv) and compile it::
    
        activate myenv
        cd /home/samuel/Documents/open-ephys/PythonPlugin/python_modules/OpenephysTridesclousPyPlugin
        python setup.py build_ext --inplace
    
    You should see a file **OpenephysTridesclousPyPlugin.cpython-37m-x86_64-linux-gnu.so**
    
    
  * **Step 5**
    
    launch openephys
    
    Construct a chain with **"Python Sink"**
    
    Inside the python plugin Select the compiled tridesclous plugin: **OpenephysTridesclousPyPlugin.cpython-37m-x86_64-linux-gnu.so**
    
    
  * **Step 6**
    run it::
  
      tdc rt_openephys --prb_file=/path/to/prb_file.prb
      









.. automodule:: tridesclous.online



