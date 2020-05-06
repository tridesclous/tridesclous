Installation
============


If your are familiar with python simply install the dependency list as usual.

tridesclous works with python 3 only.


If these are your first steps in the python world there 2 main options:
  * install python and dependencies with anaconda distribution (prefered on window or OSX)
  * use python from your system (in a virtual environement) and install dependencies with standard pip (prefered on Linux Ubuntu/Debian/Mint)

Note that you are free to install Anaconda on Linux but conda add an heavy layer of package
management in parallel of your linux distro pacake management that can be messy.

Here 2 recipes to install tridesclous in an "environement".
If you don't known what an "environement" is : remember that it is an isolated installation
of python with a fixed version and many libraries (modules) with also fixed version  somwhere in a folder in your profile.
This folder won't be affected by upgrading your system and so it should work always.
This quite important because other spike projects don't use same libraries version (for instance PyQt4 vs PyQt5).
If you want to compare them, you will need environement.



Case 1 : with anaconda (prefered on window or OSX)
--------------------------------------------------

Do:

  1. Download anaconda here https://www.continuum.io/downloads. Take **python 3.7**
  2. Install it in user mode (no admin password)
  3. Lanch **anaconda navigator**
  4. Go on the tab **environements**, click on **base** context menu.
  5. **Open Terminal**
  6. For the basic::
    
       conda install scipy numpy pandas scikit-learn matplotlib seaborn tqdm openpyxl numba
       conda install -c conda-forge hdbscan
     
  
  7. For GUI and running example::
  
       conda install pyqt=5 jupyter
       pip install pyqtgraph==0.10 quantities neo
     
     
  8. And finally install tridesclous from github::
  
       pip install https://github.com/tridesclous/tridesclous/archive/master.zip




Optional if you're up for a fight and you really want fast computing with OpenCL:

  1. install driver for GPU (nvidia/intel/amd), this is quite hard in some cases because you need to download some OpenCL (or cuda) toolkit.
  2. Download PyOpenCl here for windows : http://www.lfd.uci.edu/~gohlke/pythonlibs/
  3. cd C:/users/...../Downloads
  4. pip install pyopencl‑2019.1.2+cl12‑cp37‑cp37m‑win_amd64.whl
 
  

.. WARNING::

    Some user with windows report strong problems. Anaconda is hard to install and also in
    the tridesclous GUI, when a file dialog should open python surddenly crash.
    One possible reason is : on Dell computer an application **Dell Backup and Recovery**
    is installed. This application also used Qt5. For some versions (1.8.xx and maybe others)
    of **Dell Backup and Recovery** this Qt5 have bug and theses Qt5 ddl are mixed up with
    anaconda Qt5, this lead to a total mess hard to debug. So if you have a Dell, you
    should upgrade **Dell Backup and Recovery** or remove it.


Case 2 : with pip (prefered on linux)
-------------------------------------

Here I propose my favorite method that install tridesclous with debian like distro in an
isolated environement with virtualenvwrapper. Every other method is also valid.

Open a terminal and do:

  1. sudo apt-get install virtualenvwrapper
  2. mkvirtualenv  tdc   --python=/usr/bin/python3.6    (or python3.5)
  3. workon tdc
  4. pip install scipy numpy pandas scikit-learn matplotlib seaborn tqdm openpyxl hdbscan numba
  5. pip install PyQt5 jupyter pyqtgraph==0.10 quantities neo
  6. pip install https://github.com/tridesclous/tridesclous/archive/master.zip


  
   
Big GPU, big dataset OpenCL, and CO.
------------------------------------

OpenCL is a language for coding parralel programs that can be run on GPU (graphical processor unit) and
also on CPU multi core.

Some heavy part of the processing chain is coded both in pure python (scipy/numpy) and OpenCL.
So, TDC can be run in any situations.
But if the dataset is too big, you can stop mining cryto-money for while and can try to run TDC on a big-fat-gleaming GPU.
You should gain some speedup if the number of channel is high.


Depending, the OS and the hardware it used to be difficult to settle correctly the OpenCL drivers (=opencl ICD).
Now, it is more easy (except on OSX, it is becoming more difficult, grrrr.)


Here the solution on linux ubuntu 18.04 / debian  :
   
   1. workon tdc
   2. For intel GPU: sudo apt-get install beignet
      For nvidia GPU: sudo apt-get install nvidia-opencl-XXX
   3. sudo apt-get instll opencl-headers ocl-icd-opencl-dev libclc-dev ocl-icd-libopencl1
   4. pip install pyopencl

To have more recent nvidia driver:

  1. Install ppa https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa
  2. sudo apt-get install nvidia-headless-440
  3. sudo apt-get install nvidia-utils-440

   
If you have a recent laptop you can also try the new neo-icd for intel GPU.
   
If you don't have GPU but a multi core CPU you can use POCL on linux:

   sudo apt-get install pocl


Here on windows a solution:

    1. If you have nvidia or intel a a recent windows 10, then opencl driver are already installed
    2. Download PyOpenCl here for windows : http://www.lfd.uci.edu/~gohlke/pythonlibs
    3. Take the pyopencl file that match your python
    4. cd C:/users/...../Downloads
    5. pip install pyopencl‑2019.1.2+cl12‑cp37‑cp37m‑win_amd64.whl (for instance)



   
Ephyviewer (optional)
---------------------



With neo (>=0.8) installed, if you want to view signals you can optionally install ephyviewer with::
    
    pip install ephyviewer


Upgrade tridesclous
-------------------

There are 3 sources for upgrading tridesclous package depending your need.


For **official** release at pypi::

    pip install --upgrade tridesclous


For **up-to-date** or **new-featured** version get the master version on github::

  pip install --upgrade https://github.com/tridesclous/tridesclous/archive/master.zip


For **work-in-progress** or **in-debug** version, take master version on my personal repo::

  pip install --upgrade https://github.com/samuelgarcia/tridesclous/archive/master.zip





