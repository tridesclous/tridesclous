Installation
============


If your are familiar with python simply install the depency list as usual.

trideclous works with python 3 only.



If this is your first steps in the python world you have 2 options:
  * install python and dependencies with anaconda distribution (prefered on window or OSX)
  * use python from your system (in a virtual environement) and install dependencies with standard pip (prerered on linux ubuntu/debian/mint)

Note that you are free to install anaconda on linux.




Case 1 : with anaconda (prefered on window or OSX)
--------------------------------------------------

Do:

  1. Download anaconda here https://www.continuum.io/downloads. Take **python 3.6**
  2. Install it in user mode (no admin password)
  3. Lanch **anaconda navigator**
  4. Go on the tab **environements**, click on **root** context menu.
  5. **Open Terminal**
  6. For the basic::
    
       conda install scipy numpy pandas scikit-learn matplotlib seaborn
     
  
  7. For GUI and running example::
  
       conda install pyqt=5 jupyter
       pip install pyqtgraph==0.10 neo==0.5
     
     
  8. And finally install tridesclous from github::
  
       pip install https://github.com/tridesclous/tridesclous/archive/master.zip



   
     


Optional if you want fight and you really want fast computing with OpenCl:

  1. install driver for GPU (nvidia/intel/amd), this is quite hard in some case because you need to download some OpenCL (or cuda) toolkit.
  2. Download PyOpenCl here for windows : http://www.lfd.uci.edu/~gohlke/pythonlibs/
  3. cd C:/users/...../Downloads
  4. pip install pyopencl‑2016.2.1+cl21‑cp36‑cp36m‑win_amd64.whl
 
  




Case 2 : with pip (prefered on linux)
-------------------------------------

Here I propose my method that install tridesclous in an isolateted environement with virtualenvwrapper.
Every method is also valid.

Open a terminal and do:

  1. sudo apt-get install virtualenvwrapper
  2. mkvirtualenv  tdc   --python=/usr/bin/python3.5
  3. workon tdc
  
  4. pip install scipy numpy pandas scikit-learn matplotlib seaborn
  5. pip install PyQt5 jupyter pyqtgraph==0.10 neo==0.5
  6. pip install https://github.com/tridesclous/tridesclous/archive/master.zip



   
Optional if you want fight and you really want fast computing with OpenCl and you are on linux:
   
   1. workon tdc
   2. For intel GPU: sudo apt-egt install beignet
      For nvidia GPU: sudo apt-egt install nvidia-opencl-340
   3. sudo apt-get instll opencl-headers ocl-icd-opencl-dev libclc-dev ocl-icd-libopencl1
   4. pip install pyopencl



Update tridesclous
------------------

There are no official release on pypi at the moment, so you need to take the in developpement code on github.



For updating to not repeat installation of dependencies, just uninstall and reinstall::

  pip uninstall tridesclous
  pip install https://github.com/tridesclous/tridesclous/archive/master.zip

