from setuptools import setup
import os

d = {}
exec(open("tridesclous/version.py").read(), None, d)
version = d['version']

install_requires = [
                    'numpy<1.22',
                    'scipy',
                    'pandas',
                    'openpyxl',
                    'scikit-learn>=0.22.2',
                    'matplotlib',
                    'seaborn',
                    'neo>=0.8',
                    'tqdm',
                    # 'PyQt5',  make conda buggy
                    'pyqtgraph',
                    'joblib',
                    'numba',
                    'hdbscan',
                    'loky',
                    ]
extras_require={
                            'online' : ['pyacq',],
                            'opencl' : ['pyopencl'],
                        }

long_description = ""

setup(
    name = "tridesclous",
    version = version,
    packages = ['tridesclous', 'tridesclous.gui', 'tridesclous.gui.icons',
                'tridesclous.online', 'tridesclous.scripts', 'tridesclous.tests'],
    install_requires=install_requires,
    extras_require = extras_require,
    author = "C. Pouzat, S.Garcia",
    author_email = "",
    description = "offline/online spike sorting with french touch that light the barbecue",
    long_description = long_description,
    entry_points={
          'console_scripts': ['tdc=tridesclous.scripts.tdc:main'],
          #~ 'gui_scripts': ['tdcgui=tridesclous.scripts.tdc:open_mainwindow'],
        },
     license = "MIT",
    url='https://github.com/tridesclous/trisdesclous',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering']
)
