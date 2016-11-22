from setuptools import setup
import os

import tridesclous

install_requires = [
                    'numpy',
                    'scipy',
                    'pandas',
                    'sklearn',
                    'matplotlib',
                    'seaborn',
                    'tables',
                    ]
extras_require={ 'gui' : ['PyQt5', 'pyqtgraph', 'matplotlib'],
                            'online' : 'pyacq',
                        }

long_description = ""

setup(
    name = "tridesclous",
    version = tridesclous.__version__,
    packages = ['tridesclous', ],
    install_requires=install_requires,
    extras_require = extras_require,
    author = "C. Pouzat, S.Garcia",
    author_email = "",
    description = "Simple Framework for spike sorting python.",
    long_description = long_description,
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
