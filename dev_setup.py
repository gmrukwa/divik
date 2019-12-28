"""This just builds the native extension for testing purposes"""

from glob import glob
from setuptools import setup, Extension
import numpy

setup(
    packages=[],
    # @gmrukwa: https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        'numpy>=0.12.1',
    ],
    python_requires='>=3.5,<=3.7',
    ext_modules=[
        Extension('gamred_native',
                  sources=glob('gamred_native/*.c'),
                  include_dirs=['gamred_native', numpy.get_include()],
                  define_macros=[
                      ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
                  ]),
    ],
)
