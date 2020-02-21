"""This just builds the native extension for testing purposes"""

from glob import glob
import os
from setuptools import setup, Extension
import sys
import numpy

LINUX_OPTS = {
    'extra_link_args': [
        '-fopenmp',
    ],
    'extra_compile_args': [
        '-fopenmp',
        '-Wno-strict-prototypes',
        '-Wno-maybe-uninitialized',
        '-O3',
        '-std=c99',
    ],
}
OSX_OPTS = {
    'extra_link_args': [
        '-L/usr/local/opt/libomp/lib',
        '-fopenmp',
    ],
    'extra_compile_args': [
        '-Wno-strict-prototypes',
        '-Wno-uninitialized',
        '-fopenmp',
        '-O3',
    ],
}
WINDOWS_OPTS = {
    'extra_compile_args': [
        '/Ox',
    ]
}


if os.name == 'posix':
    if sys.platform.startswith('darwin'):
        OPTS = OSX_OPTS
    else:
        OPTS = LINUX_OPTS
else:
    OPTS = WINDOWS_OPTS

setup(
    name='gamred_native',
    version='0.0.1',
    packages=[],
    # @gmrukwa: https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        'numpy>=0.12.1',
    ],
    setup_requires=[
        'numpy>=0.12.1',
    ],
    python_requires='>=3.6',
    ext_modules=[
        Extension('gamred_native',
                  sources=glob('gamred_native/*.c'),
                  include_dirs=['gamred_native', numpy.get_include()],
                  define_macros=[
                      ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
                  ],
                  **OPTS),
    ],
)
