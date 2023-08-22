import os
import sys
from glob import glob

from setuptools import Extension

################################################
# Hack for https://github.com/python-poetry/poetry/issues/6154#issue-1336927568

import subprocess

try:
    import numpy
except ImportError:
    subprocess.run(["pip", "install", "oldest-supported-numpy"])


import numpy


################################################

LINUX_OPTS = {
    "extra_link_args": [
        "-fopenmp",
    ],
    "extra_compile_args": [
        "-fopenmp",
        "-Wno-strict-prototypes",
        "-Wno-maybe-uninitialized",
        "-O3",
        "-std=c99",
    ],
}
OSX_OPTS = {
    "extra_link_args": [
        "-L/usr/local/opt/libomp/lib",
        "-fopenmp",
    ],
    "extra_compile_args": [
        "-Wno-strict-prototypes",
        "-Wno-uninitialized",
        "-fopenmp",
        "-O3",
    ],
}
WINDOWS_OPTS = {
    "extra_compile_args": [
        "/Ox",
    ]
}


if os.name == "posix":
    if sys.platform.startswith("darwin"):
        OPTS = OSX_OPTS
    else:
        OPTS = LINUX_OPTS
else:
    OPTS = WINDOWS_OPTS


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": [
                Extension(
                    "gamred_native",
                    sources=glob("gamred_native/*.c"),
                    include_dirs=["gamred_native", numpy.get_include()],
                    define_macros=[
                        ("NPY_NO_DEPRECATED_API", None),
                        ("NPY_1_7_API_VERSION", None),
                    ],
                    **OPTS,
                ),
            ],
        }
    )
