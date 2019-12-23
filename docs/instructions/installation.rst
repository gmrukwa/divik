Installation
============

Docker
------

The recommended way to use this software is through
`Docker <https://www.docker.com/>`_. This is the most convenient way, if you
want to use `divik` application, since it requires *MATLAB Compiler Runtime*
and more dependencies.

To install latest stable version use::

    docker pull gmrukwa/divik

To install specific version, you can specify it in the command, e.g.::

    docker pull gmrukwa/divik:2.3.6

Python package
--------------

Prerequisites for installation of base package:

- Python 3.5

These are required for using `divik` application and GMM-based filtering:

- `MATLAB Compiler Runtime <https://www.mathworks.com/products/compiler/matlab-runtime.html>`_, version 2016b or newer, installed to default path
- `compiled package with legacy code <https://github.com/spectre-team/matlab-legacy/releases/tag/legacy-v4.0.9>`_

Installation process may be clearer with insight into Docker images used for
application deployment:

- `python_mcr image <https://github.com/spectre-team/python_mcr>`_ - installs MCR r2016b onto Python 3.5 image
- `python_msi image <https://github.com/spectre-team/python_msi>`_ - installs compiled legacy code onto MCR image
- `divik image <https://github.com/spectre-team/spectre-divik/blob/master/dockerfile>`_ - installs DiviK software onto legacy code image

Having prerequisites installed, one can install latest base version of the
package::

    pip install divik

or any stable tagged version, e.g.::

    pip install divik==2.3.6
