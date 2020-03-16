Installation
============

Docker
------

The recommended way to use this software is through
`Docker <https://www.docker.com/>`_. This is the most convenient way, if you
want to use `divik` application.

To install latest stable version use::

    docker pull gmrukwa/divik

To install specific version, you can specify it in the command, e.g.::

    docker pull gmrukwa/divik:2.5.0

Python package
--------------

Prerequisites for installation of base package:

- Python 3.6 / 3.7 / 3.8
- compiler capable of compiling the native C code

Having prerequisites installed, one can install latest base version of the
package::

    pip install divik

or any stable tagged version, e.g.::

    pip install divik==2.5.0

If you want to have compatibility with
`gin-config <https://github.com/google/gin-config>`_, you can install
necessary extras with::

    pip install divik[gin]

.. note:: Remember about `\` before `[` and `]` in `zsh` shell.
