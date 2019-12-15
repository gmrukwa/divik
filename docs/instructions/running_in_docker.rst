Running in Docker
=================

Prerequisites
-------------

First of all, you need to have Docker installed. You can proceed with the
official instructions:

- `Windows <https://docs.docker.com/docker-for-windows/install/>`_
- `Ubuntu <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`_
- `Mac <https://docs.docker.com/docker-for-mac/install/>`_

Under Windows and Mac you need to perform additional configuration steps before
running the analysis, since data processing requires additional resources as
compared to simple web applications.

1. Right-click the running Docker icon (a whale with squares).
2. Go to *Preferences*
3. Allow Docker to run with all the CPUs and reasonable RAM (at least 16 GB, as much as possible recommended).

.. note:: Under Ubuntu these steps are not required as Docker runs natively.

Run the Container
-----------------

The container is launched with the default Docker syntax, as described
`here <https://docs.docker.com/engine/reference/run/>`_.
You can use the following:

- under UNIX::

    docker run \
        --rm -it \
        --volume $(pwd):/data \
        gmrukwa/divik \
        bash

- under Windows::

    docker run^
        --rm -it^
        --volume %cd%:/data^
        gmrukwa/divik^
        bash

In both cases, the directory where the command is ran is mounted to the
``\data`` directory in the container, so the data and / or configuration is
available (see `Data`_). ``--rm`` indicates that the container gets removed
after it finishes running. ``-it`` indicates that the console will get attached
to the running container. ``gmrukwa/divik`` is the image name. Finally,
``bash`` launches the shell in the container. You can launch any other
command there.

Code
----

Code of the installed package is available at the `/app` directory
in the case of need to reinstall.

Data
----

Your data should be mounted into the container in the ``/data`` directory.
It is assumed to be the working directory of the Python interpreter.
Please remember that all the paths should be relative to this directory
or absolute with root at ``/data``. This is maintained by the switch
``-v $(pwd):/data`` under UNIX or ``-v %cd%:/data`` under Windows.

I/O Buffering
-------------

Python interpreter I/O buffering is turned off by default, so all the
outputs appear immediately. Otherwise it would be impossible to track the
actual progress of the computations. You can turn this off by setting
``PYTHONUNBUFFERED`` environment variable to ``FALSE``.
