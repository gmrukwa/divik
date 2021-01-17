.. divik documentation master file, created by
   sphinx-quickstart on Sat Oct 26 12:33:52 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to divik's documentation!
=================================

Here you can find a list of documentation topics covered by this page.


.. toctree::
   :maxdepth: 4
   :caption: Command Line Interface

   cli/fit-clusters


Computational Modules
---------------------

.. toctree::
   :maxdepth: 4
   :caption: Computational Modules
   :hidden:
   
   modules/divik.cluster
   modules/divik.feature_extraction
   modules/divik.feature_selection
   modules/divik.sampler

.. autosummary::
   :recursive:

   divik.cluster
   divik.feature_extraction
   divik.feature_selection
   divik.sampler


Utility Packages
----------------

.. toctree::
   :maxdepth: 4
   :caption: Utility Packages
   :hidden:

   modules/divik
   modules/divik.core
   modules/divik.core.io
   modules/divik.core.gin_sklearn_configurables

.. autosummary::
   :recursive:

   divik
   divik.core
   divik.core.io
   divik.core.gin_sklearn_configurables


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
