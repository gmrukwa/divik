Cluster analysis with ``fit-clusters`` executable
=================================================

.. note::
    ``fit-clusters`` requires installation with ``gin`` extras, e.g.
    ``pip install divik[gin]``

`fit-clusters` is just one CLI executable that allows you to run DiviK
algorithm, any other clustering algorithms supported by `scikit-learn`
or even a pipeline with pre-processing.

Usage
-----

CLI interface
^^^^^^^^^^^^^

There are two types of parameters:

1.  ``--param`` - this way you can set the value of a parameter during
    `fit-clusters` executable launch, i.e. you can overwrite parameter provided
    in a config file or a default.

2.  ``--config`` - this way you can provide a list of config files. Their
    content will be treated as a one big (ordered) list of settings. In case of
    conflict, the later file overwrites a setting provided by earlier one.

These go directly to the CLI.

.. code-block:: bash

    usage: fit-clusters [-h] [--param [PARAM [PARAM ...]]]
                    [--config [CONFIG [CONFIG ...]]]

    optional arguments:
    -h, --help            show this help message and exit
    --param [PARAM [PARAM ...]]
                            List of Gin parameter bindings
    --config [CONFIG [CONFIG ...]]
                            List of paths to the config files

Sample ``fit-clusters`` call:

.. code-block:: bash

    fit-clusters \
      --param \
        load_data.path='/data/my_data.csv' \
        DiviK.distance='euclidean' \
        DiviK.use_logfilters=False \
        DiviK.n_jobs=-1 \
      --config \
        my-defaults.gin \
        my-overrides.gin

The elaboration of all the parameters is included in `Experiment configuration`_
and `Model setup`_.

Experiment configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Following parameters are available when launching experiments:

#.  ``load_data.path`` - path to the file with data for clustering. Observations
    in rows, features in columns.

#.  ``load_xy.path`` - path to the file with X and Y coordinates for the
    observations. The number of coordinate pairs must be the same as the number
    of observations. Only integer coordinates are supported now.

#. ``experiment.model`` - the clustering model to fit to the data. See more in
   `Model setup`_.

#.  ``experiment.steps_that_require_xy`` - when using scikit-learn Pipeline,
    it may be required to provide spatial coordinates to fit specific algorithms.
    This parameter accepts the list of the steps that should be provided with
    spatial coordinates during pipeline execution (e.g. ``EximsSelector``).

#.  ``experiment.destination`` - the destination directory for the experiment
    outputs. Default ``result``.

#.  ``experiment.omit_datetime`` - if ``True``, the destination directory will be
    directly populated with the results of the experiment. Otherwise, a
    subdirectory with date and time will be created to keep separation between
    runs. Default ``False``.

#.  ``experiment.verbose`` - if ``True``, extends the messaging on the console.
    Default `False`.

#.  ``experiment.exist_ok`` - if ``True``, the experiment will not fail if the
    destination directory exists. This is to avoid results overwrites. Default
    ``False``.


Model setup
-----------

``divik`` models
^^^^^^^^^^^^^^^^

To use DiviK algorithm in the experiment, a config file must:

#.  Import the algorithms to the scope, e.g.::
    
        import divik.cluster

#.  Point experiment which algorithm to use, e.g.::

        experiment.model = @DiviK()

#.  Configure the algorithm, e.g.::

        DiviK.distance = 'euclidean'
        DiviK.verbose = True

Sample config with ``KMeans``
*****************************

Below you can check sample configuration file, that sets up simple KMeans::

    import divik.cluster

    KMeans.n_clusters = 3
    KMeans.distance = "correlation"
    KMeans.init = "kdtree_percentile"
    KMeans.leaf_size = 0.01
    KMeans.percentile = 99.0
    KMeans.max_iter = 100
    KMeans.normalize_rows = True

    experiment.model = @KMeans()
    experiment.omit_datetime = True
    experiment.verbose = True
    experiment.exist_ok = True


Sample config with ``DiviK``
****************************

Below is the configuration file with full setup of DiviK. ``DiviK`` requires
an automated clustering method for stop condition and a separate one for
clustering. Here we use ``GAPSearch`` for stop condition and ``DunnSearch``
for selecting the number of clusters. These in turn require a ``KMeans``
method set for a specific distance method, etc.::

    import divik.cluster

    KMeans.n_clusters = 1
    KMeans.distance = "correlation"
    KMeans.init = "kdtree_percentile"
    KMeans.leaf_size = 0.01
    KMeans.percentile = 99.0
    KMeans.max_iter = 100
    KMeans.normalize_rows = True

    GAPSearch.kmeans = @KMeans()
    GAPSearch.max_clusters = 2
    GAPSearch.n_jobs = 1
    GAPSearch.seed = 42
    GAPSearch.n_trials = 10
    GAPSearch.sample_size = 1000
    GAPSearch.drop_unfit = True
    GAPSearch.verbose = True

    DunnSearch.kmeans = @KMeans()
    DunnSearch.max_clusters = 10
    DunnSearch.method = "auto"
    DunnSearch.inter = "closest"
    DunnSearch.intra = "furthest"
    DunnSearch.sample_size = 1000
    DunnSearch.seed = 42
    DunnSearch.n_jobs = 1
    DunnSearch.drop_unfit = True
    DunnSearch.verbose = True

    DiviK.kmeans = @DunnSearch()
    DiviK.fast_kmeans = @GAPSearch()
    DiviK.distance = "correlation"
    DiviK.minimal_size = 200
    DiviK.rejection_size = 2
    DiviK.minimal_features_percentage = 0.005
    DiviK.features_percentage = 1.0
    DiviK.normalize_rows = True
    DiviK.use_logfilters = True
    DiviK.filter_type = "gmm"
    DiviK.n_jobs = 1
    DiviK.verbose = True

    experiment.model = @DiviK()
    experiment.omit_datetime = True
    experiment.verbose = True
    experiment.exist_ok = True


``scikit-learn`` models
^^^^^^^^^^^^^^^^^^^^^^^

For a model to be used with ``fit-clusters``, it needs to be marked as
``gin.configurable``. While it is true for DiviK and remaining algorithms
within ``divik`` package, ``scikit-learn`` requires additional setup.

#.  Import helper module::

        import divik.core.gin_sklearn_configurables

#.  Point experiment which algorithm to use, e.g.::

        experiment.model = @MeanShift()

#.  Configure the algorithm, e.g.::

        MeanShift.n_jobs = -1
        MeanShift.max_iter = 300

.. warning::
    Importing both ``scikit-learn`` and ``divik`` will result in an ambiguity
    when using e.g. ``KMeans``. In such a case it is necesary to point specific
    algorithms by a full name, e.g. ``divik.cluster._kmeans._core.KMeans``.

Sample config with ``MeanShift``
********************************

Below you can check sample configuration file, that sets up simple MeanShift::

    import divik.core.gin_sklearn_configurables

    MeanShift.cluster_all = True
    MeanShift.n_jobs = -1
    MeanShift.max_iter = 300

    experiment.model = @MeanShift()
    experiment.omit_datetime = True
    experiment.verbose = True
    experiment.exist_ok = True


Pipelines
^^^^^^^^^

``scikit-learn`` Pipelines have a separate section to provide an additional
explanation, even though these are part of ``scikit-learn``.

#.  Import helper module::

        import divik.core.gin_sklearn_configurables

#.  Import the algorithms into the scope::

        import divik.feature_extraction

#.  Point experiment which algorithm to use, e.g.::

        experiment.model = @Pipeline()

#.  Configure the algorithms, e.g.::

        MeanShift.n_jobs = -1
        MeanShift.max_iter = 300

#.  Configure the pipeline::

        Pipeline.steps = [
            ('histogram_equalization', @HistogramEqualization()),
            ('exims', @EximsSelector()),
            ('pca', @KneePCA()),
            ('mean_shift', @MeanShift()),
        ]

#.  (If needed) configure steps that require spatial coordinates::

        experiment.steps_that_require_xy = ['exims']


Sample config with ``Pipeline``
*******************************

Below you can check sample configuration file, that sets up simple Pipeline::

    import divik.core.gin_sklearn_configurables
    import divik.feature_extraction

    MeanShift.n_jobs = -1
    MeanShift.max_iter = 300

    Pipeline.steps = [
        ('histogram_equalization', @HistogramEqualization()),
        ('exims', @EximsSelector()),
        ('pca', @KneePCA()),
        ('mean_shift', @MeanShift()),
    ]

    experiment.model = @Pipeline()
    experiment.steps_that_require_xy = ['exims']
    experiment.omit_datetime = True
    experiment.verbose = True
    experiment.exist_ok = True


Custom models
^^^^^^^^^^^^^

The ``fit-clusters`` executable can work with custom algorithms as well.

#.  Mark an algorithm class ``gin.configurable`` at the definition time::

        import gin

        @gin.configurable
        class MyClustering:
            pass
    
    or when importing them from a library::

        import gin

        gin.external_configurable(MyClustering)


#.  Define artifacts saving methods::

        from divik.core.io import saver

        @saver
        def save_my_clustering(model, fname_fn, **kwargs):
            if not hasattr(model, 'my_custom_field_'):
                return
            # custom saving logic comes here

    There are some default savers defined, which are compatible with
    lots of ``divik`` and ``scikit-learn`` algorithms, supporting things like:

    - model pickling
    - JSON summary saving
    - labels saving (``.npy``, ``.csv``)
    - centroids saving (``.npy``, ``.csv``)
    - pipeline saving

    A ``saver`` should be highly reusable and could be a pleasant contribution
    to the ``divik`` library.

#.  In config, import the module which marks your algorithm configurable::

        import myclustering

#.  Continue with the algorithm setup and plumbing as in the previous scenarios
