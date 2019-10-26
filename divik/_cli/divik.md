# `divik`

This package is responsible for DiviK clustering of data.

## Features

`DiviK` tool allows for:

1) Segmenting dataset into clusters
2) Adaptive filtering of features, dependent on local characteristics
3) Automated analysis stop, adaptive to data characteristic
4) Use of any metric specified in `scipy`

## Parameters

All the parameters can be checked via `divik --help`:

```
usage: divik [-h] --source SOURCE --destination DESTINATION --config CONFIG
             [--xy XY] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE, -s SOURCE
                        Source file with data. Should contain observations in
                        rows and features in columns.
  --destination DESTINATION, -d DESTINATION
                        Destination directory into which results will be
                        saved.
  --config CONFIG, -c CONFIG
                        Configuration file for the experiment.
  --xy XY               File with spatial coordinates X and Y. Should contain
                        X in first column, Y in second. The number of rows
                        should be equal to the data.
  --verbose, -v
```

### `source`

Path to the embedded points.

This parameter supports:

- `.csv` files without header
- `.npy` files
- single-variable `.mat`-files

### `destination`

Path to the directory in which result will be generated.

### `config`

Path to the JSON [configuration file](#configuration-file).

### `xy`

Path to the spatial coordinates of clustered points.

This parameter supports:

- `.csv` files without header
- `.npy` files
- single-variable `.mat`-files

There are two columns expected: first with X coordinate and second with Y coordinate of each clustered point. Note: it is crucial to preserve the same order as in the clustered points.

### `verbose`

If specified, collects more comprehensive logs.

## Configuration file

### Structure

Configuration file should be a JSON file as follows:

```json
{
  "gap_trials": 10,
  "distance_percentile": 99.0,
  "max_iter": 100,
  "distance": "correlation",
  "minimal_size": 16,
  "rejection_size": 2,
  "minimal_features_percentage": 0.01,
  "fast_kmeans_iters": 10,
  "k_max": 10,
  "normalize_rows": true,
  "use_logfilters": true,
  "n_jobs": -1,
  "random_seed": 0,
  "verbose": true
}

```

#### `gap_trials`

Number of times that reference dataset will be drawn from uniform distribution
for computation of GAP index. Default `10`.

#### `distance_percentile`

Distance percentile, at which algorithm should look for initial cluster centers.
Should be between `0.0` and `100.0`.

#### `max_iter`

Limit of number of iterations in k-means algorithm. Default `100`. Rather should
not be less.

#### `distance`

Distance measure, defaults to `euclidean`. For Mass Spectrometry Imaging
purposes `correlation` metric may be more useful. These are the distances
supported by `scipy` package. All supported values:

- `braycurtis`
- `canberra`
- `chebyshev`
- `cityblock`
- `correlation`
- `cosine`
- `dice`
- `euclidean`
- `hamming`
- `jaccard`
- `kulsinski`
- `mahalanobis`
- `atching`
- `minkowski`
- `rogerstanimoto`
- `russellrao`
- `sokalmichener`
- `sokalsneath`
- `sqeuclidean`
- `yule`

#### `minimal_size`

Size of cluster that will not be further considered for clustering. Could be
adjusted to `0.1%` of the dataset, but it really depends on the goal you want to
achieve. Sometimes it just makes no sense to segment artificial clusters. On
the other hand, if actual structures may be smaller, it could be selected
smaller.

#### `rejection_size`

If cluster of size less or equal to this number appears, such segmentation will
be rejected. Default `2`. To disable this mechanism, just set it to `0`.

#### `minimal_features_percentage`

Percent of features that are enforced to be preserved after each filtering.
Default `0.01` (corresponding to `1%`).

#### `fast_kmeans_iters`

Number of k-means iterations performed during GAP trial. Default `10`. In most
cases this is sufficient.

#### `k_max`

Maximal number of clusters. Default `10`, since Dunn's index for selection of
optimal number of clusters favorizes low number of clusters. If there is a
suspicion that it is not enough, may be increased, but will slow down
computations. 

#### `normalize_rows`

Specifies, if rows should be centered and their norm should be set to `1.0`.
This should be `true` for `correlation` metric, while for others in most cases
should be `false`.

#### `use_logfilters`

Whether to compute logarithm of feature characteristic instead of the
characteristic itself. This may improve feature filtering performance, depending
on the distribution of features, however all the characteristics (mean,
variance) have to be positive for that - filtering will fail otherwise. This is
useful for specific cases in biology where the distribution of data may actually
require this option for any efficient filtering.

#### `n_jobs`

The number of jobs to use for the computation. This works by computing each of
the GAP index evaluations in parallel and by making predictions in parallel.

#### `random_seed`

Seed to initialize the random number generator.

#### `verbose`

Whether to report the progress of the computations.

## How to run?

There are two ways to launch the segmentation: via installed package or inside
Docker container.

### Installed package

Package in the latest stable version could be installed via:

```bash
pip install divik
```

Then, the analysis is launched via:

```bash
divik \
    --source my/data/path.csv \
    --config divik.json \
    --destination results
```

### Docker

To download the latest version of `divik` software run:

```bash
docker pull gmrukwa/divik
```

Launching analysis inside Docker container requires additional switch:

- `--volume $(pwd):/data` (UNIX) / `--volume %cd%:/data` (Windows) - to mount
the current directory into a container's working directory. If you want to mount
any other directory, just replace `$(pwd)` or `%cd%` with the directory of
choice.

Full command to launch the analysis inside Docker (for UNIX) looks like:

```bash
docker run \
    --volume $(pwd):/data \
    gmrukwa/divik \
    divik \
    --source my/data/path.csv \
    --config divik.json \
    --destination results
```

and for Windows:

```cmd
docker run^
    --volume %cd%:/data^
    gmrukwa/divik^
    divik^
    --source my/data/path.csv^
    --config divik.json^
    --destination results
```

**Note:** Path to data and config should be relative with respect to current
directory.