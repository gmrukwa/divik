# `kmeans`

This package is responsible for k-means segmentation of data.

## Features

`kmeans` tool allows for:

1) Segmentation of dataset into clusters.
2) Quality scoring of clusters obtained.
3) Determination of optimal set of clusters.
4) Visualization of optimal segmentation.

## Parameters

All the parameters can be checked via `kmeans --help`:

```
usage: kmeans [-h] --source SOURCE --destination DESTINATION --config CONFIG
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

Path to the clustered points.

This parameter supports:

- `.csv` files without header
- `.npy` files
- single-variable `.mat`-files
- path to the variable within Quilt dataset (requires installation with
optional `spdivik[quilt_packages]`)

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
- path to the variable within Quilt dataset (requires installation with
optional `spdivik[quilt_packages]`)

There are two columns expected: first with X coordinate and second with Y
coordinate of each clustered point. **Note:** it is crucial to preserve the same
order as in the clustered points.

### `verbose`

If specified, collects more comprehensive logs.

## Configuration file

### Structure

Configuration file should be a JSON file as follows:

```json
{
  "max_clusters": 50,
  "min_clusters": 2,
  "distance": "correlation",
  "n_jobs": -1,
  "method": "dunn",
  "init": "percentile",
  "percentile": 95.0,
  "max_iter": 100,
  "normalize_rows": true,
  "gap": {
      "correction": true,
      "max_iter": 10,
      "seed": 0,
      "trials": 10
  },
  "verbose":true
}
```

Each parameter is explained below. All parameters but `max_clusters` are
optional, however specifying them all to default values may improve
reproducibility of the experiment.

### `max_clusters`

Maximal number of clusters checked in the scoring procedure.

### `min_clusters`

Minimal number of clusters checked in the scoring procedure. Default `1`.

### `distance`

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

### `n_jobs`

The number of jobs to use for the computation. This works by computing
each of the clustering & scoring runs in parallel. Default `1`.

### `method`

The method to select the best number of clusters. Defaults to `dunn`.
 
1. `dunn` - computes score that relates dispersion inside a cluster
to distances between clusters. Never selects 1 cluster.
2. `gap` - compares dispersion of a clustering to a dispersion in
grouping of a reference uniformly distributed dataset

### `init`

Method for initialization, defaults to `percentile`.

1. `percentile` - selects initial cluster centers for k-mean clustering
starting from specified percentile of distance to already selected clusters.
2. `extreme` - selects initial cluster centers for k-mean clustering starting
from the furthest points to already specified clusters.

### `percentile`

Specifies the starting percentile for `percentile` initialization.
Defaults to `95.0`. Must be within range [0.0, 100.0]. At 100.0 it is
equivalent to `extreme` initialization.

### `max_iter`

Maximum number of iterations of the k-means algorithm for a single run.
Defaults to `100`.

### `normalize_rows`

If `true`, rows are translated to mean of 0.0 and scaled to norm of 1.0.
Defaults to `false`.

### `gap`

Configuration of GAP statistic in a form of a dictionary.

#### `correction`

If `true`, the correction is applied and the first feasible solution
is selected. Otherwise the globally maximal GAP is used. Defaults to `true`.

#### `max_iter`

Maximal number of iterations KMeans will do for computing statistic.
Defaults to `10`.

#### `seed`

Random seed for generating uniform data sets. Defaults to `0`.

#### `trials`

Number of data sets drawn as a reference. Defaults to `10`.

### `verbose`

If `true`, shows progress bar. Defaults to `false`.

## How to run?

There are two ways to launch the segmentation: via installed package or inside
Docker container.

### Installed package

Package in the latest version could be installed via:

```bash
pip install git+https://github.com/spectre-team/spectre-divik.git@master#egg=spectre-divik
```

Then, the analysis is launched via:

```bash
kmeans \
    --source my/data/path.npy \
    --config kmeans.json \
    --destination results \
    --xy my/data/xy.npy
```

### Docker

To download the latest version of `kmeans` software run:

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
    kmeans \
    --source my/data/path.npy \
    --config kmeans.json \
    --destination results \
    --xy my/data/xy.npy
```

and for Windows:

```cmd
docker run^
    --volume %cd%:/data^
    gmrukwa/divik^
    kmeans^
    --source my/data/path.npy^
    --config kmeans.json^
    --destination results^
    --xy my/data/xy.npy
```
