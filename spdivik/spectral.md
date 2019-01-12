# `spectral`

This package is responsible for spectral clustering of data.

## Features

`spectral` tool allows for:

1) Segmentation of dataset into clusters.
2) Determination of optimal number of clusters.
3) Visualization of optimal segmentation.

## Parameters

All the parameters can be checked via `spectral --help`:

```
usage: spectral [-h] --source SOURCE --destination DESTINATION --config CONFIG
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
  "distance": "euclidean",
  "n_components": 2,
  "random_state": 0,
  "eigen_solver": null,
  "n_neighbors": 7,
  "n_jobs": -1
}
```

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

#### `n_components`

The dimension of the projected subspace.

#### `random_state`

Random seed for generating uniform data sets. Defaults to `0`.

#### `eigen_solver`

All supported values:

- `null`
- `arpack`
- `lobpcg`
- `amg`

The eigenvalue decomposition strategy to use. AMG requires `pyamg`
to be installed. It can be faster on very large, sparse problems,
but may also lead to instabilities.

#### `n_neighbors`

Number of neighbors to use when constructing the locally adjusted
affinity matrix using RBF kernel.

#### `n_jobs`

The number of jobs to use for the computation. This works by computing
each of the clustering & scoring runs in parallel. Default `1`.

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
spectral \
    --source my/data/path.npy \
    --config spectral.json \
    --destination results \
    --xy my/data/xy.npy
```

### Docker

To download the latest version of `spectral` software run:

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
    spectral \
    --source my/data/path.npy \
    --config spectral.json \
    --destination results \
    --xy my/data/xy.npy
```

and for Windows:

```cmd
docker run^
    --volume %cd%:/data^
    gmrukwa/divik^
    spectral^
    --source my/data/path.npy^
    --config spectral.json^
    --destination results^
    --xy my/data/xy.npy
```
