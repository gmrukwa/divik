[![CodeFactor](https://www.codefactor.io/repository/github/gmrukwa/divik/badge)](https://www.codefactor.io/repository/github/gmrukwa/divik)
[![BCH compliance](https://bettercodehub.com/edge/badge/gmrukwa/divik?branch=master)](https://bettercodehub.com/)
[![Maintainability](https://api.codeclimate.com/v1/badges/4cf5d42d0a0076c38445/maintainability)](https://codeclimate.com/github/gmrukwa/divik/maintainability)
![](https://github.com/gmrukwa/divik/workflows/Build%20and%20push%20deployment%20images/badge.svg)
![](https://github.com/gmrukwa/divik/workflows/Run%20unit%20tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/divik/badge/?version=latest)](https://divik.readthedocs.io/en/latest/?badge=latest)

# divik

Python implementation of Divisive iK-means (DiviK) algorithm.

# Tools within this package

> This section will be further developed soon.

1) [`divik`](divik/_cli/divik.md) - runs DiviK in GAP-only scenario
2) [`dunn-divik`](dunn-divik/_cli/dunn_divik.md) - runs DiviK in GAP & Dunn scenario
2) [`kmeans`](divik/_cli/auto_kmeans.md) - runs K-means with GAP statistic
3) `linkage` - runs agglomerative clustering
4) [`inspect`](divik/_cli/inspect.md) - visualizes DiviK result
5) `visualize` - generates `.png` file with visualization of clusters for 2D
maps
6) [`spectral`](divik/_cli/spectral.md) - generates spectral embedding of a
dataset

# Installation

## Docker

The recommended way to use this software is through
[Docker](https://www.docker.com/). This is the most convenient way, if you want
to use `divik` application.

To install latest stable version use:

```bash
docker pull gmrukwa/divik
```

To install specific version, you can specify it in the command, e.g.:

```bash
docker pull gmrukwa/divik:2.5.12
```

## Python package

Prerequisites for installation of base package:

- Python 3.6 / 3.7 / 3.8
- compiler capable of compiling the native C code and OpenMP support

#### Installation of OpenMP for Ubuntu / Debian

You should have it already installed with GCC compiler, but if somehow
not, try the following:

```bash
sudo apt-get install libgomp1
```

#### Installation of OpenMP for Mac

OpenMP is available as part of LLVM. You may need to install in with:

```bash
brew install llvm libomp
```

#### DiviK Installation

Having prerequisites installed, one can install latest base version of the
package:

```bash
pip install divik
```

or any stable tagged version, e.g.:

```bash
pip install divik==2.5.12
```

If you want to have compatibility with
[`gin-config`](https://github.com/google/gin-config), you can install
necessary extras with:

```bash
pip install divik[gin]
```

**Note:** Remember about `\` before `[` and `]` in `zsh` shell.

If you want to launch `inspect` tool, you need to install extras with:

```bash
pip install divik[inspect]
```

You can install all extras with:

```bash
pip install divik[all]
```

# High-Volume Data Considerations

If you are using DiviK to run the analysis that could fail to fit RAM of your
computer, consider disabling the default parallelism and switch to
[dask](https://dask.org/). It's easy to achieve through configuration:

- set all parameters named `n_jobs` to `1`;
- set all parameters named `allow_dask` to `True`.

Never set `n_jobs>1` and `allow_dask=True` at the same time, the computations
will freeze due to how `multiprocessing` and `dask` handle parallelism.

# References

This software is part of contribution made by [Data Mining Group of Silesian
University of Technology](http://www.zaed.polsl.pl/), rest of which is
published [here](https://github.com/ZAEDPolSl).

+ [Mrukwa, G. and Polanska, J., 2020. DiviK: Divisive intelligent K-means for
hands-free unsupervised clustering in biological big data. *arXiv preprint
arXiv:2009.10706.*][1]

[1]: https://arxiv.org/abs/2009.10706
