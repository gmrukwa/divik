# `inspect`

This package is responsible for visualization of DiviK algorithm results.

## Features

`inspect` tool allows for:

1) Visualization of DiviK clustering result at different levels.
2) Disabling clusters from in-depth visualization.
3) Re-coloring the clusters for improved color perception.
4) Exporting visualized clusters to `.png`.
5) Saving and loading profiles of interactive analysis for resuming later on.

## Parameters

All the parameters can be checked via `inspect --help`:

```
usage: inspect [-h] --result RESULT --xy XY [--host HOST] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --result RESULT, -r RESULT
                        divik result directory
  --xy XY               coordinates of points
  --host HOST           Sets up a host interface to run the visualization on.
                        Use 0.0.0.0 while in Docker.
  --debug               enables debug mode
```

### `result`

Path to the DiviK result directory. This is the directory containing files like
`result.pkl`, `final_partition.npy`, `centroids.npy`, etc.

### `xy`

Path to the spatial coordinates of clustered points.

This parameter supports:

- `.csv` files without header
- `.npy` files
- single-variable `.mat`-files

There are two columns expected: first with X coordinate and second with Y
coordinate of each clustered point. **Note:** it is crucial to preserve the same
order as in the clustered points.

### `host` (optional)

Address of the interface on which visualization will be ran. Defaults to
`127.0.0.1` if not set. When using inside Docker, should be set to `0.0.0.0`.

### `debug` (optional)

If set, launches the application in debug mode.

## How to run?

There are two ways to launch the visualization: via installed package or inside
Docker container.

### Installed package

Package in the latest version could be installed via:

```bash
pip install git+https://github.com/spectre-team/spectre-divik.git@master#egg=spectre-divik
```

Then, the visualization is launched via:

```bash
inspect --result results/my-divik-analysis/20181204-063622 --xy xy.npy
```

Finally, you can visit [`http://127.0.0.1:8050`](http://127.0.0.1:8050) and view
the obtained clusters.

To turn off visualization after the analysis, just use combination of `Ctrl+C`
inside the console.

### Docker

To download the latest version of `inspect` software run:

```bash
docker pull gmrukwa/divik
```

Launching visualization inside Docker container requires additional switches:

- `--volume $(pwd):/data` (UNIX) / `--volume %cd%:/data` (Windows) - to mount
the current directory into a container's working directory. If you want to mount
any other directory, just replace `$(pwd)` or `%cd%` with the directory of
choice.
- `--port 8050:8050` - binds port `8050` from the container to the host's port
`8050`. If you wish to bind the visualization to different port, you can use
something like `--port 1234:8050` to have the application at the port `1234`.
It will be accessible under [`http://127.0.0.1:1234`](http://127.0.0.1:1234).
This is useful is you want to run several visualizations at the same time, since
each one requires separate port.

Full command to launch the visualization inside Docker (for UNIX) looks like:

```bash
docker run \
    --volume $(pwd):/data \
    -p 8050:8050 \
    gmrukwa/divik \
    inspect \
    --result results/my-divik-analysis/20181204-063622 \
    --xy xy.npy
```

and for Windows:

```cmd
docker run^
    --volume %cd%:/data^
    -p 8050:8050^
    gmrukwa/divik^
    inspect^
    --result results/my-divik-analysis/20181204-063622^
    --xy xy.npy
```

You can find it running at [`http://127.0.0.1:8050`](http://127.0.0.1:8050).

To turn off visualization after the analysis, just use combination of `Ctrl+C`
inside the console.
