import json
import logging
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from divik.core import configurable, visualize

_SAVERS = set()


def saver(fn):
    """Register the function as handler for saving model and related summaries

    The saver function should be reusable for different models exhibiting the
    required variables. Rather prefer checking the required attributes than the
    model class.

    Examples
    --------

    >>> from divik.core.io import saver
    >>> @saver
    ... def my_saver(model, destination, **kwargs):
    ...     if not hasattr(model, 'my_custom_field_'):
    ...         return
    ...     if not 'my_param' in kwargs:
    ...         return
    ...     # custom saving logic comes here

    You can also make this function configurable:

    >>> import gin
    >>> from divik.core.io import saver
    >>> @saver
    ... @gin.configurable(allowlist=['my_param'])
    ... def configurable_saver(model, destination, my_param=None, **kwargs):
    ...     if not hasattr(model, 'my_custom_field_'):
    ...         return
    ...     if my_param is None:
    ...         return
    ...     # custom saving logic comes here
    """
    _SAVERS.add(fn)


def save(model, destination, **kwargs):
    """Save model and related summaries into specified destination directory"""
    fname_fn = partial(os.path.join, destination)
    for save_fn in _SAVERS:
        save_fn(model, fname_fn, **kwargs)


@saver
@configurable(allowlist=["enabled"])
def save_pickle(model, fname_fn, enabled=True, **kwargs):
    if not enabled:
        return
    logging.info("Saving model pickle.")
    with open(fname_fn("model.pkl"), "wb") as pkl:
        pickle.dump(model, pkl)


@saver
def save_summary(model, fname_fn, **kwargs):
    if not hasattr(model, "labels_"):
        return
    logging.info("Saving JSON summary.")
    n_clusters = getattr(model, "n_clusters_", np.unique(model.labels_).size)
    with open(fname_fn("summary.json"), "w") as smr:
        json.dump(
            {
                "depth": getattr(model, "depth_", 1),
                "number_of_clusters": n_clusters,
                "mean_cluster_size": model.labels_.size / float(n_clusters),
            },
            smr,
        )


@saver
def save_labels(model, fname_fn, **kwargs):
    if not hasattr(model, "labels_"):
        return
    logging.info("Saving final partition.")
    np.save(fname_fn("final_partition.npy"), model.labels_)
    np.savetxt(fname_fn("final_partition.csv"), model.labels_, delimiter=", ", fmt="%i")
    if "xy" in kwargs:
        import skimage.io

        visualization = visualize(model.labels_, xy=kwargs["xy"])
        skimage.io.imsave(fname_fn("final_partition.png"), visualization)


@saver
def save_multiple_labels(model, fname_fn, **kwargs):
    if not hasattr(model, "estimators_") or not hasattr(model.estimators[0], "labels_"):
        return
    logging.info("Saving all considered partitions.")
    part = np.hstack([e.labels_.reshape(-1, 1) for e in model.estimators_])
    np.save(fname_fn("partitions.npy"), part)
    np.savetxt(fname_fn("partitions.csv"), part, delimiter=", ", fmt="%i")

    import skimage.io

    for i in range(part.shape[1]):
        np.savetxt(
            fname_fn("partitions.{0}.csv").format(i),
            part[:, i].reshape(-1, 1),
            delimiter=", ",
            fmt="%i",
        )
        if "xy" in kwargs:
            visualization = visualize(part, xy=kwargs["xy"])
            skimage.io.imsave(fname_fn("partitions.{0}.png").format(i), visualization)


@saver
def save_centroids(model, fname_fn, **kwargs):
    if not hasattr(model, "centroids_"):
        return
    logging.info("Saving centroids.")
    np.save(fname_fn("centroids.npy"), model.centroids_)
    np.savetxt(fname_fn("centroids.csv"), model.centroids_, delimiter=", ")


@saver
def save_filters(model, fname_fn, **kwargs):
    if not hasattr(model, "filters_"):
        return
    logging.info("Saving filters.")
    np.save(fname_fn("filters.npy"), model.filters_)
    np.savetxt(fname_fn("filters.csv"), model.filters_, delimiter=", ", fmt="%i")


@saver
def save_cluster_paths(model, fname_fn, **kwargs):
    if not hasattr(model, "reverse_paths_"):
        return
    rev = ["_".join(map(str, p)) for p in model.reverse_paths_]
    pd.DataFrame(
        {"path": rev, "cluster_number": list(model.reverse_paths_.values())}
    ).to_csv(fname_fn("paths.csv"))


@saver
def save_pipeline(model, fname_fn, **kwargs):
    if not isinstance(model, Pipeline):
        return
    feature_selector = model[:-1]
    clustering = model[-1]
    if isinstance(clustering, Pipeline):
        logging.info("Saving pre-extractor pickle.")
        with open(fname_fn("feature_pre_extractor.pkl"), "wb") as pkl:
            pickle.dump(feature_selector, pkl)
        return save(clustering, fname_fn, **kwargs)
    logging.info("Saving model pickle.")
    with open(fname_fn("feature_selector.pkl"), "wb") as pkl:
        pickle.dump(feature_selector, pkl)
    save(clustering, fname_fn, **kwargs)
