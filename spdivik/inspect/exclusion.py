import numpy as np

from spdivik.summary import merged_partition
from spdivik.inspect.app import divik_result


def get_options(level):
    partition = merged_partition(divik_result(), level)
    return [{'label': e, 'value': e} for e in np.unique(partition)]
