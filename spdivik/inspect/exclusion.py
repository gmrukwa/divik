import json

import numpy as np

from spdivik.summary import merged_partition
from spdivik.inspect.app import divik_result


def initialize_storage(level):
    return json.dumps({
        'excluded': [],
        'level': level
    })


def vec_in(v1, v2):
    if np.size(v2) == 0:
        return np.zeros((np.size(v1),), dtype=bool)
    return np.max(np.reshape(v1, (-1, 1)) == np.reshape(v2, (1, -1)), axis=1)


def update_excluded(old_level, new_level, old_excluded):
    # if any subcluster is enabled, parent cluster will be enabled
    # if parent is disabled, children will be disabled
    old_partition = merged_partition(divik_result(), old_level)
    old_enabled_flag = vec_in(old_partition, old_excluded) == 0

    new_partition = merged_partition(divik_result(), new_level)
    new_enabled_ids = np.unique(new_partition[old_enabled_flag])
    new_disabled_ids = np.setdiff1d(new_partition, new_enabled_ids)

    return sorted(list(map(int, new_disabled_ids)))


def update_storage(level, disabled_clusters, old_state):
    state = json.loads(old_state)
    state['excluded'] = disabled_clusters
    state['level'] = level
    return json.dumps(state)


def get_options(level):
    partition = merged_partition(divik_result(), level)
    return [{'label': e, 'value': e} for e in np.unique(partition)]


def update_actual_clusters(level, storage):
    if not storage:
        return []
    storage = json.loads(storage)
    if level == storage['level']:
        return storage['excluded']
    return update_excluded(storage['level'], level, storage['excluded'])
