from ._dunn import DunnPicker
from ._gap import gap, GapPicker
from ._picker import Picker


def make_picker(method, n_jobs: int = 1, gap=None):
    if method == 'dunn':
        picker = DunnPicker(n_jobs=n_jobs)
    elif method == 'gap':
        if gap is None:
            gap = {}
        max_iter = gap.get('max_iter', 10)
        seed = gap.get('seed', 0)
        trials = gap.get('trials', 10)
        correction = gap.get('correction', True)
        picker = GapPicker(max_iter, seed, trials, correction, n_jobs=n_jobs)
    else:
        raise ValueError('Unknown quality measure {0}'.format(method))
    return picker
