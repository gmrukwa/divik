from ._dunn import DunnPicker
from ._gap import gap, GapPicker
from ._sampled_gap import sampled_gap, SamplingGapPicker
from ._picker import Picker


def make_picker(method, n_jobs: int = 1, gap=None):
    if gap is None:
        gap = {}
    if method == 'dunn':
        return DunnPicker(n_jobs=n_jobs)
    if method == 'gap':
        return GapPicker(n_jobs=n_jobs, **gap)
    if method == 'sampled_gap':
        return SamplingGapPicker(n_jobs=n_jobs, **gap)
    raise ValueError('Unknown quality measure {0}'.format(method))
