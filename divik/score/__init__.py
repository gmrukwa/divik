from ._optimizer import Optimizer, ParameterValues
from ._dunn import dunn, DunnPicker
from ._gap import gap, GapPicker
from ._picker import Picker


def make_picker(method, gap=None):
    if method == 'dunn':
        picker = DunnPicker()
    elif method == 'gap':
        if gap is None:
            gap = {}
        max_iter = gap.get('max_iter', 10)
        seed = gap.get('seed', 0)
        trials = gap.get('trials', 10)
        correction = gap.get('correction', True)
        picker = GapPicker(max_iter, seed, trials, correction)
    else:
        raise ValueError('Unknown quality measure {0}'.format(method))
    return picker
