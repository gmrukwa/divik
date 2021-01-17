import numpy as np

from divik.core import Subsets


def test_subsets():
    subsets = Subsets(n_splits=3, random_state=42)
    # scatters data into datasets
    sct = subsets.scatter(np.arange(30).reshape(15, 2))
    assert len(sct) == 3, len(sct)
    assert sct[0].shape == (5, 2)
    assert sct[1].shape == (5, 2)
    assert sct[2].shape == (5, 2)
    assert tuple(sct[0][1]) != (2, 3)
    # combines data back
    combined = subsets.combine(sct)
    assert np.all(combined == np.arange(30).reshape(15, 2)), combined
    # combine other simple data back
    test_data = [s[:, 0] / 2 for s in sct]
    combined = subsets.combine(test_data)
    assert np.all(combined == np.arange(15)), combined
