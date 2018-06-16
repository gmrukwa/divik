from contextlib import contextmanager
from functools import wraps

import numpy as np


@contextmanager
def seed(seed_: int=0):
    state = np.random.get_state()
    np.random.seed(seed_)
    yield
    np.random.set_state(state)


def seeded(wrapped_requires_seed: bool=False):
    get = dict.get if wrapped_requires_seed else dict.pop

    def _seeded_maker(func):
        @wraps(func)
        def _seeded(*args, **kwargs):
            _seed = get(kwargs, 'seed', 0)
            with seed(_seed):
                return func(*args, **kwargs)
        return _seeded

    return _seeded_maker
