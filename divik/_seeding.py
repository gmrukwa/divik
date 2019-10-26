"""Convenient control over seeding scopes.

_seeding.py

Copyright 2019 Grzegorz Mrukwa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from contextlib import contextmanager
from functools import wraps

import numpy as np


@contextmanager
def seed(seed_: int = 0):
    """Crete seeded scope."""
    state = np.random.get_state()
    np.random.seed(seed_)
    yield
    np.random.set_state(state)


def seeded(wrapped_requires_seed: bool = False):
    """Create seeded scope for function call.

    @param wrapped_requires_seed: if true, passes seed parameter to the inner
    function
    """
    get = dict.get if wrapped_requires_seed else dict.pop

    def _seeded_maker(func):
        @wraps(func)
        def _seeded(*args, **kwargs):
            _seed = get(kwargs, 'seed', 0)
            with seed(_seed):
                return func(*args, **kwargs)
        return _seeded

    return _seeded_maker
