import colorsys
import itertools
from fractions import Fraction

import numpy as np


def _get_fractions():
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield 0
    for k in itertools.count():
        i = 2 ** k
        for j in range(1, i, 2):
            yield Fraction(j, i)


_DEFAULT_SATURATIONS = [Fraction(6, 10)]
_DEFAULT_VALUES = [Fraction(8, 10), Fraction(5, 10)]


def _generate_hsv_neighbourhood(h):
    for s in _DEFAULT_SATURATIONS:
        for v in _DEFAULT_VALUES:
            yield (h, s, v)


def as_colormap_color(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    r, g, b = int(255 * r), int(255 * g), int(255 * b)
    return 'rgb({0},{1},{2})'.format(r, g, b)


def colormap():
    fractions = _get_fractions()
    neighbors = map(_generate_hsv_neighbourhood, fractions)
    neighbors = itertools.chain.from_iterable(neighbors)
    for args in neighbors:
        yield as_colormap_color(*args)


_DISABLED_COLOR = 'rgb(128, 128, 128)'


def _pick_color(value, color, disabled, overrides):
    if value in disabled:
        return _DISABLED_COLOR
    if str(value) in overrides:
        return overrides[str(value)]
    return color


def make_colormap(values, disabled=None, overrides=None):
    if disabled is None:
        disabled = []
    if overrides is None:
        overrides = {}
    values = np.array(values)
    return [
        [
            value / values.max(),
            _pick_color(value, color, disabled, overrides)
        ]
        for value, color
        in zip(np.unique(values), colormap())
    ]
