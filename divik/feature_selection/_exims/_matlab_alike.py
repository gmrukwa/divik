import numpy as np


def quantile(values, quantiles):
    """Compute MATLAB-alike quantiles

    Arguments:
        values - (np.ndarray) Input array or object that can be converted to an
        array.
        quantiles - (float, np.ndarray) float in range of [0,1] (or sequence
        of floats). Quantile to compute, which must be between 0 and 1
        inclusive.


    location of first element as a quantile -> 0.5 / n
    location of last element as a quantile -> (n - 0.5) / n

         ^
         |     (n-0.5) / n    1.0
     1.0 ----------------|---|
         |              /
         |             /
         |            /
         |           /                  <- this is how quantiles look in MATLAB
         |          /
         |         /
         |        /
         |       /
     0.0 ----|---|------------->
         | 0.0   0.5 / n


    y = ax + b


    0 = a * 0.5 / n + b         <- 0th element is treated as 0.5 / n quantile
    1 = a * (n - 0.5) / n + b   <- last element is treated as (n - 0.5) / n quantile

    a = n / (n - 1)
    b = - 0.5 / (n - 1)

    """
    values = np.array(values)
    n = float(values.size)
    a = n / (n - 1)
    b = -0.5 / (n - 1)
    quantiles = np.array(quantiles)
    matlab_alike_quantiles = np.clip(a * quantiles + b, a_min=0.0, a_max=1.0)
    return np.percentile(values, q=100.0 * matlab_alike_quantiles)


def n_quantiles(values, N, unbiased=True, backend=quantile):
    return backend(values, np.arange(1, N + 1, dtype=float) / (N + int(unbiased)))


def iqr(values, rng=(25, 75)):
    q1, q3 = quantile(values, 0.01 * np.array(rng))
    return q3 - q1
