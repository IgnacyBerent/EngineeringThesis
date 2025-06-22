import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2
from itertools import product


def get_embedded_vector(x: np.array, d: int, tau: int = 1) -> np.array:
    """
    Embedded vector U_t^{d,\tau} for a variable x is the following vector:

    U_t^{d,\tau} = (U_t,U_{t-\tau},U_{t-2\tau},\dots,U_{t-(d-1)\tau})

    Parameters
    ----------
        x (np.array): A 1D numpy array representing the random variable.
        d (int): The embedding dimension.
        tau (int): The time delay.

    Returns
    -------
        embedded : np.array
            A 2D numpy array where each row is an embedded vector.
    """
    x = np.asarray(x)
    n = len(x) - (d - 1) * tau
    if n <= 0:
        raise ValueError("Time series too short for given embedding.")

    embedded = np.column_stack([x[i : i + n] for i in range(0, d * tau, tau)])
    return embedded


def dv_partition_nd(data, mins, maxs, alpha=0.05) -> list[dict]:
    """
    Generic Darbellay-Vajda adaptive partitioning.

    Parameters
    ----------
    data : (N, d) int ndarray
        Ordinal-ranked samples.  Each row is a point in d-dim space.
    mins, maxs : (d,) int arrays
        Current hyper-box bounds (inclusive).
    alpha : float
        Significance level for the χ² uniformity test.

    Returns
    -------
    partitions : list[dict]
        Each dict has keys:
        - 'mins': (d,) int array, lower bounds of the partition box.
        - 'maxs': (d,) int array, upper bounds of the partition box.
        - 'N': int, number of points in this partition box.
    """
    # select points inside current box
    inside = np.all((data >= mins) & (data <= maxs), axis=1)
    sub = data[inside]
    n = len(sub)
    if n == 0:
        return []

    # mid-points along each dimension
    divs = (mins + maxs) // 2
    d = len(mins)

    # count in every 2^d child box
    counts = []
    masks = []
    for bits in product([0, 1], repeat=d):  # 000… to 111…
        lo = np.where(np.array(bits) == 0, mins, divs + 1)
        hi = np.where(np.array(bits) == 0, divs, maxs)
        mask = np.all((sub >= lo) & (sub <= hi), axis=1)
        masks.append(mask)
        counts.append(mask.sum())
    counts = np.asarray(counts)
    mean = counts.mean()

    # χ² statistic
    if mean == 0:
        return []
    T = np.sum((mean - counts) ** 2 / mean)
    crit = chi2.ppf(1 - alpha, df=(2**d) - 1)

    # if non-uniform and splittable, recurse
    if (T > crit) and np.any(maxs - mins):
        parts = []
        for bits, mask in zip(product([0, 1], repeat=d), masks):
            if mask.sum() == 0:
                continue
            lo = np.where(np.array(bits) == 0, mins, divs + 1)
            hi = np.where(np.array(bits) == 0, divs, maxs)
            parts.extend(dv_partition_nd(data, lo, hi, alpha))
        return parts
    # else this box is a leaf
    return [{"mins": mins.copy(), "maxs": maxs.copy(), "N": int(n)}]
