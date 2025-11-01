from itertools import product
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, rankdata

from src.constants import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_SIGNIFICANCE_LEVEL, DEFAULT_TIME_DELAY

_DEFAULT_RANKING_METHOD = 'ordinal'


class DVPartition(TypedDict):
    mins: NDArray[np.integer]
    maxs: NDArray[np.integer]
    N: int


def rank_transform(x: NDArray[np.floating]) -> NDArray[np.integer]:
    return rankdata(x, method=_DEFAULT_RANKING_METHOD)


def get_deleyed_vector(x: NDArray, d: int = DEFAULT_EMBEDDING_DIMENSION, tau: int = DEFAULT_TIME_DELAY) -> NDArray:
    """
    Embedded (delay) vector U_t^{d,τ} for a variable x is defined as:

    U_t^{d,τ} = (U_{t-(d-1)τ}, U_{t-(d-2)τ}, ..., U_t)
    """
    n = len(x) - (d - 1) * tau
    if n <= 0:
        raise ValueError('Time series too short for given embedding.')
    return np.column_stack([x[i : i + n] for i in range((d - 1) * tau, -1, -tau)])


def dv_partition_nd(
    data: NDArray[np.integer],
    mins: NDArray[np.integer] | None = None,
    maxs: NDArray[np.integer] | None = None,
    alpha: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> list[DVPartition]:
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
    if mins is None or maxs is None:
        mins = cast(NDArray[np.integer], np.min(data, axis=0))
        maxs = cast(NDArray[np.integer], np.max(data, axis=0))

    # select points inside current box
    inside = np.all((data >= mins) & (data <= maxs), axis=1)
    sub = data[inside]
    n = len(sub)
    if n == 0:
        return []

    # mid-points along each dimension
    divs: NDArray[np.integer] = (mins + maxs) // 2
    d = len(mins)

    counts, masks = _count_child_boxes(sub, d, divs, mins, maxs)

    is_uniform = _is_uniform(d=d, counts=counts, alpha=alpha)
    if is_uniform is None:
        return []

    if (not is_uniform) and np.any(maxs - mins):
        parts = []
        for bits, mask in zip(product([0, 1], repeat=d), masks, strict=False):
            if mask.sum() == 0:
                continue
            lo = np.where(np.array(bits) == 0, mins, divs + 1)
            hi = np.where(np.array(bits) == 0, divs, maxs)
            parts.extend(dv_partition_nd(data, lo, hi, alpha))
        return parts
    # else this box is a leaf
    return [{'mins': mins.copy(), 'maxs': maxs.copy(), 'N': int(n)}]


def _count_child_boxes(
    sub: NDArray[np.integer], d: int, divs: NDArray[np.integer], mins: NDArray[np.integer], maxs: NDArray[np.integer]
) -> tuple[NDArray[np.integer], list[NDArray[np.bool]]]:
    counts: list[int] = []
    masks: list[NDArray[np.bool]] = []
    for bits in product([0, 1], repeat=d):  # 000… to 111…
        lo = np.where(np.array(bits) == 0, mins, divs + 1)
        hi = np.where(np.array(bits) == 0, divs, maxs)
        mask = np.all((sub >= lo) & (sub <= hi), axis=1)
        masks.append(mask)
        counts.append(mask.sum())
    return np.asarray(counts), masks


def _is_uniform(d: int, counts: NDArray, alpha: float) -> None | bool:
    mean = counts.mean()
    if mean == 0:
        return None
    T = np.sum((mean - counts) ** 2 / mean)
    crit = chi2.ppf(1 - alpha, df=(2**d) - 1)
    return crit >= T
