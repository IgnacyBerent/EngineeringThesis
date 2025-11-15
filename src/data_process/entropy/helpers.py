from itertools import product
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, rankdata

from src.common.constants import DEFAULT_SIGNIFICANCE_LEVEL
from src.common.mytypes import FloatArray

_DEFAULT_RANKING_METHOD = 'ordinal'


class DVPartition(TypedDict):
    mins: NDArray[np.integer]
    maxs: NDArray[np.integer]
    N: int


def get_points_from_range(
    points: NDArray[np.integer], dv_part: DVPartition, ranges: tuple[tuple[int, int], ...]
) -> int:
    mins: NDArray[np.integer] | None = None
    maxs: NDArray[np.integer] | None = None

    for start, stop in ranges:
        part_mins, part_maxs = _get_min_max(dv_part, start, stop)

        # Initialize on first loop
        if mins is None or maxs is None:
            mins = part_mins
            maxs = part_maxs
        else:
            mins = np.hstack([mins, part_mins])
            maxs = np.hstack([maxs, part_maxs])

    mins, maxs = cast(NDArray[np.integer], mins), cast(NDArray[np.integer], maxs)
    mask = np.all((points >= mins) & (points <= maxs), axis=1)
    return int(mask.sum())


def _get_min_max(dv_part: DVPartition, start: int, stop: int) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    return dv_part['mins'][start:stop], dv_part['maxs'][start:stop]


def get_past_vectors(signal: FloatArray, d: int, tau: int) -> NDArray[np.integer]:
    embedded = get_deleyed_vector(signal, d=d, tau=tau)
    return np.apply_along_axis(lambda x: rank_transform(x), axis=0, arr=embedded)


def get_deleyed_vector(x: NDArray, d: int, tau: int) -> NDArray:
    """
    Embedded (delay) vector U_t^{d,τ} for a variable x is defined as:

    U_t^{d,τ} = (U_{t-(d-1)τ}, U_{t-(d-2)τ}, ..., U_t)
    """
    n = len(x) - d * tau
    if n <= 0:
        raise ValueError('Time series too short for given embedding.')
    return np.column_stack([x[i - (tau - 1) : i + n - (tau - 1)] for i in range(d * tau - 1, -1, -tau)])


def get_future_vector(signal: NDArray, d: int, tau: int) -> NDArray:
    return rank_transform(signal[d * tau :])


def rank_transform(x: NDArray[np.floating]) -> NDArray[np.integer]:
    return rankdata(x, method=_DEFAULT_RANKING_METHOD)


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
