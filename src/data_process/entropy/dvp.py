from itertools import product
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from src.common.constants import DEFAULT_SIGNIFICANCE_LEVEL


class DVPartition(TypedDict):
    mins: NDArray[np.integer]
    maxs: NDArray[np.integer]
    N: int


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
    is_initial = False
    if mins is None or maxs is None:
        is_initial = True
        mins = cast(NDArray[np.integer], np.min(data, axis=0))
        maxs = cast(NDArray[np.integer], np.max(data, axis=0))
    dimensions = len(mins)

    current_box_data = _get_current_box_data(data, mins, maxs)
    n = len(current_box_data)
    if n == 0:
        return []

    children, counts = _get_children_with_counts(current_box_data, mins, maxs)

    is_uniform = _is_uniform(dimensions, counts=counts, alpha=alpha)
    if is_uniform is None:
        return []

    parts = []
    if is_initial or ((not is_uniform) and np.any(maxs - mins)):
        for child_mins, child_maxs in (child for child in children if child is not None):
            parts.extend(dv_partition_nd(data, child_mins, child_maxs, alpha))

        return parts

    # else this box is a leaf
    return [{'mins': mins.copy(), 'maxs': maxs.copy(), 'N': int(n)}]


def _get_current_box_data(
    data: NDArray[np.integer],
    mins: NDArray[np.integer],
    maxs: NDArray[np.integer],
) -> NDArray[np.integer]:
    """
    Selects the subset of data points that fall within the current hyper-box.
    """
    inside = np.all((data >= mins) & (data <= maxs), axis=1)
    return data[inside]


def _get_children_with_counts(
    current_box_data: NDArray[np.integer],
    mins: NDArray[np.integer],
    maxs: NDArray[np.integer],
) -> tuple[list[tuple[NDArray[np.integer], NDArray[np.integer]] | None], NDArray[np.integer]]:
    midpoints: NDArray[np.integer] = (mins + maxs) / 2  # type: ignore
    dimensions = len(mins)

    children = []
    counts = []
    for bits in product([0, 1], repeat=dimensions):
        child_mins, child_maxs = _get_child_box_bounds(bits, mins, maxs, midpoints)
        count = np.all((current_box_data >= child_mins) & (current_box_data <= child_maxs), axis=1).sum()
        counts.append(count)
        if count == 0:
            children.append(None)
        else:
            children.append((child_mins, child_maxs))

    return children, np.asarray(counts)


def _get_child_box_bounds(
    bits: tuple[int, ...], mins: NDArray[np.integer], maxs: NDArray[np.integer], midpoints: NDArray[np.integer]
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Calculates mins and maxs for bounds of a child box."""
    # If bit is 0, use mins/midpoints; if bit is 1, use midpoints+1/maxs
    bits_array = np.array(bits)
    child_mins = np.where(bits_array == 0, mins, midpoints + 1)
    child_maxs = np.where(bits_array == 0, midpoints, maxs)
    return child_mins, child_maxs


def _is_uniform(d: int, counts: NDArray, alpha: float) -> None | bool:
    mean = counts.mean()
    if mean == 0:
        return None
    T = np.sum((mean - counts) ** 2 / mean)
    crit = chi2.ppf(1 - alpha, df=(2**d) - 1)
    return crit >= T
