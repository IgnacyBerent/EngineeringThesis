from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from src.common.mytypes import FloatArray
from src.data_process.entropy.dvp import DVPartition

MINIMAL_VALID_NUMBER_OF_DV_PARTITONS = 2
_DEFAULT_RANKING_METHOD = 'average'


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
    return np.all((points >= mins) & (points <= maxs), axis=1).sum()


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
