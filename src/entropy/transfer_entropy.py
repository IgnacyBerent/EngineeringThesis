import numpy as np
from numpy.typing import NDArray

from src.common.constants import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_SIGNIFICANCE_LEVEL, DEFAULT_TIME_DELAY
from src.entropy.entropy_helpers import DVPartition, dv_partition_nd, get_deleyed_vector, rank_transform


def transfer_entropy(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    time_delay: int = DEFAULT_TIME_DELAY,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    dvp_alpha=DEFAULT_SIGNIFICANCE_LEVEL,
) -> float:
    """
    Calculates the transfer entropy of TE_{Y->X}
    """

    trimmedX = signalX[:-time_delay]
    trimmedY = signalY[:-time_delay]

    embeddedX, embeddedY = (
        get_deleyed_vector(trimmedX, d=embedding_dimension, tau=time_delay),
        get_deleyed_vector(trimmedY, d=embedding_dimension, tau=time_delay),
    )
    futureX = signalX[(embedding_dimension - 1) * time_delay + 1 :]

    futureX = rank_transform(futureX)
    embeddedX = np.apply_along_axis(lambda x: rank_transform(x), axis=0, arr=embeddedX)
    embeddedY = np.apply_along_axis(lambda x: rank_transform(x), axis=0, arr=embeddedY)

    fXeXeY = np.column_stack([futureX, embeddedX, embeddedY])
    eX = np.column_stack([embeddedX])
    fXeX = np.column_stack([futureX, embeddedX])
    eXeY = np.column_stack([embeddedX, embeddedY])

    dv_result = dv_partition_nd(fXeXeY, alpha=dvp_alpha)

    dimensions = fXeXeY.shape[1]
    n_total = fXeXeY.shape[0]
    te: float = 0
    for dv_part in dv_result:
        n_fXpeXpeYp = dv_part['N']
        n_eX = _get_points_from_range(eX, dv_part, start=1, stop=1 + embedding_dimension)
        n_fXeX = _get_points_from_range(fXeX, dv_part, start=0, stop=1 + embedding_dimension)
        n_eXeY = _get_points_from_range(eXeY, dv_part, start=1, stop=dimensions)

        arg1 = n_fXpeXpeYp * n_eX
        arg2 = n_fXeX * n_eXeY

        te += n_fXpeXpeYp / n_total * (np.log2(arg1) - np.log2(arg2))

    return te


def _get_points_from_range(points: NDArray[np.integer], dv_part: DVPartition, start: int, stop: int) -> int:
    mins, maxs = _get_min_max(dv_part, start, stop)
    mask = np.all((points >= mins) & (points <= maxs), axis=1)
    return int(mask.sum())


def _get_min_max(dv_part: DVPartition, start: int, stop: int) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    return dv_part['mins'][start:stop], dv_part['maxs'][start:stop]
