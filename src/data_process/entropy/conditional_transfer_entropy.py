import numpy as np
from numpy.typing import NDArray

from src.common.constants import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_SIGNIFICANCE_LEVEL, DEFAULT_TIME_DELAY
from src.common.logger import logger
from src.data_process.entropy.helpers import (
    dv_partition_nd,
    get_future_vector,
    get_past_vectors,
    get_points_from_range,
)


def cte_dv(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    signalZ: NDArray[np.floating],
    time_delay: int = DEFAULT_TIME_DELAY,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    dvp_alpha=DEFAULT_SIGNIFICANCE_LEVEL,
) -> float:
    """
    Calculates the conditional transfer entropy of CTE_{Y->X|Z}
    """
    if not len(signalX) == len(signalY) == len(signalZ):
        logger.error(
            f"""Signals should have the same legth, instead have: \n
            X:{len(signalX)}, Y:{len(signalY)}, Z:{len(signalZ)}"""
        )
        raise ValueError('time series entries need to have same length')

    futureX = get_future_vector(signalX, d=embedding_dimension, tau=time_delay)
    pastX, pastY, pastZ = (
        get_past_vectors(signal, d=embedding_dimension, tau=time_delay) for signal in [signalX, signalY, signalZ]
    )

    a = np.column_stack([futureX, pastX, pastY, pastZ])
    b = np.column_stack([pastX, pastZ])
    c = np.column_stack([futureX, pastX, pastZ])
    d = np.column_stack([pastX, pastY, pastZ])

    dv_result = dv_partition_nd(a, alpha=dvp_alpha)
    dimensions = a.shape[1]
    n_total = a.shape[0]

    futureX_start, futureX_end = 0, 1
    pastX_start, pastX_end = futureX_end, embedding_dimension + 1
    pastZ_start, pastZ_end = 2 * embedding_dimension + 1, dimensions

    cte: float = 0
    for dv_part in dv_result:
        na = dv_part['N']
        nb = get_points_from_range(b, dv_part, ranges=((pastX_start, pastX_end), (pastZ_start, pastZ_end)))
        nc = get_points_from_range(c, dv_part, ranges=((futureX_start, pastX_end), (pastZ_start, pastZ_end)))
        nd = get_points_from_range(d, dv_part, ranges=((pastX_start, pastZ_end),))

        cte += na / n_total * (np.log2(na * nb) - np.log2(nc * nd))

    return cte
