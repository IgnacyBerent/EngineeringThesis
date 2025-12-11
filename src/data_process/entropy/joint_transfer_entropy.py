import numpy as np
from numpy.typing import NDArray

from src.common.constants import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_SIGNIFICANCE_LEVEL, DEFAULT_TIME_DELAY
from src.common.logger import logger
from src.data_process.entropy.dvp import dv_partition_nd
from src.data_process.entropy.utils import (
    get_future_vector,
    get_past_vectors,
    get_points_from_range,
)


def jte_dv(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    signalZ: NDArray[np.floating],
    time_delay: int = DEFAULT_TIME_DELAY,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    dvp_alpha=DEFAULT_SIGNIFICANCE_LEVEL,
) -> float:
    """
    Calculates the conditional transfer entropy of JTE_{(X,Y)->Z}
    """
    if not len(signalX) == len(signalY) == len(signalZ):
        logger.error(
            f"""Signals should have the same legth, instead have: \n
            X:{len(signalX)}, Y:{len(signalY)}, Z:{len(signalZ)}"""
        )
        raise ValueError('time series entries need to have same length')

    futureZ = get_future_vector(signalZ, d=embedding_dimension, tau=time_delay)
    pastX, pastY, pastZ = (
        get_past_vectors(signal, d=embedding_dimension, tau=time_delay) for signal in [signalX, signalY, signalZ]
    )

    a = np.column_stack([futureZ, pastZ, pastX, pastY])
    b = np.column_stack([pastZ])
    c = np.column_stack([futureZ, pastZ])
    d = np.column_stack([pastZ, pastX, pastY])

    dv_result = dv_partition_nd(a, alpha=dvp_alpha)
    dimensions = a.shape[1]
    n_total = a.shape[0]

    futureZ_start, futureZ_end = 0, 1
    pastZ_start, pastZ_end = futureZ_end, embedding_dimension + 1
    pastY_end = dimensions

    jte: float = 0
    for dv_part in dv_result:
        na = dv_part['N']
        nb = get_points_from_range(b, dv_part, ranges=((pastZ_start, pastZ_end),))
        nc = get_points_from_range(c, dv_part, ranges=((futureZ_start, pastZ_end),))
        nd = get_points_from_range(d, dv_part, ranges=((pastZ_start, pastY_end),))

        jte += na / n_total * (np.log2(na * nb) - np.log2(nc * nd))

    return jte
