import numpy as np
from numpy.typing import NDArray

from src.common.constants import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_SIGNIFICANCE_LEVEL, DEFAULT_TIME_DELAY
from src.common.logger import logger
from src.data_process.entropy.helpers import (
    dv_partition_nd,
    get_points_from_range,
    rank_transform,
    trim_embed_rank,
)


def cjte_dv(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    signalZ: NDArray[np.floating],
    signalW: NDArray[np.floating],
    time_delay: int = DEFAULT_TIME_DELAY,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    dvp_alpha=DEFAULT_SIGNIFICANCE_LEVEL,
) -> float:
    """
    Calculates the conditional joint transfer entropy of CJTE_{(X,Y)->Z|W}
    """
    if not len(signalX) == len(signalY) == len(signalZ) == len(signalW):
        logger.error(
            f"""Signals should have the same legth, instead have: \n
            X:{len(signalX)}, Y:{len(signalY)}, Z:{len(signalZ)}, W:{len(signalW)}"""
        )
        raise ValueError('time series entries need to have same length')

    futureZ = rank_transform(signalZ[(embedding_dimension - 1) * time_delay + 1 :])
    pastX, pastY, pastZ, pastW = (
        trim_embed_rank(signal, d=embedding_dimension, tau=time_delay)
        for signal in [signalX, signalY, signalZ, signalW]
    )

    a = np.column_stack([futureZ, pastZ, pastX, pastY, pastW])
    b = np.column_stack([pastZ, pastW])
    c = np.column_stack([futureZ, pastZ, pastW])
    d = np.column_stack([pastZ, pastX, pastY, pastW])

    dv_result = dv_partition_nd(a, alpha=dvp_alpha)
    dimensions = a.shape[1]
    n_total = a.shape[0]

    futureZ_start, futureZ_end = 0, 1
    pastZ_start, pastZ_end = futureZ_end, embedding_dimension + 1
    pastW_start, pastW_end = 3 * embedding_dimension + 1, dimensions

    cjte: float = 0
    for dv_part in dv_result:
        na = dv_part['N']
        nb = get_points_from_range(b, dv_part, ranges=((pastZ_start, pastZ_end), (pastW_start, pastW_end)))
        nc = get_points_from_range(c, dv_part, ranges=((futureZ_start, pastZ_end), (pastW_start, pastW_end)))
        nd = get_points_from_range(d, dv_part, ranges=((pastZ_start, pastW_end),))

        cjte += na / n_total * (np.log2(na * nb) - np.log2(nc * nd))

    return cjte
