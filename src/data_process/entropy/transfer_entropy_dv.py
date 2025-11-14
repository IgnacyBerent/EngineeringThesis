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


def te_dv(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    time_delay: int = DEFAULT_TIME_DELAY,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    dvp_alpha=DEFAULT_SIGNIFICANCE_LEVEL,
) -> float:
    """
    Calculates the transfer entropy of TE_{Y->X}
    """
    if len(signalX) != len(signalY):
        logger.error(
            f"""Signals should have the same legth, instead have: \n
            X:{len(signalX)}, Y:{len(signalY)}"""
        )
        raise ValueError('time series entries need to have same length')

    futureX = rank_transform(signalX[(embedding_dimension - 1) * time_delay + 1 :])
    pastX, pastY = (trim_embed_rank(signal, d=embedding_dimension, tau=time_delay) for signal in [signalX, signalY])

    a = np.column_stack([futureX, pastX, pastY])
    b = np.column_stack([pastX])
    c = np.column_stack([futureX, pastX])
    d = np.column_stack([pastX, pastY])

    dv_result = dv_partition_nd(a, alpha=dvp_alpha)
    dimensions = a.shape[1]
    n_total = a.shape[0]

    futureX_start, futureX_end = 0, 1
    pastX_start, pastX_end = futureX_end, embedding_dimension + 1
    pastY_end = dimensions

    te: float = 0
    for dv_part in dv_result:
        na = dv_part['N']
        nb = get_points_from_range(b, dv_part, ranges=((pastX_start, pastX_end),))
        nc = get_points_from_range(c, dv_part, ranges=((futureX_start, pastX_end),))
        nd = get_points_from_range(d, dv_part, ranges=((pastX_start, pastY_end),))

        te += na / n_total * (np.log2(na * nb) - np.log2(nc * nd))

    return te
