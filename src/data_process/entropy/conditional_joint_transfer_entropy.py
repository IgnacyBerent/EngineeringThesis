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


def cjte_dv(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    signalZ: NDArray[np.floating],
    signalW: NDArray[np.floating] | None = None,
    time_delay: int = DEFAULT_TIME_DELAY,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    dvp_alpha: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> float:
    """
    Calculates the conditional joint transfer entropy

    If W is not None, unique signal then uses formula:
        CJTE_{(X,Y)->Z|W}

    If W is None, then the following formula is assumed:
        CJTE_{(X,Y)->Z|Y}
    """
    if signalW is None:
        return _cjte_y_is_w(
            signalX=signalX,
            signalY=signalY,
            signalZ=signalZ,
            time_delay=time_delay,
            embedding_dimension=embedding_dimension,
            dvp_alpha=dvp_alpha,
        )
    return _cjte_w_is_different(
        signalX=signalX,
        signalY=signalY,
        signalZ=signalZ,
        signalW=signalW,
        time_delay=time_delay,
        embedding_dimension=embedding_dimension,
        dvp_alpha=dvp_alpha,
    )


def _cjte_y_is_w(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    signalZ: NDArray[np.floating],
    time_delay: int,
    embedding_dimension: int,
    dvp_alpha: float,
) -> float:
    """
    Calculates the conditional joint transfer entropy of CJTE_{(X,Y)->Z|Y}
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
    b = np.column_stack([pastZ, pastY])
    c = np.column_stack([futureZ, pastZ, pastY])
    d = np.column_stack([pastZ, pastX, pastY])

    dv_result = dv_partition_nd(a, alpha=dvp_alpha)
    dimensions = a.shape[1]
    n_total = a.shape[0]

    futureZ_start, futureZ_end = 0, 1
    pastZ_start, pastZ_end = futureZ_end, embedding_dimension + 1
    pastY_start, pastY_end = 2 * embedding_dimension + 1, dimensions

    cjte: float = 0
    for dv_part in dv_result:
        na = dv_part['N']
        nb = get_points_from_range(b, dv_part, ranges=((pastZ_start, pastZ_end), (pastY_start, pastY_end)))
        nc = get_points_from_range(c, dv_part, ranges=((futureZ_start, pastZ_end), (pastY_start, pastY_end)))
        nd = get_points_from_range(d, dv_part, ranges=((pastZ_start, pastY_end),))

        cjte += na / n_total * (np.log2(na * nb) - np.log2(nc * nd))

    return cjte


def _cjte_w_is_different(
    signalX: NDArray[np.floating],
    signalY: NDArray[np.floating],
    signalZ: NDArray[np.floating],
    signalW: NDArray[np.floating],
    time_delay: int,
    embedding_dimension: int,
    dvp_alpha: float,
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

    futureZ = get_future_vector(signalZ, d=embedding_dimension, tau=time_delay)
    pastX, pastY, pastZ, pastW = (
        get_past_vectors(signal, d=embedding_dimension, tau=time_delay)
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
