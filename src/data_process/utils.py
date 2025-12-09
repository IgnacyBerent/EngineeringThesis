from enum import Enum
from typing import cast

import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray

from src.common.constants import SAMPLING_FREQUENCY
from src.common.mytypes import FloatArray

_DEFAULT_MIN_DELAY = 0.3
_DEFAULT_FIND_PEAKS_METHOD = 'elgendi'


class PeaksMode(Enum):
    UP = 'up'
    DOWN = 'down'
    BOTH = 'both'


def get_peaks(
    signal: NDArray[np.floating],
    mode: PeaksMode = PeaksMode.UP,
    sampling_rate: int = SAMPLING_FREQUENCY,
    method: str = _DEFAULT_FIND_PEAKS_METHOD,
    mindelay: float = _DEFAULT_MIN_DELAY,
) -> NDArray[np.floating]:
    filled_signal = nk.signal_fillmissing(signal)
    cleaned_signal = cast(NDArray[np.floating], nk.ppg_clean(filled_signal, sampling_rate=sampling_rate, method=method))

    peaks_up: NDArray[np.floating] | None = None
    peaks_down: NDArray[np.floating] | None = None

    if mode in (PeaksMode.UP, PeaksMode.BOTH):
        peaks_up = _find_peaks(
            cleaned_signal,
            sampling_rate=sampling_rate,
            method=method,
            mindelay=mindelay,
        )
    if mode in (PeaksMode.DOWN, PeaksMode.BOTH):
        peaks_down = _find_peaks(
            cleaned_signal * -1,
            sampling_rate=sampling_rate,
            method=method,
            mindelay=mindelay,
        )

    if peaks_up is not None and peaks_down is not None:
        return np.sort(np.concatenate((peaks_up, peaks_down)))
    if peaks_up is not None:
        return peaks_up
    return cast(NDArray[np.floating], peaks_down)


def _find_peaks(
    cleaned_signal: NDArray[np.floating],
    sampling_rate: int = SAMPLING_FREQUENCY,
    method: str = _DEFAULT_FIND_PEAKS_METHOD,
    mindelay: float = _DEFAULT_MIN_DELAY,
) -> NDArray[np.floating]:
    peaks = nk.ppg_findpeaks(
        cleaned_signal,
        sampling_rate=sampling_rate,
        method=method,
        mindelay=mindelay,
    )['PPG_Peaks']
    return cast(NDArray[np.floating], peaks)


def get_hp_from_peaks(peaks: NDArray[np.floating], sampling_rate: int = SAMPLING_FREQUENCY) -> NDArray[np.floating]:
    sampling_period = 1 / sampling_rate * 1000
    return np.diff(peaks) * sampling_period


def get_hp_from_abp(abp: NDArray[np.floating], sampling_rate: int = SAMPLING_FREQUENCY) -> NDArray[np.floating]:
    peaks = get_peaks(abp, PeaksMode.UP, sampling_rate)
    return get_hp_from_peaks(peaks)


def get_sap(abp: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate Systolic Amplitude Peaks (SAP) from abp signal and its peak indices.
        It is assumed that the peaks are upward peaks.
        SAP(i) equals the value of the signal at the peak index.
    """
    peaks = get_peaks(abp, PeaksMode.UP)
    return np.array([abp[peak] for peak in peaks])[1:]  # skip first peak to match length of hp


def get_map(abp: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate Mean Arterial Pressure (MAP) from abp signal and its peak indices.
        It assumes that the peaks are alternating between downward and upward peaks.
        The first peak is assumed to be a downward peak.
    """
    peaks = get_peaks(abp, PeaksMode.BOTH)
    first_downward_peak_index = 0 if peaks[0] < peaks[1] else 1

    map_ = []
    for i in range(first_downward_peak_index, len(peaks) - 1, 2):
        dp = abp[peaks[i]]
        sp = abp[peaks[i + 1]]
        map_.append((2 * dp + sp) / 3)

    return np.array(map_)


def get_mfv(fv: FloatArray) -> FloatArray:
    peaks = get_peaks(fv, PeaksMode.UP)
    return np.array([np.mean(fv[peaks[i - 1] : peaks[i]]) for i, _ in enumerate(peaks) if i > 0])
