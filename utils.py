import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy.signal as sp_sig
import re


def find_peaks(
    signal: np.array,
    sampling_rate: int = 200,
    mindelay: float = 0.3,
    mode: str = "up",
) -> np.array:
    """
    Find peaks in a signal.

    args:
        signal (np.array): signal data.
        sampling_rate (int): Sampling rate of the signal in Hz.
        mindelay (float): Minimum delay between peaks in seconds.
        mode (str): "up" for upward peaks, "down" for downward peaks, "both" for both.
    returns:
        np.array: Indices of the detected peaks in the signal.
    exceptions:
        ValueError: If mode is not one of "up", "down", or "both".
    """

    filled_signal = nk.signal_fillmissing(signal)
    cleaned_signal = nk.ppg_clean(
        filled_signal, sampling_rate=sampling_rate, method="elgendi"
    )

    peaks_up = None
    peaks_down = None

    if mode not in ["up", "down", "both"]:
        raise ValueError("Mode must be 'up', 'down', or 'both'.")

    if mode == "up" or mode == "both":
        peaks_up = nk.ppg_findpeaks(
            cleaned_signal,
            sampling_rate=sampling_rate,
            method="elgendi",
            mindelay=mindelay,
        )["PPG_Peaks"]
    elif mode == "down" or mode == "both":
        peaks_down = nk.ppg_findpeaks(
            cleaned_signal * -1,
            sampling_rate=sampling_rate,
            method="elgendi",
            mindelay=mindelay,
        )["PPG_Peaks"]

    if mode == "up":
        return peaks_up
    elif mode == "down":
        return peaks_down
    elif mode == "both":
        peaks = np.sort(np.concatenate((peaks_up, peaks_down)))
        return peaks


def get_hp(peaks: np.array, sampling_rate: int = 200):
    """
    Calculate heart period (HP) from peak indices of abp signal.

        args:
            peaks (np.array): Indices of the detected upward peaks in the signal.
            sampling_rate (int): Sampling rate of the signal in Hz.
        returns:
            np.array: Heart period (HP) in seconds.
    """
    rr = np.diff(peaks) / sampling_rate
    hp = 1 / rr
    return hp


def get_sap(signal: np.array, peaks: np.array) -> np.array:
    """
    Calculate Systolic Amplitude Peaks (SAP) from abp signal and its peak indices.
        It skips the first peak to match the length of heart period (HP).
        It is assumed that the peaks are upward peaks.
        SAP(i) equals the value of the signal at the peak index.

        args:
            signal (np.array): signal data.
            peaks (np.array): Indices of the detected upward peaks in the signal.
        returns:
            np.array: Systolic Amplitude Peaks (SAP) in the signal.
    """
    sap = np.array([signal[peak] for peak in peaks])[
        1:
    ]  # skip first peak to match length of hp
    return sap


def get_map(signal: np.array, peaks: np.array) -> np.array:
    """
    Calculate Mean Arterial Pressure (MAP) from abp signal and its peak indices.
        It assumes that the peaks are alternating between downward and upward peaks.
        The first peak is assumed to be a downward peak.
        The MAP is calculated as (2 * DP + SP) / 3, where DP is the value at the downward peak and SP is the value at the upward peak.

        args:
            signal (np.array): signal data.
            peaks (np.array): Indices of the detected upward and downward peaks in the signal.
        returns:
            np.array: Mean Arterial Pressure (MAP) calculated from the peaks in the signal.
    """
    first_peak = 0 if peaks[0] < peaks[1] else 1

    # MAP = (2*DP + SP) / 3
    map_ = []
    for i in range(first_peak, len(peaks) - 2, 2):
        dp = signal[peaks[i]]
        sp = signal[peaks[i + 1]]
        map_.append((2 * dp + sp) / 3)

    return np.array(map_)
