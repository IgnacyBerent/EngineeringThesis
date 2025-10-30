from typing import cast

import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt

from src.constants import SAMPLING_FREQUENCY
from src.mytypes import FFT_Result

DEFAULT_BUTTERFILT_ORDER = 4
DEFAULT_FREQUENCY_SEPARATION = 0.2


def fft_process(x: NDArray[np.floating], fc: int, fs: int = SAMPLING_FREQUENCY) -> FFT_Result:
    """
    Compute the FFT of a detrended, low-pass filtered signal.
    """
    x = x - np.mean(x)
    x = nk.signal_detrend(x)

    coeff_b, coeff_a = cast(
        tuple[NDArray[np.floating], NDArray[np.floating]], butter(DEFAULT_BUTTERFILT_ORDER, fc / (fs / 2), btype='low')
    )
    x_filt = filtfilt(coeff_b, coeff_a, x)

    N = len(x_filt)
    X = cast(NDArray[np.complex128], fft(x_filt))

    mag = np.abs(X) / N
    mag = mag[: N // 2]
    f = np.linspace(0, fs / 2, N // 2, endpoint=False)
    phases = np.angle(X[: N // 2])

    return FFT_Result(X, f, mag, phases)


def find_most_significant_frequencies(
    f: NDArray[np.floating],
    mag: NDArray[np.floating],
    n: int,
    min_sep: float = DEFAULT_FREQUENCY_SEPARATION,
) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
    """
    Select the top-n frequencies by magnitude, ensuring minimum separation.

    Parameters
    ----------
    f : NDArray[np.floating]
        Frequency bins.
    mag : NDArray[np.floating]
        Magnitude spectrum.
    min_sep : float
        Minimum frequency separation (in Hz) between picks.
    n : int
        Number of frequencies to select.

    Returns
    -------
    idxs : NDArray[np.integer]
        Indices in X/f corresponding to the selected frequencies.
    freqs : NDArray[np.floating]
        The selected frequency values.
    """
    sorted_indices = np.argsort(mag)[::-1]

    selected_indices: list[int] = []
    selected_freqs: list[float] = []

    for idx in sorted_indices:
        freq = f[idx]
        if all(abs(freq - f[i]) >= min_sep for i in selected_indices):
            selected_indices.append(int(idx))
            selected_freqs.append(float(freq))
        if len(selected_indices) >= n:
            break

    return np.array(selected_indices, dtype=int), np.array(selected_freqs, dtype=float)


def filter_frequencies(X: NDArray[np.complex128], idxs: NDArray[np.integer]) -> NDArray[np.complex128]:
    """
    Zero out all FFT bins except those at the specified indices.

    Parameters
    ----------
    X : NDArray[np.complex_]
        Full FFT spectrum.
    idxs : NDArray[np.integer]
        Indices of the bins to retain.

    Returns
    -------
    X_filtered : NDArray[np.complex_]
        FFT spectrum with only the selected bins non-zero.
    """
    X_filtered = np.zeros_like(X)
    X_filtered[idxs] = X[idxs]
    X_filtered[-idxs] = X[-idxs]
    return X_filtered


def reconstruct_signal(X: NDArray[np.complex128], mean_value: float) -> NDArray[np.floating]:
    """
    Inverse FFT to recover a real-valued signal, then add back the mean.

    Parameters
    ----------
    X : NDArray[np.complex_]
        FFT spectrum.
    mean_value : float
        Mean of the original signal to re-add.

    Returns
    -------
    reconstructed : NDArray[np.floating]
        Real part of the inverse FFT plus the mean.
    """
    reconstructed = cast(NDArray[np.complex128], ifft(X)).real
    reconstructed += mean_value
    return reconstructed
