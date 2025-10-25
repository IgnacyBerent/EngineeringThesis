import pandas as pd
import numpy as np
from numpy.typing import NDArray  # added for type annotations
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft


def fft_process(x: NDArray[np.floating], fc: float = 30, fs: int = 200) -> tuple[
    NDArray[np.complex_],  # full FFT spectrum
    NDArray[np.floating],  # frequency bins
    NDArray[np.floating],  # one‐sided magnitude spectrum
    NDArray[np.floating],  # phase angles
]:
    """
    Compute the FFT of a detrended, low‐pass filtered signal.

    Parameters
    ----------
    x : NDArray[np.floating]
        Input time‐series.
    fc : float
        Cut‐off frequency for the Butterworth filter.
    fs : int
        Sampling frequency of the signal.

    Returns
    -------
    X : NDArray[np.complex_]
        Complex FFT spectrum of the filtered signal.
    f : NDArray[np.floating]
        Array of frequencies corresponding to the FFT bins.
    mag : NDArray[np.floating]
        One‐sided magnitude spectrum.
    phases : NDArray[np.floating]
        Phase angles of the FFT coefficients.
    """
    x = x - np.mean(x)
    x = nk.signal_detrend(x)

    b, a = butter(4, fc / (fs / 2), btype="low")

    x_filt = filtfilt(b, a, x)

    N = len(x_filt)
    X = fft(x_filt)

    mag = np.abs(X) / N
    mag = mag[: N // 2]
    f = np.linspace(0, fs / 2, N // 2, endpoint=False)
    phases = np.angle(X[: N // 2])

    return X, f, mag, phases


def find_most_significant_frequencies(
    X: NDArray[np.complex_],
    f: NDArray[np.floating],
    mag: NDArray[np.floating],
    min_sep: float = 0.2,
    n: int = 3,
) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
    """
    Select the top‐n frequencies by magnitude, ensuring minimum separation.

    Parameters
    ----------
    X : NDArray[np.complex_]
        FFT spectrum.
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


def filter_frequencies(
    X: NDArray[np.complex_], idxs: NDArray[np.integer]
) -> NDArray[np.complex_]:
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
        FFT spectrum with only the selected bins non‐zero.
    """
    X_filtered = np.zeros_like(X)
    X_filtered[idxs] = X[idxs]
    X_filtered[-idxs] = X[-idxs]
    return X_filtered


def reconstruct_signal(
    X: NDArray[np.complex_], mean_value: float
) -> NDArray[np.floating]:
    """
    Inverse FFT to recover a real‐valued signal, then add back the mean.

    Parameters
    ----------
    X : NDArray[np.complex_]
        (Possibly filtered) FFT spectrum.
    mean_value : float
        Mean of the original signal to re‐add.

    Returns
    -------
    reconstructed : NDArray[np.floating]
        Real part of the inverse FFT plus the mean.
    """
    reconstructed = ifft(X).real
    reconstructed += mean_value
    return reconstructed


def plot_reconstructed_signal(
    t: NDArray[np.floating], reconstructed: NDArray[np.floating]
) -> None:
    """
    Plot the reconstructed time‐series.

    Parameters
    ----------
    t : NDArray[np.floating]
        Time vector.
    reconstructed : NDArray[np.floating]
        Signal values to plot.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t, reconstructed, color="blue", label="reconstructed")
    plt.title("Reconstructed Sine Waves from the Most Significant Frequencies")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mmHg]")
    plt.legend()
