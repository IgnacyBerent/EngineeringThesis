import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.data_process.fft import filter_frequencies, reconstruct_signal
from src.plots.constatns import DPI, RECTANGLE_FIG_SIZE


def plot_fft(f, mag, fc=None, freqs=None, title=None) -> None:
    plt.figure(figsize=RECTANGLE_FIG_SIZE, dpi=DPI)
    if fc is not None:
        plt.plot(f[f <= fc], mag[f <= fc])
    else:
        plt.plot(f, mag)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    # Highlight the most significant frequencies
    if freqs is not None:
        for freq in freqs:
            plt.axvline(freq, color='red', linestyle='--', label=f'Sig. freq: {freq:.2f} Hz')
    plt.title(title if title else 'FFT Magnitude Spectrum')
    plt.tight_layout()
    plt.show()


def plot_signal_with_reconstructed_sines(t, x, X, f, selected_indices) -> None:
    separate_sins = [reconstruct_signal(filter_frequencies(X, idx), 0) for idx in selected_indices]
    plt.figure(figsize=RECTANGLE_FIG_SIZE, dpi=DPI)

    # original in black, reconstructed in blue
    plt.plot(t, x, color='black', label='original')

    # generate shades of yellow
    cmap = plt.get_cmap('YlOrBr', len(separate_sins) + 1)
    for i, (idx, sin) in enumerate(zip(selected_indices, separate_sins, strict=False)):
        plt.plot(t, sin, color=cmap(i + 1), label=f'Sine {f[idx]:.2f} Hz')

    plt.title('Reconstructed Sine Waves from the most Significant Frequencies')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mmHg]')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_signal_with_reconstructed(t, original, reconstructed) -> None:
    plt.figure(figsize=RECTANGLE_FIG_SIZE, dpi=DPI)

    plt.plot(t, original, color='black', label='original')
    plt.plot(t, reconstructed, '--', color='blue', label='reconstructed')

    plt.title('Reconstructed Sine Waves from the most Significant Frequencies')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mmHg]')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reconstructed_signal(t: NDArray[np.floating], reconstructed: NDArray[np.floating]) -> None:
    plt.figure(figsize=RECTANGLE_FIG_SIZE, dpi=DPI)
    plt.plot(t, reconstructed, color='blue', label='reconstructed')
    plt.title('Reconstructed Sine Waves from the Most Significant Frequencies')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mmHg]')
    plt.legend()
