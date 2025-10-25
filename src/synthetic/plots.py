import numpy as np
import matplotlib.pyplot as plt

from src.synthetic.synthetic import reconstruct_signal, filter_frequencies


def plot_fft(f, mag, fc=None, freqs=None, title=None):
    plt.figure(figsize=(12, 6))
    if fc is not None:
        plt.plot(f[f <= fc], mag[f <= fc])
    else:
        plt.plot(f, mag)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    # Highlight the most significant frequencies
    if freqs is not None:
        for freq in freqs:
            plt.axvline(
                freq, color="red", linestyle="--", label=f"Sig. freq: {freq:.2f} Hz"
            )
    plt.title(title if title else "FFT Magnitude Spectrum")
    plt.tight_layout()
    plt.show()


def plot_signal_with_reconstructed_sines(t, x, X, f, selected_indices):
    separate_sins = [
        reconstruct_signal(filter_frequencies(X, idx), 0) for idx in selected_indices
    ]
    plt.figure(figsize=(12, 6))

    # original in black, reconstructed in blue
    plt.plot(t, x, color="black", label="original")

    # generate shades of yellow
    cmap = plt.get_cmap("YlOrBr", len(separate_sins) + 1)
    for i, (idx, sin) in enumerate(zip(selected_indices, separate_sins)):
        plt.plot(t, sin, color=cmap(i + 1), label=f"Sine {f[idx]:.2f} Hz")

    plt.title("Reconstructed Sine Waves from the most Significant Frequencies")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mmHg]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_signal_with_reconstructed(t, original, reconstructed):
    plt.figure(figsize=(12, 6))

    plt.plot(t, original, color="black", label="original")
    plt.plot(t, reconstructed, "--", color="blue", label="reconstructed")

    plt.title("Reconstructed Sine Waves from  the most Significant Frequencies")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mmHg]")
    plt.legend()
    plt.tight_layout()
    plt.show()
