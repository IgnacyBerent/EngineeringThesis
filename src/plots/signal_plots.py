from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.common.constants import SAMPLING_FREQUENCY
from src.common.mytypes import FloatArray
from src.plots.constatns import DEFAULT_MARKER_COLOR, DEFAULT_SIGNAL_COLOR, RECTANGLE_FIG_SIZE


class TimeUnit(Enum):
    S = 1
    MS = 1000


def plot_single_signal(signal: FloatArray, ylabel: str, title: str, time_unit: TimeUnit | None = None) -> None:
    plt.figure(figsize=RECTANGLE_FIG_SIZE)
    indices = _get_indices(signal)
    x_data = _indices_to_time(indices, time_unit) if time_unit else indices
    x_label = _get_x_label(time_unit)
    plt.plot(
        x_data,
        signal,
        linestyle='-',
        color=DEFAULT_SIGNAL_COLOR,
    )

    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.show()


def plot_single_signal_with_peaks(
    signal: FloatArray, peaks: NDArray[np.integer], ylabel: str, title: str, time_unit: TimeUnit | None
) -> None:
    plt.figure(figsize=RECTANGLE_FIG_SIZE)
    indices = _get_indices(signal)
    x_data = _indices_to_time(indices, time_unit) if time_unit else indices
    x_label = _get_x_label(time_unit)
    plt.plot(
        x_data,
        signal,
        linestyle='-',
        color=DEFAULT_SIGNAL_COLOR,
    )
    plt.scatter(
        _indices_to_time(peaks, time_unit) if time_unit else peaks,
        [signal[peak] for peak in peaks if peak <= len(signal)],
        label='detected peaks',
        marker='o',
        color=DEFAULT_MARKER_COLOR,
    )

    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.png')
    plt.show()


def plot_multiple_signals_shared_x(
    signals: list[FloatArray], labels: list[str], title: str, time_unit: TimeUnit | None
) -> None:
    N = len(signals)
    assert N > 0, ValueError('The list of signals cannot be empty.')
    assert len(labels) == N, ValueError('The number of signals must equal the number of labels.')

    # Check if all signals are the same length
    first_signal_len = len(signals[0])
    for signal in signals:
        assert len(signal) == first_signal_len, ValueError('All signals must have the same length.')

    indices = _get_indices(signal)
    x_data = _indices_to_time(indices, time_unit) if time_unit else indices
    x_label = _get_x_label(time_unit)

    fig, axes = plt.subplots(
        nrows=N,
        sharex=True,
        figsize=RECTANGLE_FIG_SIZE,
    )

    if N == 1:
        axes = [axes]

    for i, (ax, signal, ylabel) in enumerate(zip(axes, signals, labels, strict=False)):
        # Plot the signal
        ax.plot(
            x_data,
            signal,
            linestyle='-',
            color=DEFAULT_SIGNAL_COLOR,
        )

        ax.set_ylabel(ylabel)

        if i == 0:
            ax.set_title(title)

        if i == N - 1:
            ax.set_xlabel(x_label)

    plt.tight_layout()
    fig.savefig(f'{title}.png')
    plt.show()


def _get_indices(signal: NDArray) -> NDArray:
    return np.arange(len(signal))


def _indices_to_time(indices: NDArray, unit: TimeUnit) -> NDArray:
    return indices / SAMPLING_FREQUENCY * unit.value


def _get_x_label(time_unit: TimeUnit | None) -> str:
    match time_unit:
        case TimeUnit.S:
            return 'Time [s]'
        case TimeUnit.MS:
            return 'Time [ms]'
        case _:
            return 'Index'
