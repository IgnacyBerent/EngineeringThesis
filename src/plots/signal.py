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
    signal: FloatArray, peaks: NDArray[np.floating], ylabel: str, title: str, time_unit: TimeUnit | None
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


def plot_two_signals_shared_x(
    signal_top: FloatArray, ylabel_top: str, signal_bottom: FloatArray, ylabel_bottom: str, title: str
) -> None:
    assert len(signal_top) == len(signal_bottom), ValueError('Signals have to be the same length')

    indices = _get_indices(signal_top)
    x_data = _indices_to_time(indices, TimeUnit.S)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=RECTANGLE_FIG_SIZE,
    )

    ax1.plot(
        x_data,
        signal_top,
        linestyle='-',
        color=DEFAULT_SIGNAL_COLOR,
    )
    ax1.set_ylabel(ylabel_top)
    ax1.set_title(title)
    ax2.plot(
        x_data,
        signal_bottom,
        linestyle='-',
        color=DEFAULT_SIGNAL_COLOR,
    )
    ax2.set_ylabel(ylabel_bottom)
    ax2.set_xlabel('Time [s]')
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
