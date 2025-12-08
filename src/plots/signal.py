import matplotlib.pyplot as plt
import numpy as np

from src.common.constants import SAMPLING_FREQUENCY
from src.common.mytypes import FloatArray
from src.plots.constatns import DEFAULT_SIGNAL_COLOR, RECTANGLE_FIG_SIZE


def plot_single_signal(signal: FloatArray, ylabel: str, title: str) -> None:
    plt.figure(figsize=RECTANGLE_FIG_SIZE)
    x_data = np.arange(len(signal)) / SAMPLING_FREQUENCY
    plt.plot(
        x_data,
        signal,
        linestyle='-',
        color=DEFAULT_SIGNAL_COLOR,
    )

    # 3. Finalize plot
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.show()


def plot_two_signals_shared_x(
    signal_top: FloatArray, ylabel_top: str, signal_bottom: FloatArray, ylabel_bottom: str, title: str
) -> None:
    assert len(signal_top) == len(signal_bottom), ValueError('Signals have to be the same length')

    x_data = np.arange(len(signal_top)) / SAMPLING_FREQUENCY
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
