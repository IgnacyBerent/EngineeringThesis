import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.plots.constatns import DEFAULT_SIGNAL_COLOR, DPI, RECTANGLE_FIG_SIZE

bottom = 627.4
top = 1221.0
v50 = 124.4
slope_at_v50 = 16.12

k = (top - bottom) / (4 * slope_at_v50)


def _sigmoid(x: NDArray[np.number]) -> NDArray[np.floating]:
    return bottom + (top - bottom) / (1 + np.exp((v50 - x) / k))


def plot_baroreflex() -> None:
    rng = np.random.default_rng(1)
    x_data = np.array([88, 92, 98, 103, 108, 113, 118, 122, 127, 133, 138, 145, 151, 158, 163, 175])
    y_ideal = _sigmoid(x_data)
    noise = rng.laplace(0, 25, size=len(x_data))
    y_data = y_ideal + noise

    y_err = np.array([50, 60, 80, 100, 120, 130, 150, 180, 200, 220, 210, 180, 150, 120, 100, 80])

    x_fit = np.linspace(80, 180, 500)
    y_fit = _sigmoid(x_fit)

    # 7. Recreate the Plot
    plt.figure(figsize=RECTANGLE_FIG_SIZE, dpi=DPI)

    plt.plot(x_fit, y_fit, color=DEFAULT_SIGNAL_COLOR, linewidth=2.5, zorder=1)
    plt.errorbar(
        x_data,
        y_data,
        yerr=y_err,
        fmt='o',
        color='black',
        markerfacecolor='white',
        markeredgecolor='black',
        markersize=9,
        capsize=0,
        elinewidth=1.5,
        mew=1.5,
        zorder=2,
    )

    plt.xlabel('Systolic Arterial Pressure [mmHg]')
    plt.ylabel('Heart Period [ms]')
    plt.xlim(75, 185)
    plt.ylim(550, 1400)
    plt.title('Baroreflex Curve')
    plt.tight_layout()
    plt.savefig('baroreflex-curve.png')
    plt.show()
