import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import NDArray

from src.plots.constatns import AXIS_LABEL_FONT_SIZE, AXIS_TICKS_FONT_SIZE, DPI, SQUARE_FIG_SIZE, TITLE_FONT_SIZE


def plot_boxplot(data: dict[str, NDArray], title: str, y_label: str) -> None:
    labels = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=SQUARE_FIG_SIZE, dpi=DPI)
    plt.boxplot(values, label=labels, tick_labels=labels)
    plt.title(title, size=TITLE_FONT_SIZE)
    plt.xticks(size=AXIS_TICKS_FONT_SIZE)
    plt.yticks(size=AXIS_TICKS_FONT_SIZE)
    plt.ylabel(y_label, size=AXIS_LABEL_FONT_SIZE)
    plt.savefig(f'{title}.png')
    plt.show()


def plot_boxplot_w_posthoc(data: dict[str, NDArray], title: str, y_label: str, posthoc: pd.DataFrame) -> None:
    labels = list(data.keys())
    values = list(data.values())

    _, ax = plt.subplots(figsize=SQUARE_FIG_SIZE, dpi=DPI)
    _ = ax.boxplot(values, patch_artist=True, label=labels, tick_labels=labels)

    ax.set_title(title, size=TITLE_FONT_SIZE)
    ax.set_ylabel(y_label, size=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='x', labelsize=AXIS_TICKS_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=AXIS_TICKS_FONT_SIZE)

    y_max = max([max(v) for v in values])
    h = 0.05 * y_max
    num = 1

    for _, row in posthoc.iterrows():
        if row['p-corr'] < 0.05:
            i = labels.index(row['A'])  # type: ignore
            j = labels.index(row['B'])  # type: ignore
            x1, x2 = i + 1, j + 1
            y = y_max + h * num
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
            if row['p-corr'] < 0.001:
                stars = '***'
            elif row['p-corr'] < 0.01:
                stars = '**'
            else:
                stars = '*'
            ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', color='k')
            num += 1

    plt.savefig(f'{title}.png')
    plt.show()
