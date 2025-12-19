import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from numpy.typing import NDArray

from src.plots.constatns import DPI, SQUARE_FIG_SIZE


def plot_boxplot(data: dict[str, NDArray], title: str, y_label: str) -> None:
    labels = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=SQUARE_FIG_SIZE, dpi=DPI)
    plt.boxplot(values, label=labels, tick_labels=labels)
    plt.title(title)
    plt.ylabel(y_label + ' [bits]')
    plt.savefig(f'{title}.png')
    plt.show()


def plot_boxplot_w_posthoc(data: dict[str, NDArray], title: str, y_label: str, posthoc: pd.DataFrame) -> None:
    labels = list(data.keys())
    values = list(data.values())
    [print(f'{label}: {np.mean(value):.3f} +- {np.std(value):.3f}') for label, value in list(data.items())]

    _, ax = plt.subplots(figsize=SQUARE_FIG_SIZE, dpi=DPI)
    _ = ax.boxplot(values, patch_artist=True, label=labels, tick_labels=labels)

    ax.set_title(title)
    ax.set_ylabel(y_label + ' [bits]')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    y_max = max([max(v) for v in values])
    h = 0.05 * y_max
    num = 1

    for _, row in posthoc.iterrows():
        p_corr = row.get('p-corr')
        if p_corr is not None and p_corr < 0.05:
            i = labels.index(row['A'])  # type: ignore
            j = labels.index(row['B'])  # type: ignore
            x1, x2 = i + 1, j + 1
            y = y_max + h * num
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
            if p_corr < 0.001:
                stars = '***'
            elif p_corr < 0.01:
                stars = '**'
            else:
                stars = '*'
            ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', color='k')
            num += 1

    plt.savefig(f'results/{title}.png')
    plt.show()


def plot_paired_boxplot(data: pd.DataFrame, field: str, within: str, subject: str, order: list[str]) -> None:
    plt.figure(figsize=SQUARE_FIG_SIZE, dpi=DPI)
    pg.plot_paired(
        data=data,
        dv=field,
        within=within,
        subject=subject,
        boxplot=True,
        orient='v',
        order=order,
    )
    plt.title(f'{field}')
    plt.ylabel(f'{field} [bits]')
    plt.savefig(f'{field} paired')
    plt.show()
