import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from src.data_process.entropy import DVPartition
from src.plots.constatns import DPI, SQUARE_FIG_SIZE

_COLOR_MAP = 'viridis'
_ALPHA = 0.6
_SUB_ALPHA = 0.2
_SIZE = 10

_LABEL = 'Data (ranked)'
_COLOR_LABEL = 'N'

_TITLE = 'Darbellay-Vajda Adaptive Partitioning'


def plot_3d_partitions(
    partitions: list[DVPartition],
    X: NDArray[np.integer],
    Y: NDArray[np.integer],
    Z: NDArray[np.integer],
    xlabel: str,
    ylabel: str,
    zlabel: str,
) -> None:
    """
    Plot the 3D partitioning of the data.

    Parameters:
    partitions: List of partition dictionaries. Each dictionary should contain:
        - 'mins': np.array of minimum coordinates for the partition box.
        - 'maxs': np.array of maximum coordinates for the partition box.
        - 'N': Number of points in the partition.
    X, Y, Z: 1D numpy arrays representing the ranked random variables.
    """
    fig = plt.figure(figsize=SQUARE_FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, s=_SIZE, alpha=_ALPHA, label=_LABEL)  # type: ignore

    Ns = [int(part['N']) for part in partitions]
    norm = colors.Normalize(vmin=min(Ns), vmax=max(Ns))
    cmap = matplotlib.colormaps[_COLOR_MAP]

    for part in partitions:
        xmin, ymin, zmin = map(int, part['mins'])
        xmax, ymax, zmax = map(int, part['maxs'])
        n = int(part['N'])
        corners = np.array(list(itertools.product([xmin, xmax], [ymin, ymax], [zmin, zmax])))
        faces = [
            [corners[0], corners[1], corners[3], corners[2]],  # bottom
            [corners[4], corners[5], corners[7], corners[6]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[0], corners[2], corners[6], corners[4]],  # left
            [corners[1], corners[3], corners[7], corners[5]],  # right
        ]

        poly = Poly3DCollection(faces, alpha=_SUB_ALPHA, linewidths=1)
        face_color = cmap(norm(n))
        poly.set_facecolor(face_color)
        ax.add_collection3d(poly)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=_COLOR_LABEL)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f'{_TITLE} (3D)')
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    plt.savefig(f'{_TITLE} (3D).png')
    plt.show()


def plot_2d_partitions(
    partitions: list[DVPartition], X: NDArray[np.integer], Y: NDArray[np.integer], xlabel: str, ylabel: str
) -> None:
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE, dpi=DPI)
    ax.scatter(X, Y, s=_SIZE, alpha=_ALPHA, label=_LABEL)

    Ns = [int(part['N']) for part in partitions]
    norm = colors.Normalize(vmin=min(Ns), vmax=max(Ns))
    cmap = matplotlib.colormaps[_COLOR_MAP]

    for part in partitions:
        xmin, ymin = map(int, part['mins'])
        xmax, ymax = map(int, part['maxs'])
        n = int(part['N'])

        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=True,
                color=cmap(norm(n)),
                alpha=_SUB_ALPHA,
            )
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=_COLOR_LABEL)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{_TITLE} (2D)')
    ax.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f'{_TITLE} (2D).png')
    plt.show()
