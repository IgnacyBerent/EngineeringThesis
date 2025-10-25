import matplotlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from matplotlib import cm, colors


def plot_3d_partitions(partitions: list[dict], X, Y, Z):
    """
    Plot the 3D partitioning of the data.

    Parameters:
    partitions: List of partition dictionaries. Each dictionary should contain:
        - 'mins': np.array of minimum coordinates for the partition box.
        - 'maxs': np.array of maximum coordinates for the partition box.
        - 'N': Number of points in the partition.
    X, Y, Z: 1D numpy arrays representing the ranked random variables.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X, Y, Z, s=10, alpha=0.6, label="Data (ranked)")

    Ns = [int(part["N"]) for part in partitions]
    norm = colors.Normalize(vmin=min(Ns), vmax=max(Ns))
    cmap = matplotlib.colormaps["viridis"]

    for part in partitions:
        xmin, ymin, zmin = map(int, part["mins"])
        xmax, ymax, zmax = map(int, part["maxs"])
        n = int(part["N"])
        corners = np.array(
            list(itertools.product([xmin, xmax], [ymin, ymax], [zmin, zmax]))
        )
        faces = [
            [corners[0], corners[1], corners[3], corners[2]],  # bottom
            [corners[4], corners[5], corners[7], corners[6]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[0], corners[2], corners[6], corners[4]],  # left
            [corners[1], corners[3], corners[7], corners[5]],  # right
        ]

        poly = Poly3DCollection(faces, alpha=0.2, linewidths=1)
        face_color = cmap(norm(n))
        poly.set_facecolor(face_color)
        ax.add_collection3d(poly)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="N")

    ax.set_xlabel("X (ranked)")
    ax.set_ylabel("Y (ranked)")
    ax.set_zlabel("Z (ranked)")
    ax.set_title("Darbellay-Vajda Adaptive Partitioning (3D)")
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def plot_2d_partitions(partitions: list[dict], X, Y):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X, Y, s=10, alpha=0.6, label="Data (ranked)")

    Ns = [int(part["N"]) for part in partitions]
    norm = colors.Normalize(vmin=min(Ns), vmax=max(Ns))
    cmap = matplotlib.colormaps["viridis"]

    for part in partitions:
        xmin, ymin = map(int, part["mins"])
        xmax, ymax = map(int, part["maxs"])
        n = int(part["N"])

        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=True,
                color=cmap(norm(n)),
                alpha=0.2,
            )
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="N")

    ax.set_xlabel("X (ranked)")
    ax.set_ylabel("Y (ranked)")
    ax.set_title("Darbellay-Vajda Adaptive Partitioning (2D)")
    ax.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
