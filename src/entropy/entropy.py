from typing import cast

import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray


def shannon_entropy(x: NDArray[np.floating]) -> float:
    e, _ = nk.entropy_shannon(x, base=np.e)  # type: ignore
    return cast(float, e)


def joint_entropy(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    je, _ = nk.entropy_shannon_joint(
        x=x,
        y=y,
        base=np.e,  # type: ignore
    )
    return cast(float, je)


def conditional_entropy(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    # Calculate joint entropy H(X, Y)
    je_xy = joint_entropy(x, y)

    # Calculate entropy H(Y)
    e_y = shannon_entropy(y)

    # Conditional entropy H(X|Y) = H(X, Y) - H(Y)
    return je_xy - e_y
