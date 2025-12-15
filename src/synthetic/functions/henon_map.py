import numpy as np
from src.common.mytypes import FloatArray


def generate_coupled_henon_map(
    n: int,
    a: float = 1.4,
    b: float = 0.3,
    gamma: float = 0.2,
    n_skip: int = 2000,
    seed: int | None = None,
) -> dict[str, FloatArray]:
    """
    Generates a bounded, unidirectionally coupled Hénon map (X -> Y)
    using canonical diffusive coupling.
    """

    if not 0.0 <= gamma <= 1.0:
        raise ValueError('Gamma must be in [0, 1].')

    if seed is not None:
        np.random.seed(seed)

    n_total = n + n_skip

    X = np.zeros(n_total)
    Y = np.zeros(n_total)

    # Generic initial conditions (avoid nongeneric orbits)
    X[0:2] = np.random.uniform(-0.5, 0.5, 2)
    Y[0:2] = np.random.uniform(-0.5, 0.5, 2)

    for t in range(2, n_total):
        fx = 1.0 - a * X[t - 1] ** 2 + b * X[t - 2]
        fy = 1.0 - a * Y[t - 1] ** 2 + b * Y[t - 2]

        X[t] = fx
        Y[t] = (1.0 - gamma) * fy + gamma * fx

    return {
        'x': X[n_skip:],
        'y': Y[n_skip:],
    }
