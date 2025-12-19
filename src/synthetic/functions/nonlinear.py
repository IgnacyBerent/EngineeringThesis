import numpy as np

from src.common.mytypes import FloatArray
from src.synthetic.common import DEFAULT_N_SKIP


def generate_nonlinear_bivariate_process(
    length: int,
    seed: int,
    b: float,
    n_skip: int = DEFAULT_N_SKIP,
) -> dict[str, FloatArray]:
    """
    Generates a nonlinear bivariate process based on Lee et al.
    x_n = s_xn + noise_xn
    y_n = (b * x_{n-tau})^2 + noise_yn
    """
    rng = np.random.default_rng(seed)
    n_total = length + n_skip

    sx = rng.normal(10, 1, n_total)

    noise_x = rng.laplace(0, 1, n_total)
    noise_y = rng.laplace(0, 1, n_total)

    X = sx + noise_x

    Y = np.zeros(n_total)
    for n in range(1, n_total):
        Y[n] = (b * X[n - 1]) ** 2 + noise_y[n]

    return {
        'x': X[n_skip:],
        'y': Y[n_skip:],
    }
