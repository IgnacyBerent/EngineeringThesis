import numpy as np

from src.common.mytypes import FloatArray
from src.synthetic.common import DEFAULT_N_SKIP


def generate_bivariate_ar(
    length: int,
    a: float,
    seed: int,
    snr: int | None = None,
    n_skip: int = DEFAULT_N_SKIP,
) -> dict[str, FloatArray]:
    """
    Generates a linear bivariate process based on Faes et al.
    x_n = -0.5x_{n-1} + e_xn
    y_n = -0.5y_{n-1} + ax_{n-1} + e_yn
    """
    rng = np.random.default_rng(seed)
    n_total = length + n_skip

    X = np.zeros(n_total)
    Y = np.zeros(n_total)

    ex = rng.standard_normal(n_total)
    ey = rng.standard_normal(n_total)

    for n in range(1, n_total):
        X[n] = -0.5 * X[n - 1] + ex[n]
        Y[n] = -0.5 * Y[n - 1] + a * X[n - 1] + ey[n]

    X, Y = X[n_skip:], Y[n_skip:]

    if snr is None:
        return {'x': X, 'y': Y}

    power_x = np.var(X)
    power_y = np.var(Y)

    noise_var_x = power_x / (10 ** (snr / 10))
    noise_var_y = power_y / (10 ** (snr / 10))

    x_noisy = X + rng.normal(0, np.sqrt(noise_var_x), length)
    y_noisy = Y + rng.normal(0, np.sqrt(noise_var_y), length)

    return {'x': x_noisy, 'y': y_noisy}


def generate_trivariate_ar(
    n: int, az: float, ax: float, seed: int, n_skip: int = DEFAULT_N_SKIP
) -> dict[str, FloatArray]:
    """
    Generates a Trivariate Autoregressive process (AR(1)) that models
    MEDIATED CAUSALITY.

    Z acts as a common driver for X and Y, and X also directly influences Y.

    Equations (Z is the common driver/modulator):
    Z_t = 0.8 * Z_{t-1} + e_Z,t
    X_t = 0.5 * X_{t-1} + az * Z_{t-1} + e_X,t
    Y_t = 0.5 * Y_{t-1} + ax * X_{t-1} + az * Z_{t-1} + e_Y,t

    Parameters
    ----------
    n : int
        The desired length of the resulting time series.
    a : tuple[float, ...]
        Autoregressive and coupling coefficients (ax, ay, az, axz, ayz, ayx).
    n_skip : int, optional
        Number of initial transient points to discard, by default 2000.

    Returns
    -------
    dict[str, FloatArray]
    """
    rng = np.random.default_rng(seed)
    n_plus_skip = n + n_skip
    X = np.zeros(n_plus_skip)
    Y = np.zeros(n_plus_skip)
    Z = np.zeros(n_plus_skip)

    epsilon_X = rng.normal(0, 1, n_plus_skip)
    epsilon_Y = rng.normal(0, 1, n_plus_skip)
    epsilon_Z = rng.normal(0, 1, n_plus_skip)

    X[0] = epsilon_X[0]
    Y[0] = epsilon_Y[0]
    Z[0] = epsilon_Z[0]

    for t in range(1, n_plus_skip):
        Z[t] = 0.8 * Z[t - 1] + epsilon_Z[t]
        X[t] = 0.5 * X[t - 1] + az * Z[t - 1] + epsilon_X[t]
        Y[t] = 0.5 * Y[t - 1] + ax * X[t - 1] + az * Z[t - 1] + epsilon_Y[t]

    return {'x': X[n_skip:], 'y': Y[n_skip:], 'z': Z[n_skip:]}
