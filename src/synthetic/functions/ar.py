import numpy as np
from src.common.mytypes import FloatArray
from src.synthetic.functions.utils import generate_normal_gaussian_noise


def generate_bivariate_ar_open_loop(
    n: int, a: tuple[float, float, float], e: float, n_skip: int = 2000
) -> dict[str, FloatArray]:
    """
    Generates a Bivariate Autoregressive process (AR(1)) with UNIDIRECTIONAL
    (open-loop) causality: X -> Y.

    Equations:
    X_t = a11 * X_{t-1} + epsilon_X,t
    Y_t = a22 * Y_{t-1} + a21 * X_{t-1} + epsilon_Y,t
    Causality strength is set by a21 (X -> Y).

    Parameters
    ----------
    n : int
        The desired length of the resulting time series.
    a : tuple[float, float, float]
        Autoregressive and coupling coefficients (a11, a22, a21).
    e : float
        Standard deviation of the Gaussian white noise (epsilon).
    n_skip : int, optional
        Number of initial transient points to discard, by default 2000.

    Returns
    -------
    dict[str, FloatArray]
    """
    n_plus_skip = n + n_skip
    X = np.zeros(n_plus_skip)
    Y = np.zeros(n_plus_skip)

    epsilon_X = generate_normal_gaussian_noise(n_plus_skip, e)
    epsilon_Y = generate_normal_gaussian_noise(n_plus_skip, e)

    X[0] = epsilon_X[0]
    Y[0] = epsilon_Y[0]

    a11, a22, a21 = a

    for t in range(1, n_plus_skip):
        X[t] = a11 * X[t - 1] + epsilon_X[t]
        Y[t] = a22 * Y[t - 1] + a21 * X[t - 1] + epsilon_Y[t]

    return {'x': X[n_skip:], 'y': Y[n_skip:]}


def generate_bivariate_ar_closed_loop(
    n: int, a: tuple[float, float, float, float], e: float, n_skip: int = 2000
) -> dict[str, FloatArray]:
    """
    Generates a Bivariate Autoregressive process (AR(1)) with BIDIRECTIONAL
    (closed-loop) causality: X <-> Y.

    Equations:
    X_t = a11 * X_{t-1} + a12 * Y_{t-1} + epsilon_X,t
    Y_t = a22 * Y_{t-1} + a21 * X_{t-1} + epsilon_Y,t
    Causality is set by a12 (Y -> X) and a21 (X -> Y).

    Parameters
    ----------
    n : int
        The desired length of the resulting time series.
    a : tuple[float, float, float, float]
        Autoregressive and coupling coefficients (a11, a22, a12, a21).
    e : float
        Standard deviation of the Gaussian white noise (epsilon).
    n_skip : int, optional
        Number of initial transient points to discard, by default 2000.

    Returns
    -------
    dict[str, FloatArray]
    """
    n_plus_skip = n + n_skip
    X = np.zeros(n_plus_skip)
    Y = np.zeros(n_plus_skip)

    epsilon_X = generate_normal_gaussian_noise(n_plus_skip, e)
    epsilon_Y = generate_normal_gaussian_noise(n_plus_skip, e)

    X[0] = epsilon_X[0]
    Y[0] = epsilon_Y[0]

    a11, a22, a12, a21 = a

    for t in range(1, n_plus_skip):
        X[t] = a11 * X[t - 1] + a12 * Y[t - 1] + epsilon_X[t]
        Y[t] = a22 * Y[t - 1] + a21 * X[t - 1] + epsilon_Y[t]

    return {'x': X[n_skip:], 'y': Y[n_skip:]}


def generate_trivariate_ar(
    n: int, a: tuple[float, float, float, float, float, float], e: float, n_skip: int = 2000
) -> dict[str, FloatArray]:
    """
    Generates a Trivariate Autoregressive process (AR(1)) that models
    MEDIATED CAUSALITY.

    Z acts as a common driver for X and Y, and X also directly influences Y.

    Equations (Z is the common driver/modulator):
    Z_t = az * Z_{t-1} + epsilon_Z,t
    X_t = ax * X_{t-1} + axz * Z_{t-1} + epsilon_X,t
    Y_t = ay * Y_{t-1} + ayx * X_{t-1} + ayz * Z_{t-1} + epsilon_Y,t

    Parameters
    ----------
    n : int
        The desired length of the resulting time series.
    a : tuple[float, ...]
        Autoregressive and coupling coefficients (ax, ay, az, axz, ayz, ayx).
    e : float
        Standard deviation of the Gaussian white noise (epsilon).
    n_skip : int, optional
        Number of initial transient points to discard, by default 2000.

    Returns
    -------
    dict[str, FloatArray]
    """
    n_plus_skip = n + n_skip
    X = np.zeros(n_plus_skip)
    Y = np.zeros(n_plus_skip)
    Z = np.zeros(n_plus_skip)

    epsilon_X = generate_normal_gaussian_noise(n_plus_skip, e)
    epsilon_Y = generate_normal_gaussian_noise(n_plus_skip, e)
    epsilon_Z = generate_normal_gaussian_noise(n_plus_skip, e)

    X[0] = epsilon_X[0]
    Y[0] = epsilon_Y[0]
    Z[0] = epsilon_Z[0]

    ax, ay, az, axz, ayz, ayx = a

    for t in range(1, n_plus_skip):
        Z[t] = az * Z[t - 1] + epsilon_Z[t]
        X[t] = ax * X[t - 1] + axz * Z[t - 1] + epsilon_X[t]
        Y[t] = ay * Y[t - 1] + ayx * X[t - 1] + ayz * Z[t - 1] + epsilon_Y[t]

    return {'x': X[n_skip:], 'y': Y[n_skip:], 'z': Z[n_skip:]}
