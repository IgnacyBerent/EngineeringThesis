import numpy as np
from src.common.mytypes import FloatArray


def generate_normal_gaussian_noise(n: int, e: float) -> FloatArray:
    return np.random.normal(0, e, n)

