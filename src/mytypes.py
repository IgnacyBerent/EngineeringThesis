from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FFT_Result:
    X: NDArray[np.complex128]
    f: NDArray[np.floating]
    mag: NDArray[np.floating]
    phases: NDArray[np.floating]
