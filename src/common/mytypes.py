from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray


@dataclass
class FFT_Result:
    X: NDArray[np.complex128]
    f: NDArray[np.floating]
    mag: NDArray[np.floating]
    phases: NDArray[np.floating]


@dataclass
class CBData:
    @classmethod
    def get_field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


@dataclass
class BaroreflexRawData(CBData):
    abp: NDArray[np.floating]
    etco2: NDArray[np.floating]


@dataclass
class BaroreflexProcessedData(CBData):
    sap: NDArray[np.floating]
    hp: NDArray[np.floating]
    etco2: NDArray[np.floating]


@dataclass
class PatientData[T: CBData]:
    id: int
    baseline: T | None = None
    cb_6b: T | None = None
    cb_10b: T | None = None
    cb_15b: T | None = None

    def get_fields(self) -> dict[str, T]:
        return {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != 'id' and getattr(self, f.name) is not None
        }
