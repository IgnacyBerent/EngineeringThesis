from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.common.mytypes import CBData, PatientData


class DataProcessor[DataInType: CBData, DataOutType: CBData](ABC):
    @abstractmethod
    def _process_single_cb(self, cb_raw_data: DataInType) -> DataOutType:
        pass

    def process(self, patient_raw_data: PatientData[DataInType]) -> PatientData[DataOutType]:
        return PatientData(
            id=patient_raw_data.id,
            **{
                cb_field_name: self._process_single_cb(cb_raw_data)
                for cb_field_name, cb_raw_data in patient_raw_data.get_fields().items()
            },
        )

    @staticmethod
    def _average_to_length(signal_to_shorten: NDArray[np.floating], target_length: int) -> NDArray[np.floating]:
        """
        shortens signal to a given length by spliting it to windows and averaging them out
        """
        signal_windowed = np.array_split(signal_to_shorten, target_length)
        return np.array([np.mean(window) for window in signal_windowed])
