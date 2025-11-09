from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, FloatArray, PatientData


class DataProcessor(ABC):
    @abstractmethod
    def _process_single_cb(self, cb_raw_data: ArrayDataDict) -> ArrayDataDict:
        pass

    def process_all(self, patients_raw_data: list[PatientData]) -> list[PatientData]:
        processed_data = []
        for patient_raw_data in patients_raw_data:
            if processed_patient_data := self.process(patient_raw_data):
                processed_data.append(processed_patient_data)
        return processed_data

    def process(self, patient_raw_data: PatientData) -> PatientData | None:
        try:
            return {
                'id': (patient_raw_data.get('id', 404)),  # type: ignore
                **{
                    cb_field_name: self._process_single_cb(cast(ArrayDataDict, cb_raw_data))
                    for cb_field_name, cb_raw_data in patient_raw_data.items()
                    if cb_field_name != 'id'
                },
            }
        except ValueError:
            logger.error(f'Missmatching field names for: {patient_raw_data}')

    @staticmethod
    def _average_to_length(signal_to_shorten: FloatArray, target_length: int) -> FloatArray:
        """
        shortens signal to a given length by spliting it to windows and averaging them out
        """
        signal_windowed = np.array_split(signal_to_shorten, target_length)
        return np.array([np.mean(window) for window in signal_windowed])
