from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.common.logger import logger
from src.common.mytypes import CBData, PatientData

_DEFAULT_CSV_DECIMAL = ','
_DEFAULT_CSV_SEPARATOR = ';'


class CBFileError(Exception):
    pass


class DataLoader[DataOutType: CBData](ABC):
    _data_class: type[DataOutType]

    def __init__(self, data_class: type[DataOutType]) -> None:
        self._data_class = data_class

    @property
    @abstractmethod
    def _data_directory(self) -> Path:
        pass

    @property
    @abstractmethod
    def _csv_columns(self) -> list[str]:
        pass

    @property
    def _csv_separator(self) -> str:
        return _DEFAULT_CSV_SEPARATOR

    @property
    def _csv_decimal(self) -> str:
        return _DEFAULT_CSV_DECIMAL

    @abstractmethod
    def load_single_patient_raw_data(self, patient_directory: Path) -> PatientData | None:
        pass

    def load_all_patient_raw_data(self) -> list[PatientData]:
        patients_raw_data_list: list[PatientData] = []
        for patient_directory in self._data_directory.iterdir():
            if not patient_directory.is_dir():
                logger.debug(f'Skipping folder: {patient_directory}')
                continue

            patient_raw_data = self.load_single_patient_raw_data(patient_directory)
            if patient_raw_data is not None:
                patients_raw_data_list.append(patient_raw_data)

        logger.info('Loaded all patients')
        return patients_raw_data_list

    def load_single_cb_csv_file(self, cb_file_path: Path) -> DataOutType:
        if not cb_file_path.exists():
            raise FileNotFoundError
        patient_df = pd.read_csv(cb_file_path, sep=self._csv_separator, decimal=self._csv_decimal)
        try:
            cb_data_csv = {
                field_name: cast(NDArray[np.floating], patient_df[csv_column_name].values)
                for field_name, csv_column_name in zip(
                    self._data_class.get_field_names(), self._csv_columns, strict=False
                )
            }

            return self._data_class(**cb_data_csv)
        except (ValueError, KeyError) as e:
            raise CBFileError from e

    @staticmethod
    def _get_patient_id(patient_directory) -> int:
        return int(str(patient_directory).split('_')[-1])
