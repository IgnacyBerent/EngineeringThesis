from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, PatientData

_DEFAULT_CSV_DECIMAL = ','
_DEFAULT_CSV_SEPARATOR = ';'


class CBFileError(Exception):
    pass


class DataLoader(ABC):
    @property
    @abstractmethod
    def _data_directory(self) -> Path:
        pass

    @property
    @abstractmethod
    def _csv_columns(self) -> dict[str, str]:
        pass

    @property
    def _csv_separator(self) -> str:
        return _DEFAULT_CSV_SEPARATOR

    @property
    def _csv_decimal(self) -> str:
        return _DEFAULT_CSV_DECIMAL

    @abstractmethod
    def load_single_patient_raw_data(self, patient_directory: Path) -> PatientData:
        pass

    def load_all_patient_raw_data(self) -> list[PatientData]:
        patients_raw_data_list: list[PatientData] = []
        for patient_directory in self._data_directory.iterdir():
            if not patient_directory.is_dir():
                logger.debug(f'Skipping folder: {patient_directory}')
                continue
            try:
                patient_raw_data = self.load_single_patient_raw_data(patient_directory)
            except CBFileError as e:
                logger.warning(f'Failed to load all columns in {patient_directory}\n {e}')
            except FileNotFoundError as e:
                logger.warning(f'Failed to find all cb files in {patient_directory}\n {e}')
            except UnicodeDecodeError as e:
                logger.warning(f'CSV decoding error in {patient_directory}\n {e}')
            except Exception as e:  # noqa: BLE001
                logger.error(f'Unexpected exception for {patient_directory}\n {e}')

            if patient_raw_data is not None:
                patients_raw_data_list.append(patient_raw_data)

        logger.info('Loaded all patients')
        return patients_raw_data_list

    def load_single_cb_csv_file(self, cb_file_path: Path) -> ArrayDataDict:
        if not cb_file_path.exists():
            raise FileNotFoundError(f'File: {cb_file_path}')
        try:
            patient_df = pd.read_csv(cb_file_path, sep=self._csv_separator, decimal=self._csv_decimal)
            return {
                field_name: cast(NDArray[np.floating], patient_df[csv_column_name].values)
                for field_name, csv_column_name in self._csv_columns.items()
            }
        except UnicodeDecodeError as e:
            raise Exception(f'File: {cb_file_path}\n{e}') from e
        except (ValueError, KeyError) as e:
            raise CBFileError(f'File: {cb_file_path}\n{e}') from e

    @staticmethod
    def _get_patient_id(patient_directory) -> int:
        return int(str(patient_directory).split('_')[-1])
