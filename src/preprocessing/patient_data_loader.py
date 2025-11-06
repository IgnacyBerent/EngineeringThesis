from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.common.constants import (
    BREATHING_DATA_DIRECTORY_PATH,
    CB_6,
    CB_BASELINE,
    PATIENT_DATA_FILE_TYPES,
    SignalColumnNames,
)
from src.common.logger import logger
from src.common.mytypes import BreathingRawData, PatientRawData

_DEFAULT_CSV_DECIMAL = ','
_DEFAULT_CSV_SEPARATOR = ';'


class PatientDataLoadingError(Exception):
    pass


class PatientDataLoader:
    def __init__(
        self,
        data_directory: Path = BREATHING_DATA_DIRECTORY_PATH,
        csv_decimal: str = _DEFAULT_CSV_DECIMAL,
        csv_sep: str = _DEFAULT_CSV_SEPARATOR,
    ) -> None:
        self._data_directory = data_directory
        self._csv_decimal = csv_decimal
        self._csv_sep = csv_sep

    def load_all_patients_raw_data(self) -> list[PatientRawData]:
        patients_raw_data_list: list[PatientRawData] = []
        for patient_directory in self._data_directory.iterdir():
            if not patient_directory.is_dir():
                logger.debug(f'Skipping folder: {patient_directory}')
                continue

            patient_raw_data = self.load_single_patient_raw_data(patient_directory)
            if patient_raw_data is not None:
                patients_raw_data_list.append(patient_raw_data)

        return patients_raw_data_list

    def load_single_patient_raw_data(self, patient_directory: Path) -> PatientRawData | None:
        patient_id = int(str(patient_directory).split('_')[-1])
        logger.info(f'Loading data for patient {patient_id}')

        patient_raw_data_dict: dict[str, BreathingRawData] = {}
        for patient_breathing_data_path in patient_directory.iterdir():
            if not self._is_valid_file(patient_breathing_data_path):
                logger.warning(f'Invalid file: {patient_breathing_data_path}')
                continue

            try:
                patient_brathing_data = self._load_patient_breathing_data(patient_breathing_data_path)
            except PatientDataLoadingError as e:
                logger.warning(f'Failulre on patient {patient_id}, skipping to load his data:\n {e}')
                return None

            if CB_6 in str(patient_breathing_data_path):
                patient_raw_data_dict['cb_6b'] = patient_brathing_data
            elif CB_BASELINE in str(patient_breathing_data_path):
                patient_raw_data_dict['baseline'] = patient_brathing_data

        return PatientRawData(id=patient_id, **patient_raw_data_dict)

    @staticmethod
    def _is_valid_file(patient_breathing_data_path: Path) -> bool:
        file_str = str(patient_breathing_data_path)
        return any(word in file_str for word in PATIENT_DATA_FILE_TYPES)

    def _load_patient_breathing_data(self, patient_breathing_data_path: Path) -> BreathingRawData:
        patient_df = pd.read_csv(patient_breathing_data_path, sep=self._csv_sep, decimal=self._csv_decimal)
        try:
            return BreathingRawData(
                abp=cast(NDArray[np.floating], patient_df[SignalColumnNames.ABP].values),
                rr=cast(NDArray[np.floating], patient_df[SignalColumnNames.RR].values),
                etco2=cast(NDArray[np.floating], patient_df[SignalColumnNames.ETCO2].values),
            )
        except (ValueError, KeyError) as e:
            raise PatientDataLoadingError from e
