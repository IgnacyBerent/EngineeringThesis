from pathlib import Path
from typing import override

from src.common.constants import BREATHING_DATA_DIRECTORY_PATH, CB_FILE, SignalColumns
from src.common.logger import logger
from src.common.mytypes import BaroreflexRawData, PatientData
from src.data_process.loaders.data_loader import CBFileError, DataLoader


class BaroreflexDataLoader(DataLoader[BaroreflexRawData]):
    def __init__(self) -> None:
        super().__init__(BaroreflexRawData)

    @property
    @override
    def _data_directory(self) -> Path:
        return BREATHING_DATA_DIRECTORY_PATH

    @property
    @override
    def _csv_columns(self) -> list[str]:
        return [SignalColumns.ABP, SignalColumns.ETCO2]

    @override
    def load_single_patient_raw_data(self, patient_directory: Path) -> PatientData | None:
        patient_id = self._get_patient_id(patient_directory)
        try:
            return PatientData(
                id=patient_id,
                cb_6b=self.load_single_cb_csv_file(patient_directory / CB_FILE.B6),
                baseline=self.load_single_cb_csv_file(patient_directory / CB_FILE.BASELINE),
            )
        except CBFileError as e:
            logger.warning(f'Failed to load all columns for patient {patient_id}: \n {e}')
        except FileNotFoundError as e:
            logger.warning(f'Failed to find all cb files for patient {patient_id}: \n {e}')
