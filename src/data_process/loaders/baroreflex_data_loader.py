from pathlib import Path
from typing import override

from src.common.constants import BREATHING_DATA_DIRECTORY_PATH, CB_FILE_TYPE, SignalColumns
from src.common.mytypes import SubjectData
from src.data_process.loaders.data_loader import DataLoader


class BaroreflexDataLoader(DataLoader):
    @property
    @override
    def _data_directory(self) -> Path:
        return BREATHING_DATA_DIRECTORY_PATH

    @property
    @override
    def _csv_columns(self) -> dict[str, str]:
        return {'abp': SignalColumns.ABP, 'etco2': SignalColumns.ETCO2}

    @override
    def load_single_subject_raw_data(self, subject_directory: Path) -> SubjectData:
        return {
            'id': self._get_subject_id(subject_directory),
            CB_FILE_TYPE.B6: self.load_single_condition_csv_file(subject_directory / CB_FILE_TYPE.B6.csv),
            CB_FILE_TYPE.B10: self.load_single_condition_csv_file(subject_directory / CB_FILE_TYPE.B10.csv),
            CB_FILE_TYPE.B15: self.load_single_condition_csv_file(subject_directory / CB_FILE_TYPE.B15.csv),
            CB_FILE_TYPE.BASELINE: self.load_single_condition_csv_file(subject_directory / CB_FILE_TYPE.BASELINE.csv),
        }
