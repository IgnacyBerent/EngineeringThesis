from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, SubjectData

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
    def load_single_subject_raw_data(self, subject_directory: Path) -> SubjectData:
        pass

    def load_all_raw_data(self) -> list[SubjectData]:
        raw_data: list[SubjectData] = []
        for subject_directory in self._data_directory.iterdir():
            if not subject_directory.is_dir():
                logger.debug(f'Skipping folder: {subject_directory}')
                continue
            try:
                subject_raw_data = self.load_single_subject_raw_data(subject_directory)
            except CBFileError as e:
                logger.warning(f'Failed to load all columns in {subject_directory}\n {e}')
            except FileNotFoundError as e:
                logger.warning(f'Failed to find all cb files in {subject_directory}\n {e}')
            except UnicodeDecodeError as e:
                logger.warning(f'CSV decoding error in {subject_directory}\n {e}')
            except Exception as e:  # noqa: BLE001
                logger.error(f'Unexpected exception for {subject_directory}\n {e}')

            if subject_raw_data is not None:
                raw_data.append(subject_raw_data)

        logger.info('Loaded all subjects')
        return raw_data

    def load_single_condition_csv_file(self, cb_file_path: Path) -> ArrayDataDict:
        if not cb_file_path.exists():
            raise FileNotFoundError(f'File: {cb_file_path}')
        try:
            subject_df = pd.read_csv(cb_file_path, sep=self._csv_separator, decimal=self._csv_decimal)
            return {
                field_name: cast(NDArray[np.floating], subject_df[csv_column_name].values)
                for field_name, csv_column_name in self._csv_columns.items()
            }
        except UnicodeDecodeError as e:
            raise Exception(f'File: {cb_file_path}\n{e}') from e
        except (ValueError, KeyError) as e:
            raise CBFileError(f'File: {cb_file_path}\n{e}') from e

    @staticmethod
    def _get_subject_id(subject_directory) -> int:
        return int(str(subject_directory).split('_')[-1])
