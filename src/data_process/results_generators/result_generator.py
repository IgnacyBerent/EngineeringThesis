import csv
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import cast

import numpy as np

from src.common.constants import CONDITION_FIELD, ID_FIELD
from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, FloatArray, SubjectData


class ResultsGenerator:
    def __init__(self, processed_data: list[SubjectData]) -> None:
        self.processed_data = processed_data
        self._results: dict[str, dict[int, dict[str, float | None]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        self._fieldnames: list[str] = [ID_FIELD, CONDITION_FIELD]

    def generate_results_csv(self, file_path: str) -> None:
        if not file_path.endswith('.csv'):
            file_path += '.csv'
        if Path(file_path).exists():
            logger.warning(f'Overwritting file: {file_path}!')
            # file_path = file_path.split('.')[0] + '(1).csv'

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

                for cb_type, pid_dict in self._results.items():
                    for pid, metrics in pid_dict.items():
                        row = {ID_FIELD: pid, CONDITION_FIELD: cb_type}
                        row.update(metrics)
                        writer.writerow(row)
        except Exception as e:
            logger.error('Unexpected Exception while saving to csv')
            raise e
        else:
            logger.info(f'Sucesfully saved results to {file_path}')

    def add_means(self, patinets_data: list[SubjectData]) -> None:
        for subject_id, cb_data_type, cb_data in self.iterate_cb_data(patinets_data):
            for field_name, field_value in cast(ArrayDataDict, cb_data).items():
                self._add_result(
                    condition=cb_data_type,
                    subject_id=subject_id,
                    field_name=f'{field_name}_mean',
                    value=float(np.mean(field_value)),
                )

    def iterate_cb_data(
        self, processed_data: list[SubjectData] | None = None
    ) -> Generator[tuple[int, str, ArrayDataDict]]:
        if processed_data is None:
            processed_data = self.processed_data
        for subject_data in processed_data:
            subject_id = self._get_subject_id(subject_data)
            if subject_id is None:
                continue
            for cb_data_type, cb_data in subject_data.items():
                if cb_data_type != 'id':
                    yield subject_id, cb_data_type, cast(ArrayDataDict, cb_data)

    @staticmethod
    def _get_subject_id(subject_data: SubjectData) -> int | None:
        subject_id = subject_data.get('id')
        if type(subject_id) is not int:
            logger.warning('Missing ID')
            return None
        return subject_id

    def _add_result(self, condition: str, subject_id: int, field_name: str, value: float | None) -> None:
        self._results[condition][subject_id][field_name] = value
        if field_name not in self._fieldnames:
            self._fieldnames.append(field_name)

    def _get_signal(self, cb_data: ArrayDataDict, name: str, cb_data_type: str, pid: int) -> FloatArray | None:
        signal = cb_data.get(name)
        if signal is None:
            logger.error(f'Field {name} does not exist in {cb_data_type} for subject {pid}!')
        return signal
