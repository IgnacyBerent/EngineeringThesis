import csv
from collections import defaultdict
from collections.abc import Generator
from typing import cast

import numpy as np

from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, PatientData


class ResultsGenerator:
    def __init__(self, patients_processed_data: list[PatientData]) -> None:
        self.patients_processed_data = patients_processed_data
        self._results: dict[str, dict[int, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

    def generate_results_csv(self, file_path: str) -> None:
        if not file_path.endswith('.csv'):
            file_path += '.csv'

        field_names = set()
        for _, _, cb_data in self.iterate_cb_data():
            field_names.update(cb_data.keys())

        metric_names = sorted(field_names)

        header = ['pid', 'cb_type'] + metric_names
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for cb_type, pid_dict in self._results.items():
                for pid, metrics in pid_dict.items():
                    row = {'pid': pid, 'cb_type': cb_type}
                    row.update(metrics)
                    writer.writerow(row)

    def add_means(self, patinets_data: list[PatientData]) -> None:
        for patient_id, cb_data_type, cb_data in self.iterate_cb_data(patinets_data):
            for field_name, field_value in cast(ArrayDataDict, cb_data).items():
                self._add_result(
                    cb_data_type=cb_data_type,
                    patient_id=patient_id,
                    field_name=f'{field_name}_mean',
                    value=float(np.mean(field_value)),
                )

    def iterate_cb_data(
        self, patients_data: list[PatientData] | None = None
    ) -> Generator[tuple[int, str, ArrayDataDict]]:
        if patients_data is None:
            patients_data = self.patients_processed_data
        for patient_data in patients_data:
            patient_id = self._get_patient_id(patient_data)
            print(patient_id)
            if patient_id is None:
                continue
            for cb_data_type, cb_data in patient_data.items():
                if cb_data is ArrayDataDict:
                    yield patient_id, cb_data_type, cast(ArrayDataDict, cb_data)

    @staticmethod
    def _get_patient_id(patient_data: PatientData) -> int | None:
        patient_id = patient_data.get('id')
        if patient_id is not int:
            logger.warning('Missing ID')
            return None
        return patient_id

    def _add_result(self, cb_data_type: str, patient_id: int, field_name: str, value: float) -> None:
        self._results[cb_data_type][patient_id][field_name] = value
