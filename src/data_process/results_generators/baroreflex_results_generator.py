from src.common.logger import logger
from src.common.mytypes import SubjectData
from src.data_process.entropy import cjte_dv, cte_dv, jte_dv, te_dv
from src.data_process.results_generators.result_generator import ResultsGenerator


class BaroreflexResultsGenerator(ResultsGenerator):
    def __init__(self, processed_data: list[SubjectData]) -> None:
        super().__init__(processed_data)

    def add_te(
        self,
        x_name: str,
        y_name: str,
    ) -> str:
        field_name = f'te_{y_name}->{x_name}'
        for subject_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y = (self._get_signal(cb_data, sig_name, cb_data_type, subject_id) for sig_name in [x_name, y_name])
            try:
                if x is not None and y is not None:
                    self._add_result(
                        condition=cb_data_type,
                        subject_id=subject_id,
                        field_name=field_name,
                        value=te_dv(x, y),
                    )
            except ValueError as e:
                logger.error(f'TE calculation error for P{subject_id} {cb_data_type} {e}')
                self._add_result(
                    condition=cb_data_type,
                    subject_id=subject_id,
                    field_name=field_name,
                    value=None,
                )
        return field_name

    def add_cte(self, x_name: str, y_name: str, z_name: str) -> str:
        field_name = f'cte_{y_name}->{x_name}|{z_name}'
        for subject_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y, z = (
                self._get_signal(cb_data, sig_name, cb_data_type, subject_id) for sig_name in [x_name, y_name, z_name]
            )
            if x is not None and y is not None and z is not None:
                try:
                    self._add_result(
                        condition=cb_data_type,
                        subject_id=subject_id,
                        field_name=field_name,
                        value=cte_dv(x, y, z),
                    )
                except ValueError as e:
                    logger.error(f'CTE calculation error for P{subject_id} {cb_data_type} {e}')
                    self._add_result(
                        condition=cb_data_type,
                        subject_id=subject_id,
                        field_name=field_name,
                        value=None,
                    )
        return field_name

    def add_jte(self, x_name: str, y_name: str, z_name: str) -> str:
        field_name = f'jte_({x_name},{y_name})->{z_name}'
        for subject_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y, z = (
                self._get_signal(cb_data, sig_name, cb_data_type, subject_id) for sig_name in [x_name, y_name, z_name]
            )
            if x is not None and y is not None and z is not None:
                try:
                    self._add_result(
                        condition=cb_data_type,
                        subject_id=subject_id,
                        field_name=field_name,
                        value=jte_dv(x, y, z),
                    )
                except ValueError as e:
                    logger.error(f'JTE calculation error for P{subject_id} {cb_data_type} {e}')
                    self._add_result(
                        condition=cb_data_type,
                        subject_id=subject_id,
                        field_name=field_name,
                        value=None,
                    )
        return field_name

    def add_cjte(self, x_name: str, y_name: str, z_name: str, w_name: str) -> str:
        field_name = f'cjte_({x_name},{y_name})->{z_name}|{w_name}'
        for subject_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y, z, w = (
                self._get_signal(cb_data, sig_name, cb_data_type, subject_id)
                for sig_name in [x_name, y_name, z_name, w_name]
            )
            if x is not None and y is not None and z is not None and w is not None:
                w = w if w_name != y_name else None
                try:
                    self._add_result(
                        condition=cb_data_type,
                        subject_id=subject_id,
                        field_name=field_name,
                        value=cjte_dv(x, y, z, w),
                    )
                except ValueError as e:
                    logger.error(f'CJTE calculation error for P{subject_id} {cb_data_type} {e}')
                    self._add_result(
                        condition=cb_data_type,
                        subject_id=subject_id,
                        field_name=field_name,
                        value=None,
                    )
        return field_name
