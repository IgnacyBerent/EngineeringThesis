from src.common.mytypes import PatientData
from src.data_process.entropy.conditional_transfer_entropy import cte_dv
from src.data_process.entropy.transfer_entropy_dv import te_dv
from src.data_process.entropy.transfer_entropy_hist import te_hist
from src.data_process.results_generators.result_generator import ResultsGenerator


class BaroreflexResultsGenerator(ResultsGenerator):
    def __init__(self, patients_processed_data: list[PatientData]) -> None:
        super().__init__(patients_processed_data)

    def add_te_dv(self, x_name: str, y_name: str) -> str:
        field_name = f'te_dv_{y_name}->{x_name}'
        for patient_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y = (self._get_signal(cb_data, sig_name, cb_data_type, patient_id) for sig_name in [x_name, y_name])
            if x is not None and y is not None:
                self._add_result(
                    cb_data_type=cb_data_type,
                    patient_id=patient_id,
                    field_name=field_name,
                    value=te_dv(x, y),
                )
        return field_name

    def add_te_hist(self, x_name: str, y_name: str) -> str:
        field_name = f'te_hist_{y_name}->{x_name}'
        for patient_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y = (self._get_signal(cb_data, sig_name, cb_data_type, patient_id) for sig_name in [x_name, y_name])
            if x is not None and y is not None:
                self._add_result(
                    cb_data_type=cb_data_type,
                    patient_id=patient_id,
                    field_name=field_name,
                    value=te_hist(x, y),
                )
        return field_name

    def add_cte(self, x_name: str, y_name: str, z_name: str) -> str:
        field_name = f'cte_{y_name}->{x_name}|{z_name}'
        for patient_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y, z = (
                self._get_signal(cb_data, sig_name, cb_data_type, patient_id) for sig_name in [x_name, y_name, z_name]
            )
            if x is not None and y is not None and z is not None:
                self._add_result(
                    cb_data_type=cb_data_type,
                    patient_id=patient_id,
                    field_name=field_name,
                    value=cte_dv(x, y, z),
                )
        return field_name
