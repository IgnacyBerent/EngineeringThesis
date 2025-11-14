from src.common.mytypes import PatientData
from src.data_process.entropy.conditional_joint_transfer_entropy import cjte_dv
from src.data_process.results_generators.result_generator import ResultsGenerator


class AutonomicResultsGenerator(ResultsGenerator):
    def __init__(self, patients_processed_data: list[PatientData]) -> None:
        super().__init__(patients_processed_data)

    def add_cjte(self, x_name: str, y_name: str, z_name: str, w_name: str) -> str:
        field_name = f'cjte_({x_name},{y_name})->{z_name}|{w_name}'
        for patient_id, cb_data_type, cb_data in self.iterate_cb_data():
            x, y, z, w = (
                self._get_signal(cb_data, sig_name, cb_data_type, patient_id)
                for sig_name in [x_name, y_name, z_name, w_name]
            )
            if x is not None and y is not None and z is not None and w is not None:
                self._add_result(
                    cb_data_type=cb_data_type,
                    patient_id=patient_id,
                    field_name=field_name,
                    value=cjte_dv(x, y, z, w),
                )
        return field_name
