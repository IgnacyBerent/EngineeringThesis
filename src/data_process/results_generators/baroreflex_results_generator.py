from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, PatientData
from src.data_process.entropy.transfer_entropy_dv import transfer_entropy_dv
from src.data_process.results_generators.result_generator import ResultsGenerator


class BaroreflexResultsGenerator(ResultsGenerator):
    def __init__(self, patients_processed_data: list[PatientData]) -> None:
        super().__init__(patients_processed_data)

    def add_te_dv(self, x_name: str, y_name: str, both_directions: bool = True) -> None:
        for patient_id, cb_data_type, cb_data in self.iterate_cb_data():
            x = cb_data.get(x_name)
            y = cb_data.get(y_name)
            if x is None:
                logger.error(f'Field {x_name} does not exist in {cb_data_type} for patient {patient_id}!')
                return
            if y is None:
                logger.error(f'Field {y_name} does not exist in {cb_data_type} for patient {patient_id}!')
                return
            self._add_result(
                cb_data_type=cb_data_type,
                patient_id=patient_id,
                field_name=f'te_{y_name}->{x_name}',
                value=transfer_entropy_dv(x, y),
            )
            if both_directions:
                self._add_result(
                    cb_data_type=cb_data_type,
                    patient_id=patient_id,
                    field_name=f'te_{x_name}->{y_name}',
                    value=transfer_entropy_dv(y, x),
                )
