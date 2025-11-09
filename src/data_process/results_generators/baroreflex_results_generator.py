from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, PatientData
from src.data_process.entropy.transfer_entropy_dv import transfer_entropy_dv
from src.data_process.results_generators.result_generator import ResultsGenerator


class BaroreflexResultsGenerator(ResultsGenerator):
    def __init__(self, patients_processed_data: list[PatientData]) -> None:
        super().__init__(patients_processed_data)

    def add_te_dv(self, fields: tuple[str, str], one_direction: bool = False) -> None:
        for patient_id, cb_data_type, cb_data in self.iterate_cb_data():
            field1 = cb_data.get(fields[0])
            field2 = cb_data.get(fields[1])
            if field1 is None:
                logger.error(f'Field {field1} does not exist in {cb_data_type} for patient {patient_id}!')
                return
            if field2 is None:
                logger.error(f'Field {field1} does not exist in {cb_data_type} for patient {patient_id}!')
                return
            self._add_result(
                cb_data_type=cb_data_type,
                patient_id=patient_id,
                field_name=f'te_{field2}->{field1}',
                value=transfer_entropy_dv(field1, field2),
            )
            if not one_direction:
                self._add_result(
                    cb_data_type=cb_data_type,
                    patient_id=patient_id,
                    field_name=f'te_{field1}->{field2}',
                    value=transfer_entropy_dv(field2, field1),
                )
