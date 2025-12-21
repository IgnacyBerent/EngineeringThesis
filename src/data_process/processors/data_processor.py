from abc import ABC, abstractmethod
from typing import cast

from src.common.logger import logger
from src.common.mytypes import ArrayDataDict, SubjectData


class DataProcessor(ABC):
    @abstractmethod
    def _process_single_cb(self, raw_data: ArrayDataDict) -> ArrayDataDict:
        pass

    def process_all(self, raw_data: list[SubjectData]) -> list[SubjectData]:
        processed_data = []
        for subject_raw_data in raw_data:
            if processed_subject_data := self.process(subject_raw_data):
                processed_data.append(processed_subject_data)
        return processed_data

    def process(self, subject_raw_data: SubjectData) -> SubjectData | None:
        try:
            return {
                'id': (subject_raw_data.get('id', 404)),  # type: ignore
                **{
                    cb_field_name: self._process_single_cb(cast(ArrayDataDict, cb_raw_data))
                    for cb_field_name, cb_raw_data in subject_raw_data.items()
                    if cb_field_name != 'id'
                },
            }
        except ValueError:
            logger.error(f'Missmatching field names for subject: {subject_raw_data["id"]}')
