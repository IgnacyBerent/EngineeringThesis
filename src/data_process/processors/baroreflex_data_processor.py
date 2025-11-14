from typing import override

from src.common.mytypes import ArrayDataDict
from src.data_process.processors.data_processor import DataProcessor
from src.data_process.utils import get_hp_from_abp, get_sap


class BaroreflexDataProcessor(DataProcessor):
    @override
    def _process_single_cb(self, raw_data: ArrayDataDict) -> ArrayDataDict:
        abp = raw_data.get('abp')
        etco2 = raw_data.get('etco2')
        if abp is None or etco2 is None:
            raise ValueError

        sap = get_sap(abp)
        hp = get_hp_from_abp(abp)
        etco2_shortened = self._average_to_length(etco2, len(sap))
        return {'sap': sap, 'hp': hp, 'etco2': etco2_shortened}
