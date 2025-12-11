from typing import override

from src.common.mytypes import ArrayDataDict
from src.data_process.processors.data_processor import DataProcessor
from src.data_process.processors.utils import adjust_etco2, get_hp, get_peaks, get_sap


class BaroreflexDataProcessor(DataProcessor):
    @override
    def _process_single_cb(self, raw_data: ArrayDataDict) -> ArrayDataDict:
        abp = raw_data.get('abp')
        etco2 = raw_data.get('etco2')
        if abp is None or etco2 is None:
            raise ValueError

        peaks = get_peaks(abp)
        sap = get_sap(abp, peaks)
        hp = get_hp(peaks)
        etco2_adjusted = adjust_etco2(etco2, peaks)
        return {'sap': sap, 'hp': hp, 'etco2': etco2_adjusted}
