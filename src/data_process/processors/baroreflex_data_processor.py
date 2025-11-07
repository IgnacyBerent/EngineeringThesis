from typing import override

from src.common.mytypes import BaroreflexProcessedData, BaroreflexRawData
from src.data_process.processors.data_processor import DataProcessor
from src.data_process.utils import PeaksMode, get_hp_from_abp, get_peaks, get_sap


class BaroreflexDataProcessor(DataProcessor[BaroreflexRawData, BaroreflexProcessedData]):
    @override
    def _process_single_cb(self, raw_data: BaroreflexRawData) -> BaroreflexProcessedData:
        abp = raw_data.abp
        abp_peaks = get_peaks(abp, PeaksMode.UP)

        sap = get_sap(abp, abp_peaks)
        hp = get_hp_from_abp(abp)
        etco2_shortened = self._average_to_length(raw_data.etco2, len(sap))
        return BaroreflexProcessedData(sap=sap, hp=hp, etco2=etco2_shortened)
