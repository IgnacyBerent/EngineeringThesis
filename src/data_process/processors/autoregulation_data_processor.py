from typing import override

import numpy as np

from src.common.mytypes import ArrayDataDict, FloatArray
from src.data_process.processors.data_processor import DataProcessor
from src.data_process.utils import get_hp_from_abp, get_map, get_mfv, get_sap

_MINIMAL_VALID_MEAN_FV = 30


class AutoregulationDataProcessor(DataProcessor):
    @override
    def _process_single_cb(self, raw_data: ArrayDataDict) -> ArrayDataDict:
        abp = raw_data.get('abp')
        etco2 = raw_data.get('etco2')
        fv = self._get_fv(raw_data)
        if abp is None or etco2 is None or fv is None:
            raise ValueError

        sap = get_sap(abp)
        _map = get_map(abp)
        mfv = get_mfv(fv)
        hp = get_hp_from_abp(abp)
        if len(hp) < len(_map):
            _map = _map[len(_map) - len(hp) :]
        if len(hp) > len(_map):
            sap = sap[: len(_map) - len(hp)]
            hp = hp[: len(_map) - len(hp)]
            mfv = mfv[: len(_map) - len(hp)]
        if len(mfv) - len(hp) < 0:
            sap = sap[: len(mfv) - len(hp)]
            _map = _map[: len(mfv) - len(hp)]
            hp = hp[: len(mfv) - len(hp)]
        if len(mfv) - len(hp) > 0:
            mfv = mfv[len(mfv) - len(hp) :]

        etco2_shortened = self._average_to_length(etco2, len(hp))
        return {'sap': sap, 'hp': hp, 'etco2': etco2_shortened, 'map': _map, 'mfv': mfv}

    @staticmethod
    def _get_fv(raw_data: ArrayDataDict) -> FloatArray | None:
        fvl = raw_data.get('fvl')
        fvr = raw_data.get('fvr')

        if fvl is not None and np.mean(fvl) > _MINIMAL_VALID_MEAN_FV:
            return fvl
        if fvr is not None and np.mean(fvr) > _MINIMAL_VALID_MEAN_FV:
            return fvr
        return None
