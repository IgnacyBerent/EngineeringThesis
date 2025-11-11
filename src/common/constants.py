from enum import Enum
from pathlib import Path

METADATA_PATH = Path('data/metadata.xlsx')
BREATHING_DATA_DIRECTORY_PATH = Path('data/CONTROL_BREATHING_RECORDINGS')

ID_FIELD = 'pid'
CONDITION_FIELD = 'cb_type'


class CB_FILE_TYPE(str, Enum):
    B6 = 'CB_6'
    B10 = 'CB_10'
    B15 = 'CB_15'
    BASELINE = 'CB_BASELINE'

    @property
    def csv(self) -> str:
        return f'{self.value}.csv'

    @classmethod
    def order(cls) -> list[str]:
        return [cls.BASELINE.value, cls.B6.value, cls.B10.value, cls.B15.value]


class SignalColumns(str, Enum):
    ABP = 'abp_cnap[mmHg]'
    ETCO2 = 'etco2[mmHg]'


SAMPLING_FREQUENCY = 200  # Hz

DEFAULT_TIME_DELAY = 1
DEFAULT_EMBEDDING_DIMENSION = 1
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
