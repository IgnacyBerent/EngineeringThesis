from enum import Enum
from pathlib import Path

METADATA_PATH = Path('data/metadata.xlsx')
BREATHING_DATA_DIRECTORY_PATH = Path('data/CONTROL_BREATHING_RECORDINGS')


class CB_FILE(str, Enum):
    B6 = 'CB_6.csv'
    B10 = 'CB_10.csv'
    B15 = 'CB_15.csv'
    BASELINE = 'CB_BASELINE.csv'


class SignalColumns(str, Enum):
    ABP = 'abp_cnap[mmHg]'
    ETCO2 = 'etco2[mmHg]'


SAMPLING_FREQUENCY = 200  # Hz

DEFAULT_TIME_DELAY = 1
DEFAULT_EMBEDDING_DIMENSION = 1
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
