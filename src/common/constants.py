from enum import Enum
from pathlib import Path

METADATA_PATH = Path('data/metadata.xlsx')
BREATHING_DATA_DIRECTORY_PATH = Path('data/CONTROL_BREATHING_RECORDINGS')


CB_6 = 'CB_6.csv'
CB_BASELINE = 'CB_BASELINE.csv'
PATIENT_DATA_FILE_TYPES: list[str] = [CB_6, CB_BASELINE]


class SignalColumnNames(str, Enum):
    ABP = 'abp_cnap[mmHg]'
    RR = 'rr[rpm]'
    ETCO2 = 'etco2[mmHg]'


SAMPLING_FREQUENCY = 200

DEFAULT_TIME_DELAY = 1
DEFAULT_EMBEDDING_DIMENSION = 1
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
