from src.common.mytypes import PatientData
from src.synthetic.common import DEFAULT_SIGNAL_LENGTH, REPETITIONS
from src.synthetic.functions.linear import generate_bivariate_ar

_DEFAULT_A = 0.5

lengths = [100, 200, 500, 1000]
LINEAR_BIVARIATE_DATA_L: list[PatientData] = []
for i in range(REPETITIONS):
    LINEAR_BIVARIATE_DATA_L.append(
        {
            'id': i,
            **{
                f'Length={signal_length}': generate_bivariate_ar(signal_length, _DEFAULT_A, seed=i)
                for signal_length in lengths
            },
        }
    )

snrs = [None, 30, 20, 10]
LINEAR_BIVARIATE_DATA_E: list[PatientData] = []
for i in range(REPETITIONS):
    LINEAR_BIVARIATE_DATA_E.append(
        {
            'id': i,
            **{
                f'SNR={snr if snr else "None"}': generate_bivariate_ar(
                    DEFAULT_SIGNAL_LENGTH, _DEFAULT_A, snr=snr, seed=i
                )
                for snr in snrs
            },
        }
    )

a_ranges = [0, 0.1, 0.25, 0.5]
LINEAR_BIVARIATE_DATA_A: list[PatientData] = []
for i in range(REPETITIONS):
    LINEAR_BIVARIATE_DATA_A.append(
        {
            'id': i,
            **{f'a={a}': generate_bivariate_ar(DEFAULT_SIGNAL_LENGTH, a, seed=i) for a in a_ranges},
        }
    )
