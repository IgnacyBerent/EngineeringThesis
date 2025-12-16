from src.common.mytypes import PatientData
from src.synthetic.functions.ar import generate_bivariate_ar_open_loop

_REPETITIONS = 50

_DEFAULT_SIGNAL_LENGTH = 200
_DEFAULT_EPSILON = 0.3
_DEFAULT_A11 = 0.6
_DEFAULT_A22 = 0.6
_DEFAULT_A21 = 0.5
_DEFAULT_A_AR_OPENLOOP = (_DEFAULT_A11, _DEFAULT_A22, _DEFAULT_A21)

lengths = [100, 200, 500, 1000]
LINEAR_OPENLOOP_DATA_L: list[PatientData] = []
for i in range(_REPETITIONS):
    LINEAR_OPENLOOP_DATA_L.append(
        {
            'id': i,
            **{
                f'length={signal_length}': generate_bivariate_ar_open_loop(
                    signal_length, _DEFAULT_A_AR_OPENLOOP, _DEFAULT_EPSILON
                )
                for signal_length in lengths
            },
        }
    )

epsilons = [0.05, 0.25, 0.5, 1]
LINEAR_OPENLOOP_DATA_E: list[PatientData] = []
for i in range(_REPETITIONS):
    LINEAR_OPENLOOP_DATA_E.append(
        {
            'id': i,
            **{
                f'epsilon={epsilon}': generate_bivariate_ar_open_loop(
                    _DEFAULT_SIGNAL_LENGTH, _DEFAULT_A_AR_OPENLOOP, epsilon
                )
                for epsilon in epsilons
            },
        }
    )

casualities = [
    (_DEFAULT_A11, _DEFAULT_A22, 0.05),
    (_DEFAULT_A11, _DEFAULT_A22, 0.25),
    (_DEFAULT_A11, _DEFAULT_A22, 0.5),
    (_DEFAULT_A11, _DEFAULT_A22, 1),
]
LINEAR_OPENLOOP_DATA_A: list[PatientData] = []
for i in range(_REPETITIONS):
    LINEAR_OPENLOOP_DATA_A.append(
        {
            'id': i,
            **{
                f'a21={a_tuple[2]}': generate_bivariate_ar_open_loop(_DEFAULT_SIGNAL_LENGTH, a_tuple, _DEFAULT_EPSILON)
                for a_tuple in casualities
            },
        }
    )
