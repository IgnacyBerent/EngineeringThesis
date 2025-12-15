from src.common.mytypes import PatientData
from src.synthetic.functions.henon_map import generate_coupled_henon_map

_REPETITIONS = 50
_DEFAULT_SIGNAL_LENGTH = 200

gammas = [0, 0.05, 0.1, 0.2]

NONLINEAR_CLOSEDLOOP_DATA: list[PatientData] = []
for i in range(_REPETITIONS):
    NONLINEAR_CLOSEDLOOP_DATA.append(
        {
            'id': i,
            **{f'gamma={gamma}': generate_coupled_henon_map(_DEFAULT_SIGNAL_LENGTH, gamma=gamma) for gamma in gammas},
        }
    )
