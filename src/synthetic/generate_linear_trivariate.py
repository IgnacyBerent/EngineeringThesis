from src.common.mytypes import PatientData
from src.synthetic.functions.ar import generate_trivariate_ar

_REPETITIONS = 50

_DEFAULT_SIGNAL_LENGTH = 200
_DEFAULT_EPSILON = 0.3
_DEFAULT_AX = 0.6
_DEFAULT_AY = 0.6
_DEFAULT_AZ = 0.6
_DEFAULT_AYX = 0.1


z_coupling_strengths = [0.00, 0.05, 0.1, 0.2]
a_tuples = [
    (_DEFAULT_AX, _DEFAULT_AY, _DEFAULT_AZ, z_coupling, z_coupling, _DEFAULT_AYX) for z_coupling in z_coupling_strengths
]
LINEAR_TRIVARIATE_DATA: list[PatientData] = []
for i in range(_REPETITIONS):
    LINEAR_TRIVARIATE_DATA.append(
        {
            'id': i,
            **{
                f'z_coupling={a_tuple[-2]}': generate_trivariate_ar(_DEFAULT_SIGNAL_LENGTH, a_tuple, _DEFAULT_EPSILON)
                for a_tuple in a_tuples
            },
        }
    )
