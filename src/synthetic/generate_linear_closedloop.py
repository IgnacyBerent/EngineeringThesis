from src.common.mytypes import PatientData
from src.synthetic.functions.ar import generate_bivariate_ar_closed_loop

_REPETITIONS = 50

_DEFAULT_SIGNAL_LENGTH = 200
_DEFAULT_EPSILON = 0.3
_DEFAULT_A11 = 0.6
_DEFAULT_A22 = 0.6

couplings = [0, 0.05, 0.1, 0.15]
casualities_ytox = [(_DEFAULT_A11, _DEFAULT_A22, coupling, coupling) for coupling in couplings]
LINEAR_CLOSEDLOOP_DATA_A12: list[PatientData] = []
for i in range(_REPETITIONS):
    LINEAR_CLOSEDLOOP_DATA_A12.append(
        {
            'id': i,
            **{
                f'coupling={a_tuple[2]}': generate_bivariate_ar_closed_loop(
                    _DEFAULT_SIGNAL_LENGTH, a_tuple, _DEFAULT_EPSILON
                )
                for a_tuple in casualities_ytox
            },
        }
    )

# casualities_xtoy = [
#     (_DEFAULT_A11, _DEFAULT_A22, _DEFAULT_A12, 0.05),
#     (_DEFAULT_A11, _DEFAULT_A22, _DEFAULT_A12, 0.25),
#     (_DEFAULT_A11, _DEFAULT_A22, _DEFAULT_A12, 0.5),
#     (_DEFAULT_A11, _DEFAULT_A22, _DEFAULT_A12, 1),
# ]
# AR_CLOSED_LOOP_DATA_A21: list[PatientData] = []
# for i in range(_REPETITIONS):
#     AR_CLOSED_LOOP_DATA_A21.append(
#         {
#             'id': i,
#             **{
#                 f'a21={a_tuple[2]}': generate_bivariate_ar_closed_loop(
#                     _DEFAULT_SIGNAL_LENGTH, a_tuple, _DEFAULT_EPSILON
#                 )
#                 for a_tuple in casualities_xtoy
#             },
#         }
#     )
