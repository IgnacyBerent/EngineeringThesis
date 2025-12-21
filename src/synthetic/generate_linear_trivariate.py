from src.common.mytypes import SubjectData
from src.synthetic.common import DEFAULT_SIGNAL_LENGTH, REPETITIONS
from src.synthetic.functions.linear import generate_trivariate_ar

_DEFAULT_AX = 0.3
_DEFAULT_AZ = 0.4

azs = [0, 0.1, 0.25, 0.5]
LINEAR_TRIVARIATE_DATA_Z: list[SubjectData] = []
for i in range(REPETITIONS):
    LINEAR_TRIVARIATE_DATA_Z.append(
        {
            'id': i,
            **{f'az={az}': generate_trivariate_ar(DEFAULT_SIGNAL_LENGTH, az, _DEFAULT_AX, seed=i) for az in azs},
        }
    )

axs = [0, 0.1, 0.25, 0.5]
LINEAR_TRIVARIATE_DATA_YX: list[SubjectData] = []
for i in range(REPETITIONS):
    LINEAR_TRIVARIATE_DATA_YX.append(
        {
            'id': i,
            **{f'ax={ax}': generate_trivariate_ar(DEFAULT_SIGNAL_LENGTH, _DEFAULT_AZ, ax, seed=i) for ax in axs},
        }
    )
