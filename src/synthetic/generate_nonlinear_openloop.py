from src.common.mytypes import SubjectData
from src.synthetic.common import DEFAULT_SIGNAL_LENGTH, REPETITIONS
from src.synthetic.functions.nonlinear import generate_nonlinear_bivariate_process

bs = [0, 0.1, 0.25, 0.5]

NONLINEAR_CLOSEDLOOP_DATA: list[SubjectData] = []
for i in range(REPETITIONS):
    NONLINEAR_CLOSEDLOOP_DATA.append(
        {
            'id': i,
            **{f'b={b}': generate_nonlinear_bivariate_process(DEFAULT_SIGNAL_LENGTH, b=b, seed=i) for b in bs},
        }
    )
