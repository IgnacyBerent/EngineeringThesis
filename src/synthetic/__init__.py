from src.synthetic.generate_linear_bivaraite import (
    LINEAR_BIVARIATE_DATA_A,
    LINEAR_BIVARIATE_DATA_E,
    LINEAR_BIVARIATE_DATA_L,
)
from src.synthetic.generate_linear_trivariate import (
    LINEAR_TRIVARIATE_DATA_YX,
    LINEAR_TRIVARIATE_DATA_Z,
)
from src.synthetic.generate_nonlinear_openloop import NONLINEAR_CLOSEDLOOP_DATA

BIVARIATE_SYNTHETIC_SIGNALS_DATA = {
    'Varying Length Linear Bivariate': LINEAR_BIVARIATE_DATA_L,
    'Varying SNR Linear Bivariate': LINEAR_BIVARIATE_DATA_E,
    'Varying a Linear Bivariate': LINEAR_BIVARIATE_DATA_A,
    'Varying b Nonlinear Bivariate': NONLINEAR_CLOSEDLOOP_DATA,
}

TRIVARIATE_SYNTHETIC_SIGNALS_DATA = {
    'Varying az Linear Trivariate': LINEAR_TRIVARIATE_DATA_Z,
    'Varying ax Linear Trivariate': LINEAR_TRIVARIATE_DATA_YX,
}
