import tempfile

import matplotlib
from src.common.constants import CB_FILE_TYPE
from src.common.logger import logger
from src.data_process.loaders import BaroreflexDataLoader
from src.data_process.processors import BaroreflexDataProcessor
from src.data_process.results_generators import BaroreflexResultsGenerator
from src.statistics import StatisticsAnalyzer
from src.synthetic import (
    LINEAR_CLOSEDLOOP_DATA_A12,
    LINEAR_OPENLOOP_DATA_A,
    LINEAR_OPENLOOP_DATA_E,
    LINEAR_OPENLOOP_DATA_L,
    LINEAR_TRIVARIATE_DATA,
    NONLINEAR_CLOSEDLOOP_DATA,
)

matplotlib.use('Agg')


def analyse_physiological_data() -> None:
    logger.info('Running physiological data analysis')
    PHYSIOLOGICAL_RESULTS_CSV_FILE_NAME = 'results_physiological.csv'
    try:
        data_loader = BaroreflexDataLoader()
        data_processor = BaroreflexDataProcessor()

        raw_data = data_loader.load_all_patient_raw_data()
        processed_data = data_processor.process_all(raw_data)

        results_generator_physiological = BaroreflexResultsGenerator(processed_data)
        sap_hp = results_generator_physiological.add_te(y_name='sap', x_name='hp')
        cjte_sap_hp = results_generator_physiological.add_cjte('sap', 'etco2', 'hp', 'etco2')

        results_generator_physiological.generate_results_csv(PHYSIOLOGICAL_RESULTS_CSV_FILE_NAME)

        analyzer = StatisticsAnalyzer(PHYSIOLOGICAL_RESULTS_CSV_FILE_NAME, order=CB_FILE_TYPE.order())
        [analyzer.do_rm_anova_test(field, title=f'physiological {field}') for field in [sap_hp, cjte_sap_hp]]
    except Exception as e:
        logger.warning(f'Error occured in physiological data: {e}')


def analyse_synthetic_bivaraite() -> None:
    logger.info('Running synthetic bivariate data analysis')
    rg_linear_openloop_l = BaroreflexResultsGenerator(LINEAR_OPENLOOP_DATA_L)
    rg_linear_openloop_e = BaroreflexResultsGenerator(LINEAR_OPENLOOP_DATA_E)
    rg_linear_openloop_a = BaroreflexResultsGenerator(LINEAR_OPENLOOP_DATA_A)
    rg_closed_loop = BaroreflexResultsGenerator(LINEAR_CLOSEDLOOP_DATA_A12)
    rg_nonlinear = BaroreflexResultsGenerator(NONLINEAR_CLOSEDLOOP_DATA)
    titles = [
        'linear_openloop_l',
        'rg_linear_openloop_e',
        'rg_linear_openloop_a',
        'rg_closed_loop',
        'rg_nonlinear',
    ]

    for rg, title in zip(
        [rg_linear_openloop_l, rg_linear_openloop_e, rg_linear_openloop_a, rg_closed_loop, rg_nonlinear],
        titles,
        strict=True,
    ):
        yx = rg.add_te('x', 'y')
        xy = rg.add_te('y', 'x')
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp:
            temp_file_path = tmp.name
            rg.generate_results_csv(temp_file_path)
            analyzer = StatisticsAnalyzer(temp_file_path)
            [analyzer.do_rm_anova_test(field, title=f'{title} {field}') for field in [yx, xy]]


def analyse_synthetic_trivaraite() -> None:
    logger.info('Running synthetic trivariate data analysis')
    rg = BaroreflexResultsGenerator(LINEAR_TRIVARIATE_DATA)
    yx = rg.add_te('x', 'y')
    xy = rg.add_te('y', 'x')
    xyz = rg.add_cjte('x', 'z', 'y', 'z')
    yxz = rg.add_cjte('y', 'z', 'x', 'z')
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp:
        temp_file_path = tmp.name
        rg.generate_results_csv(temp_file_path)
        analyzer = StatisticsAnalyzer(temp_file_path)
        [analyzer.do_rm_anova_test(field, title=f'trivariate {field}') for field in [yx, xy, xyz, yxz]]


if __name__ == '__main__':
    # analyse_physiological_data()
    # analyse_synthetic_bivaraite()
    analyse_synthetic_trivaraite()
