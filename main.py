import matplotlib
from src.common.constants import CB_FILE_TYPE
from src.common.logger import logger
from src.data_process.loaders import BaroreflexDataLoader
from src.data_process.processors import BaroreflexDataProcessor
from src.data_process.results_generators import BaroreflexResultsGenerator
from src.statistics import StatisticsAnalyzer
from src.synthetic import (
    BIVARIATE_SYNTHETIC_SIGNALS_DATA,
    TRIVARIATE_SYNTHETIC_SIGNALS_DATA,
)

matplotlib.use('Agg')


def analyse_physiological_data() -> None:
    logger.info('Running physiological data analysis')
    PHYSIOLOGICAL_RESULTS_CSV_FILE_NAME = 'results_physiological.csv'
    data_loader = BaroreflexDataLoader()
    data_processor = BaroreflexDataProcessor()

    raw_data = data_loader.load_all_raw_data()
    processed_data = data_processor.process_all(raw_data)

    rg = BaroreflexResultsGenerator(processed_data)
    te_sap_hp = rg.add_te(y_name='sap', x_name='hp')
    te_etco_hp = rg.add_te(y_name='etco2', x_name='hp')
    te_etco_sap = rg.add_te(y_name='etco2', x_name='sap')
    cjte_sap_hp = rg.add_cjte('sap', 'etco2', 'hp', 'etco2')

    rg.generate_results_csv(PHYSIOLOGICAL_RESULTS_CSV_FILE_NAME)
    analyzer = StatisticsAnalyzer(PHYSIOLOGICAL_RESULTS_CSV_FILE_NAME, order=CB_FILE_TYPE.order())
    [analyzer.do_rm_anova_test(field, title=f'{field}') for field in [te_sap_hp, cjte_sap_hp, te_etco_sap, te_etco_hp]]
    analyzer.compare(te_sap_hp, cjte_sap_hp)


def analyse_synthetic_bivaraite() -> None:
    logger.info('Running synthetic bivariate data analysis')
    for title, data in BIVARIATE_SYNTHETIC_SIGNALS_DATA.items():
        logger.info(f'Analysing {title}')
        rg = BaroreflexResultsGenerator(data)
        order = None
        if 'Length' in title:
            order = ['Length=100', 'Length=200', 'Length=500', 'Length=1000']
        yx = rg.add_te('x', 'y')
        xy = rg.add_te('y', 'x')
        rg.generate_results_csv(f'{title}.csv')
        analyzer = StatisticsAnalyzer(f'{title}.csv', order=order)
        [analyzer.do_rm_anova_test(field, title=f'{title} {field}') for field in [yx, xy]]


def analyse_synthetic_trivaraite() -> None:
    logger.info('Running synthetic trivariate data analysis')
    for title, data in TRIVARIATE_SYNTHETIC_SIGNALS_DATA.items():
        logger.info(f'Analysing {title}')
        rg = BaroreflexResultsGenerator(data)
        te_yx = rg.add_te('x', 'y')
        te_xy = rg.add_te('y', 'x')
        te_zx = rg.add_te('x', 'z')
        te_zy = rg.add_te('y', 'z')
        cjte_xyz = rg.add_cjte('x', 'z', 'y', 'z')
        cjte_yxz = rg.add_cjte('y', 'z', 'x', 'z')
        rg.generate_results_csv(f'{title}.csv')
        analyzer = StatisticsAnalyzer(f'{title}.csv')
        [
            analyzer.do_rm_anova_test(field, title=f'{title} {field}')
            for field in [te_yx, te_xy, cjte_xyz, cjte_yxz, te_zx, te_zy]
        ]
        analyzer.compare(te_xy, cjte_xyz)


if __name__ == '__main__':
    analyse_physiological_data()
    analyse_synthetic_bivaraite()
    analyse_synthetic_trivaraite()
    logger.info('Finished')
