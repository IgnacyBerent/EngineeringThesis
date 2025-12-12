import matplotlib
from src.data_process.loaders import BaroreflexDataLoader
from src.data_process.processors import BaroreflexDataProcessor
from src.data_process.results_generators import BaroreflexResultsGenerator
from src.statistics import StatisticsAnalyzer

matplotlib.use('Agg')

# %%
RESULTS_CSV_FILE_NAME = 'results2.csv'
data_loader = BaroreflexDataLoader()
data_processor = BaroreflexDataProcessor()

# %%
raw_data = data_loader.load_all_patient_raw_data()
# %%
processed_data = data_processor.process_all(raw_data)
# %%
results_generator = BaroreflexResultsGenerator(processed_data)

# %%
sap_hp = results_generator.add_te(y_name='sap', x_name='hp')
cjte_sap_hp = results_generator.add_cjte('sap', 'etco2', 'hp', 'etco2')

# %%
results_generator.generate_results_csv(RESULTS_CSV_FILE_NAME)

# %%
analyzer = StatisticsAnalyzer(RESULTS_CSV_FILE_NAME)
[analyzer.do_rm_anova_test(field) for field in [sap_hp, cjte_sap_hp]]
