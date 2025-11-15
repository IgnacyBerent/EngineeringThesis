from src.data_process.loaders.baroreflex_data_loader import BaroreflexDataLoader
from src.data_process.processors.baroreflex_data_processor import BaroreflexDataProcessor
from src.data_process.results_generators.baroreflex_results_generator import BaroreflexResultsGenerator
from src.statistics.anova import StatisticsAnalyzer

# %%
data_loader = BaroreflexDataLoader()
data_processor = BaroreflexDataProcessor()

# %%
raw_data = data_loader.load_all_patient_raw_data()
# %%
processed_data = data_processor.process_all(raw_data)
results_generator = BaroreflexResultsGenerator(processed_data)

# %%
sap_hp = results_generator.add_te_dv('hp', 'sap')
hp_sap = results_generator.add_te_dv('sap', 'hp')
cte = results_generator.add_cte('hp', 'sap', 'etco2')

# %%
results_generator.generate_results_csv('baroreflex.csv')

# %%
analyzer = StatisticsAnalyzer('baroreflex.csv')
[analyzer.do_rm_anova_test(field) for field in [hp_sap, sap_hp, cte]]
