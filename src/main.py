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
hp_sap_dv = results_generator.add_te_dv('sap', 'hp')
sap_hp_dv = results_generator.add_te_dv('hp', 'sap')
hp_sap_hist = results_generator.add_te_hist('sap', 'hp')
sap_hp_hist = results_generator.add_te_hist('hp', 'sap')
sap_hp_etco2 = results_generator.add_cte('hp', 'sap', 'etco2')

# %%
results_generator.generate_results_csv('baroreflex_test2.csv')

# %%
analyzer = StatisticsAnalyzer('baroreflex_test2.csv')
[analyzer.do_rm_anova_test(field) for field in [hp_sap_dv, sap_hp_dv, hp_sap_hist, sap_hp_hist, sap_hp_etco2]]
