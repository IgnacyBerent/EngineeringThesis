# %%
from src.data_process.loaders.autoregulation_data_loader import AutoregulationDataLoader
from src.data_process.processors.autoregulation_data_processor import AutoregulationDataProcessor
from src.data_process.results_generators.autoregulation_results_generator import AutoregulationResultsGenerator
from src.statistics.anova import StatisticsAnalyzer

data_loader = AutoregulationDataLoader()
data_processor = AutoregulationDataProcessor()

# %%
raw_data = data_loader.load_all_patient_raw_data()

# %%
processed_data = data_processor.process_all(raw_data)
results_generator = AutoregulationResultsGenerator(processed_data)

# %%
cjte1 = results_generator.add_cjte('hp', 'sap', 'mfv', 'etco2')
cjte2 = results_generator.add_cjte('hp', 'map', 'mfv', 'etco2')
hp = results_generator.add_cte('mfv', 'hp', 'etco2')
sap = results_generator.add_cte('mfv', 'sap', 'etco2')
sapte = results_generator.add_te('mfv', 'sap')
mapte = results_generator.add_te('mfv', 'map')
_map = results_generator.add_cte('mfv', 'map', 'etco2')

# %%
results_generator.generate_results_csv('autoregulation.csv')

# %%
analyzer = StatisticsAnalyzer('autoregulation.csv')
[analyzer.do_rm_anova_test(field) for field in [cjte1, cjte2, sap, hp, _map, sapte, mapte]]
