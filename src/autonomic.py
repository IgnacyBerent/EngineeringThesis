# %%
from src.data_process.loaders.autonomic_data_loader import AutonomicDataLoader
from src.data_process.processors.autonomic_data_processor import AutonomicDataProcessor
from src.data_process.results_generators.autonomic_results_generator import AutonomicResultsGenerator
from src.statistics.anova import StatisticsAnalyzer

data_loader = AutonomicDataLoader()
data_processor = AutonomicDataProcessor()

# %%
raw_data = data_loader.load_all_patient_raw_data()

# %%
processed_data = data_processor.process_all(raw_data)
results_generator = AutonomicResultsGenerator(processed_data)

# %%
cjte = results_generator.add_cjte('hp', 'sap', 'mfv', 'etco2')

# %%
results_generator.generate_results_csv('autonomic_test.csv')

# %%
analyzer = StatisticsAnalyzer('autonomic_test.csv')
[analyzer.do_rm_anova_test(field) for field in [cjte]]
