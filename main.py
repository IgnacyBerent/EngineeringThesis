import matplotlib
from src.data_process.loaders import BaroreflexDataLoader
from src.data_process.processors import BaroreflexDataProcessor
from src.data_process.results_generators import BaroreflexResultsGenerator
from src.statistics import StatisticsAnalyzer

matplotlib.use('Agg')

# %%
RESULTS_CSV_FILE_NAME = 'results.csv'
data_loader = BaroreflexDataLoader()
data_processor = BaroreflexDataProcessor()

# %%
raw_data = data_loader.load_all_patient_raw_data()
# %%
processed_data = data_processor.process_all(raw_data)
results_generator = BaroreflexResultsGenerator(processed_data)

# %%

# %%
sap_hp = results_generator.add_te('hp', 'sap')
hp_sap = results_generator.add_te('sap', 'hp')
cte_sap_hp = results_generator.add_cte('hp', 'sap', 'etco2')
cte_hp_sap = results_generator.add_cte('sap', 'hp', 'etco2')
jte_sap_etco2 = results_generator.add_jte('sap', 'etco2', 'hp')
jte_hp_etco2 = results_generator.add_jte('hp', 'etco2', 'hp')
cjte_sap = results_generator.add_cjte('sap', 'etco2', 'hp', 'etco2')
cjte_hp = results_generator.add_cjte('hp', 'etco2', 'sap', 'etco2')

# %%
results_generator.generate_results_csv(RESULTS_CSV_FILE_NAME)

# %%
analyzer = StatisticsAnalyzer(RESULTS_CSV_FILE_NAME)
[
    analyzer.do_rm_anova_test(field)
    for field in [sap_hp, hp_sap, cte_sap_hp, cte_hp_sap, jte_sap_etco2, jte_hp_etco2, cjte_sap, cjte_hp]
]
