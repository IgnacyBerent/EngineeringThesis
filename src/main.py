from src.data_process.entropy.transfer_entropy_dv import transfer_entropy_dv
from src.data_process.loaders.baroreflex_data_loader import BaroreflexDataLoader
from src.data_process.processors.baroreflex_data_processor import BaroreflexDataProcessor

# %%
patient_data_loader = BaroreflexDataLoader()
baroreflex_data_processor = BaroreflexDataProcessor()

# %%
raw_data = patient_data_loader.load_all_patient_raw_data()
# %%
processed_data = [baroreflex_data_processor.process(data) for data in raw_data]
pcd = processed_data[0]

# %%
pcd_b = pcd.baseline
print(transfer_entropy_dv(pcd_b.sap, pcd_b.hp))
print(transfer_entropy_dv(pcd_b.hp, pcd_b.sap))
