# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: EngineeringThesis
#     language: python
#     name: engineeringthesis
# ---

# %%
from typing import cast

import matplotlib
import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray
from src.common.constants import BREATHING_DATA_DIRECTORY_PATH, SAMPLING_FREQUENCY
from src.data_process.entropy.dvp import dv_partition_nd
from src.data_process.entropy.utils import (
    get_deleyed_vector,
    get_future_vector,
    get_past_vectors,
    get_points_from_range,
    rank_transform,
)
from src.data_process.loaders.baroreflex_data_loader import BaroreflexDataLoader
from src.data_process.processors.baroreflex_data_processor import BaroreflexDataProcessor
from src.data_process.processors.utils import PeaksMode, get_peaks
from src.data_process.results_generators.baroreflex_results_generator import BaroreflexResultsGenerator
from src.plots import TimeUnit, plot_multiple_signals_shared_x, plot_single_signal, plot_single_signal_with_peaks
from src.plots.dv_plots import plot_2d_partitions, plot_3d_partitions
from src.statistics.statistics_analyzer import StatisticsAnalyzer

matplotlib.use('Agg')
# %%

noiseX = np.random.normal(0, 1, 1000) + 1
noiseY = np.random.normal(0, 1, 1000) + 1.1 * noiseX + 2
rankedX = rank_transform(noiseX)
rankedY = rank_transform(noiseY)
partitions = dv_partition_nd(np.column_stack([rankedX, rankedY]))
dvp2d = plot_2d_partitions(partitions, rankedX, rankedY, 'Ranked noise X', 'Ranked noise Y')

# %%
data_loader = BaroreflexDataLoader()
data_processor = BaroreflexDataProcessor()

# %%
subject = 7
cb = 'CB_BASELINE'
subject_data = data_loader.load_single_subject_raw_data(list(BREATHING_DATA_DIRECTORY_PATH.iterdir())[subject])

# %%
# abp = subject_data[cb]['abp'][1000:2000]
# etco2 = subject_data[cb]['etco2'][1000:2000]

# %%

# cleaned_abp = cast(NDArray, nk.ppg_clean(abp, sampling_rate=SAMPLING_FREQUENCY, method='elgendi'))
# plot_single_signal(cleaned_abp, 'ABP [a.u.]', 'Cleaned ABP signal', TimeUnit.S)
# peaks = get_peaks(abp, PeaksMode.UP, SAMPLING_FREQUENCY)
# plot_single_signal_with_peaks(abp, peaks, 'ABP [mmHg]', 'ABP signal with detected peaks', TimeUnit.S)

# %%
# plot_multiple_signals_shared_x(
#     signals=[abp, etco2],
#     labels=['ABP [mmHg]', r'ETCO$_2$ [mmHg]'],
#     title='Preprocessed raw signals',
#     time_unit=TimeUnit.S,
# )

# %%
processed_data = data_processor.process(subject_data)

# %%
y = processed_data[cb]['etco2'][:200]
x = processed_data[cb]['hp'][:200]
print(len(x))
# %%

plot_multiple_signals_shared_x(
    signals=[x, y],
    labels=['X', 'Y'],
    title='Initial Signals',
    time_unit=None,
)

# %%
d = 1
tau = 1
fx = x[d * tau :]
px = get_deleyed_vector(x, d, tau)
py = get_deleyed_vector(y, d, tau)

plot_multiple_signals_shared_x(
    signals=[fx, px.T[0], py.T[0]],
    labels=['Future X', 'Past X', 'Past Y'],
    title='Embedded Signals',
    time_unit=None,
)

# %%

d = 1
tau = 2
fx = x[d * tau :]
px = get_deleyed_vector(x, d, tau)
py = get_deleyed_vector(y, d, tau)

plot_multiple_signals_shared_x(
    signals=[fx, px.T[0], py.T[0]],
    labels=['Future X', 'Past X', 'Past Y'],
    title=r'Embedded Signals, $\tau=2$',
    time_unit=None,
)

# %%
d = 2
tau = 1
fx = x[d * tau :]
px = get_deleyed_vector(x, d, tau)
py = get_deleyed_vector(y, d, tau)

plot_multiple_signals_shared_x(
    signals=[fx, px.T[0], px.T[1], py.T[0], py.T[1]],
    labels=['Future X', 'Past X 1', 'Past X 2', 'Past Y 1', 'Past Y 2'],
    title=r'Embedded Signals, $d=2$',
    time_unit=None,
)

# %%
d = 1
tau = 1
fx = get_future_vector(x, d, tau)
px = get_past_vectors(x, d, tau)
py = get_past_vectors(y, d, tau)

plot_multiple_signals_shared_x(
    signals=[fx, px.T[0], py.T[0]],
    labels=['Future X', 'Past X', 'Past Y'],
    title='Ordinary Rank Transformed Signals',
    time_unit=None,
)

# %%

a = np.column_stack([fx, px, py])
b = np.column_stack([px])
c = np.column_stack([fx, px])
d = np.column_stack([px, py])
partitions = dv_partition_nd(a)
plot_3d_partitions(partitions, px.T[0], py.T[0], fx, 'Past X', 'Past Y', 'Future X')

# %%
print(partitions)

# %%
dimensions = a.shape[1]
n_total = a.shape[0]
futureX_start, futureX_end = 0, 1
pastX_start, pastX_end = futureX_end, 1 + 1
pastY_end = dimensions

te: float = 0
for dv_part in partitions:
    na = dv_part['N']
    nb = get_points_from_range(b, dv_part, ranges=((pastX_start, pastX_end),))
    nc = get_points_from_range(c, dv_part, ranges=((futureX_start, pastX_end),))
    nd = get_points_from_range(d, dv_part, ranges=((pastX_start, pastY_end),))

    te += na / n_total * (np.log2(na * nb) - np.log2(nc * nd))

print(te)
