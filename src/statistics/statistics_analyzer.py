from pathlib import Path
from typing import cast

import pandas as pd
import pingouin as pg

from src.common.constants import CB_FILE_TYPE, CONDITION_FIELD, ID_FIELD
from src.common.logger import logger
from src.plots import plot_boxplot_w_posthoc, plot_paired_boxplot

_P_ADJUST = 'bonf'


class StatisticsAnalyzer:
    def __init__(self, csv_file_path: str | Path, order: list[str] | None = None) -> None:
        self._data_order = order
        self.data = self._load_data(csv_file_path)
        self._remove_patients_with_nans()

    def _load_data(self, csv_file_path: str | Path) -> pd.DataFrame:
        data = pd.read_csv(csv_file_path)
        if self._data_order is None:
            self._data_orderorder = data[CONDITION_FIELD].unique()
        data[CONDITION_FIELD] = pd.Categorical(data[CONDITION_FIELD], categories=self._data_order, ordered=True)
        return data

    def _remove_patients_with_nans(self) -> None:
        rows_with_nan = cast(pd.DataFrame, self.data[self.data.isnull().any(axis=1)])
        pids_to_remove = sorted(rows_with_nan['pid'].unique())
        logger.warning(f'PIDs that have a None/NaN value in at least one row: {pids_to_remove}')
        self.data = cast(pd.DataFrame, self.data[~self.data['pid'].isin(pids_to_remove)]).reset_index(drop=True)

    def do_rm_anova_test(self, field: str, title: str) -> None:
        rm = pg.rm_anova(
            data=self.data,
            dv=field,
            within=CONDITION_FIELD,
            subject=ID_FIELD,
        )
        posthoc = self.post_hoc(field)

        plot_data = self.data.groupby(CONDITION_FIELD, sort=True)[field].apply(list).to_dict()
        plot_boxplot_w_posthoc(data=plot_data, title=title, y_label=field, posthoc=posthoc)
        # plot_paired_boxplot(
        #     data=self.data, field=field, within=CONDITION_FIELD, subject=ID_FIELD, order=self._data_order
        # )

    def post_hoc(self, field: str) -> pd.DataFrame:
        return pg.pairwise_tests(data=self.data, dv=field, within=CONDITION_FIELD, subject=ID_FIELD, padjust=_P_ADJUST)
