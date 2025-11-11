from pathlib import Path

import pandas as pd
import pingouin as pg

from src.common.constants import CB_FILE_TYPE, CONDITION_FIELD, ID_FIELD
from src.plots.boxplots import plot_boxplot, plot_boxplot_w_posthoc

_P_ADJUST = 'bonf'


class StatisticsAnalyzer:
    def __init__(self, csv_file_path: str | Path) -> None:
        self._data_order = CB_FILE_TYPE.order()
        self.data = self._load_data(csv_file_path)

    def _load_data(self, csv_file_path: str | Path) -> pd.DataFrame:
        data = pd.read_csv(csv_file_path)
        data[CONDITION_FIELD] = pd.Categorical(data[CONDITION_FIELD], categories=self._data_order, ordered=True)
        return data

    def do_rm_anova_test(self, field: str) -> None:
        rm = pg.rm_anova(
            data=self.data,
            dv=field,
            within=CONDITION_FIELD,
            subject=ID_FIELD,
        )
        posthoc = self.post_hoc(field)

        plot_data = self.data.groupby(CONDITION_FIELD, sort=True)[field].apply(list).to_dict()
        plot_boxplot_w_posthoc(data=plot_data, title=field, y_label=field, posthoc=posthoc)

    def post_hoc(self, field: str) -> pd.DataFrame:
        return pg.pairwise_tests(data=self.data, dv=field, within=CONDITION_FIELD, subject=ID_FIELD, padjust=_P_ADJUST)
