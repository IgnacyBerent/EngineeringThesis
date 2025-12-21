from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pingouin as pg

from src.common.constants import CONDITION_FIELD, ID_FIELD
from src.common.logger import logger
from src.plots import plot_boxplot_w_posthoc

_P_ADJUST = 'bonf'


class StatisticsAnalyzer:
    def __init__(self, csv_file_path: str | Path, order: list[str] | None = None) -> None:
        self.categories: list[str] = []
        self.data = self._load_data(csv_file_path, order)
        self._remove_subjects_with_nans()

    def _load_data(self, csv_file_path: str | Path, order: list[str] | None) -> pd.DataFrame:
        data = pd.read_csv(csv_file_path)
        self.categories = list(data[CONDITION_FIELD].unique()) if order is None else order
        data[CONDITION_FIELD] = pd.Categorical(data[CONDITION_FIELD], categories=self.categories, ordered=True)
        return data

    def _remove_subjects_with_nans(self) -> None:
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
        self._save_latex_table(rm, title)
        self._save_latex_table(posthoc, title + '_posthoc')

        plot_data = self.data.groupby(CONDITION_FIELD, sort=True)[field].apply(list).to_dict()
        plot_boxplot_w_posthoc(data=plot_data, title=title, y_label=field, posthoc=posthoc)
        # plot_paired_boxplot(
        #     data=self.data, field=field, within=CONDITION_FIELD, subject=ID_FIELD, order=self._data_order
        # )

    def post_hoc(self, field: str) -> pd.DataFrame:
        return pg.pairwise_tests(data=self.data, dv=field, within=CONDITION_FIELD, subject=ID_FIELD, padjust=_P_ADJUST)

    def compare(self, field1: str, field2: str) -> None:
        results = []
        for condition_value, subset in self.data.groupby(CONDITION_FIELD):
            subset = subset.copy()
            stats = subset[[field1, field2]].agg(['mean']).T
            ttest = pg.ttest(subset[field1], subset[field2], paired=True)
            summary_row = {
                'Condition': condition_value,
                f'Mean {field1}': stats.loc[field1, 'mean'],
                f'Mean {field2}': stats.loc[field2, 'mean'],
                'T': ttest.at['T-test', 'T'],
                'p-unc': ttest.at['T-test', 'p-val'],
            }
            results.append(summary_row)

        final_summary = pd.DataFrame(results)
        title = f'comparison_{field1}_vs_{field2}'
        self._save_latex_table(final_summary, title)

    def _save_latex_table(self, result: pd.DataFrame, title: str) -> None:
        columns_to_remove = [
            'Contrast',
            'Source',
            'Paired',
            'Parametric',
            'alternative',
            'BF10',
            'hedges',
            'p-adjust',
            'p-GG-corr',
            'sphericity',
            'W-spher',
            'p-spher',
        ]
        result_filtered = result.drop(columns=columns_to_remove, axis=1, errors='ignore')

        dof_cols = [col for col in result_filtered.columns if 'dof' in col.lower()]
        for col in dof_cols:
            if col in result_filtered.columns:
                dof_values = cast(pd.Series, pd.to_numeric(result_filtered[col], errors='coerce'))
                dof_values = dof_values.astype(int).astype(str)
                result_filtered[col] = '$' + dof_values + '$'

        p_value_cols = [col for col in result_filtered.columns if col in ['p-unc', 'p-corr', 'p-value']]
        for col in p_value_cols:
            if col in result_filtered.columns:
                p_values = cast(pd.Series, pd.to_numeric(result_filtered[col], errors='coerce'))
                result_filtered[col] = p_values.apply(self._format_p_value)

        numeric_cols = result_filtered.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            result_filtered[col] = result_filtered[col].round(3)
            result_filtered[col] = '$' + result_filtered[col].astype(str) + '$'

        result_filtered = result_filtered.astype(str).replace('_', ' ', regex=True)

        latex_table_content = result_filtered.to_latex(index=False)
        with open(f'results/{title}.txt', 'w') as file:
            file.write(latex_table_content)

    @staticmethod
    def _format_p_value(p: np.number) -> str:
        if p < 0.001:
            return r'$< 0.001$'
        if p < 0.01:
            return r'$< 0.01$'
        return f'${p:.3f}$'
