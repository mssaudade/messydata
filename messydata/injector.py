from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .distributions import Reference
from .schema import AnomalySpec, DatasetSchema


def _n(total: int, anomaly: AnomalySpec) -> int:
    """How many rows (or cells) an anomaly touches: its exact count, or its rate of the total."""
    if anomaly.count is not None:
        return min(anomaly.count, total)
    return int(total * anomaly.rate)


def inject_duplicates(df: pd.DataFrame, rate: float, n: Optional[int] = None) -> pd.DataFrame:
    n_dup = int(len(df) * rate) if n is None else n
    dup_rows = df.sample(n=n_dup, replace=True)
    return pd.concat([df, dup_rows], ignore_index=True)


def inject_missing(
    df: pd.DataFrame, rate: float, cols: list[str] | None = None, n: Optional[int] = None
) -> pd.DataFrame:
    target_cols = [c for c in (df.columns if cols is None else cols) if c in df.columns]
    n_cells = int(len(df) * len(target_cols) * rate) if n is None else n
    if not target_cols or n_cells == 0:
        return df
    col_indices = [df.columns.get_loc(c) for c in target_cols]
    if n is None:
        for _ in range(n_cells):
            i = np.random.randint(0, len(df))
            j = col_indices[np.random.randint(0, len(col_indices))]
            df.iat[i, j] = np.nan
        return df
    # Exact count: pick distinct non-empty cells so the number of new NaNs is exactly n.
    candidates = [(i, j) for j in col_indices for i in np.flatnonzero(df.iloc[:, j].notna().to_numpy())]
    picks = np.random.choice(len(candidates), size=min(n_cells, len(candidates)), replace=False)
    for k in picks:
        i, j = candidates[k]
        df.iat[i, j] = np.nan
    return df


def inject_invalid_category(
    df: pd.DataFrame, rate: float, cat_cols: list[str], bad_value: str = "INVALID", n: Optional[int] = None
) -> pd.DataFrame:
    df = df.copy()
    n_bad = int(len(df) * rate) if n is None else n
    for col in cat_cols:
        if col not in df.columns:
            continue
        idx = np.random.choice(df.index, size=min(n_bad, len(df)), replace=False)
        df[col] = df[col].astype(object)
        df.loc[idx, col] = bad_value
    return df


def inject_invalid_date(
    df: pd.DataFrame, rate: float, date_cols: list[str], bad_value: str = "9999-99-99", n: Optional[int] = None
) -> pd.DataFrame:
    df = df.copy()
    n_bad = int(len(df) * rate) if n is None else n
    for col in date_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        idx = np.random.choice(df.index, size=min(n_bad, len(df)), replace=False)
        df.loc[idx, col] = bad_value
    return df


def inject_outliers(df: pd.DataFrame, rate: float, cols: list[str], distribution, n: Optional[int] = None) -> pd.DataFrame:
    df = df.copy()
    n_bad = int(len(df) * rate) if n is None else n
    for col in cols:
        if col not in df.columns:
            continue
        idx = np.random.choice(df.index, size=min(n_bad, len(df)), replace=False)
        values = distribution.sample(len(idx))
        df.loc[idx, col] = values.astype(df[col].dtype)
    return df


def inject_broken_reference(df: pd.DataFrame, n: int, cols: list[str], parent_values: dict[str, list]) -> pd.DataFrame:
    """Point n rows at parent keys that do not exist (orphans), e.g. a payment for a missing order."""
    df = df.copy()
    for col in cols:
        parents = parent_values[col]
        numeric = all(isinstance(v, (int, np.integer)) for v in parents)
        idx = np.random.choice(df.index, size=min(n, len(df)), replace=False)
        if numeric:
            start = int(max(parents)) + 1000
            bad = list(range(start, start + len(idx)))
        else:
            bad = [f"missing-{k + 1}" for k in range(len(idx))]
        df[col] = df[col].astype(object)
        df.loc[idx, col] = bad
    return df


def _temporal_column(schema: DatasetSchema) -> str:
    cols = [f.name for f in schema.fields if f.temporal]
    if not cols:
        raise ValueError("volume_drop and stale need a field with temporal: true")
    return cols[0]


def _day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def inject_volume_drop(df: pd.DataFrame, temporal: str, day: str, keep: float) -> pd.DataFrame:
    """Keep only a fraction of one day's rows, the drop a volume monitor should flag."""
    on_day = df.index[_day(df[temporal]) == date.fromisoformat(day)]
    n_drop = len(on_day) - int(round(len(on_day) * keep))
    if n_drop <= 0:
        return df
    drop = np.random.choice(on_day, size=n_drop, replace=False)
    return df.drop(index=drop).reset_index(drop=True)


def inject_stale(df: pd.DataFrame, temporal: str, after: str) -> pd.DataFrame:
    """Remove every row dated after a day, the gap a freshness check should flag."""
    keep = ~(_day(df[temporal]) > date.fromisoformat(after))
    return df[keep.to_numpy()].reset_index(drop=True)


def inject_anomalies(
    schema: DatasetSchema,
    df: pd.DataFrame,
    once: Optional[bool] = None,
    parent_values: Optional[dict[str, list]] = None,
) -> pd.DataFrame:
    """Apply the schema's anomalies to df.

    once=None applies all of them; once=False only the rate-based ones (run per generation pass);
    once=True only those that must touch the final output once (exact counts, time anomalies,
    broken references), so a count of 7 means 7 in the whole output.
    """
    non_nullable = {f.name for f in schema.fields if not f.nullable}
    for anomaly in schema.anomalies:
        if once is not None and anomaly.applies_once != once:
            continue
        if not np.random.random() > 1 - anomaly.prob:
            continue
        cols = anomaly.columns
        exact = anomaly.count if anomaly.count is not None else None
        if anomaly.name == "missing_values":
            # Only nullable columns can go missing, as the field spec promises.
            pool = list(df.columns) if cols == "any" else cols
            pool = [c for c in pool if c not in non_nullable]
            df = inject_missing(df, anomaly.rate or 0.0, cols=pool, n=exact)
        elif anomaly.name == "duplicate_values":
            df = inject_duplicates(df, anomaly.rate or 0.0, n=exact)
        elif anomaly.name == "invalid_category":
            if cols == "any":
                cols = list(df.select_dtypes(include="object").columns)
            df = inject_invalid_category(df, anomaly.rate or 0.0, cat_cols=cols, n=exact)
        elif anomaly.name == "invalid_date":
            if cols == "any":
                cols = [c for c in df.columns if "date" in c.lower()]
            df = inject_invalid_date(df, anomaly.rate or 0.0, date_cols=cols, n=exact)
        elif anomaly.name == "outliers":
            if cols == "any":
                cols = list(df.select_dtypes(include="number").columns)
            df = inject_outliers(df, anomaly.rate or 0.0, cols=cols, distribution=anomaly.distribution, n=exact)
        elif anomaly.name == "broken_reference":
            ref_cols = [f.name for f in schema.fields if isinstance(f.distribution, Reference)]
            cols = ref_cols if cols == "any" else cols
            if not parent_values or any(c not in parent_values for c in cols):
                raise ValueError("broken_reference needs reference fields in a multi-dataset config")
            df = inject_broken_reference(df, _n(len(df), anomaly), cols, parent_values)
        elif anomaly.name == "volume_drop":
            df = inject_volume_drop(df, _temporal_column(schema), anomaly.date, anomaly.keep)
        elif anomaly.name == "stale":
            df = inject_stale(df, _temporal_column(schema), anomaly.after)
    return df
