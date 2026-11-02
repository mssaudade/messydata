from __future__ import annotations

from datetime import date, timedelta
from typing import Union

import pandas as pd

from .distributions import Reference
from .generator import generate_data
from .injector import inject_anomalies
from .schema import DatasetSchema, MultiDatasetSchema, load_config


def _as_date(d: Union[date, str]) -> date:
    return date.fromisoformat(d) if isinstance(d, str) else d


class Pipeline:
    def __init__(self, schema: DatasetSchema, continue_ids: bool = True):
        self._schema = schema
        # True: ids keep counting across the days of run_date_range (they used to restart each day,
        # repeating every primary key). False restores the old behaviour.
        self.continue_ids = continue_ids

    @classmethod
    def from_config(cls, path: str) -> Union["Pipeline", "MultiPipeline"]:
        """A Pipeline for a one-dataset config, or a MultiPipeline when the config has ``datasets:``."""
        config = load_config(path)
        if isinstance(config, MultiDatasetSchema):
            return MultiPipeline(config)
        return cls(config)

    def run(self, n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
        df = generate_data(self._schema, n_rows=n_rows, seed=seed)
        df = inject_anomalies(self._schema, df)
        return df

    def run_for_date(
        self,
        target_date: Union[date, str],
        n_rows: int = 1000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate data pinned to a single date.

        Requires exactly one field with ``temporal=True`` in the schema.
        That field's distribution is overridden to emit ``target_date`` for every row.
        """
        df = self._generate_for_date(target_date, n_rows, seed)
        df = inject_anomalies(self._schema, df)
        return df

    def _generate_for_date(self, target_date, n_rows, seed, seq_state=None) -> pd.DataFrame:
        temporal_fields = [f for f in self._schema.fields if f.temporal]
        if not temporal_fields:
            raise ValueError(
                "run_for_date requires at least one field with temporal=True in the schema."
            )
        return generate_data(self._schema, n_rows=n_rows, seed=seed, date_override=_as_date(target_date),
                             seq_state=seq_state)

    def run_date_range(
        self,
        start: Union[date, str],
        end: Union[date, str],
        rows_per_day: int = 1000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate data for every calendar day in [start, end] inclusive.

        Each day gets its own generation pass with the temporal field pinned to that date.
        The seed is offset per day so rate-based anomaly patterns vary across days. Anomalies
        with an exact ``count``, and the time anomalies (volume_drop, stale), apply once to the
        whole range, so a count of 7 means 7 rows in the output, not 7 per day.
        """
        frames = self._range_frames(start, end, rows_per_day, seed, inject=True)
        df = pd.concat(frames, ignore_index=True)
        return inject_anomalies(self._schema, df, once=True)

    def _range_frames(self, start, end, rows_per_day, seed, inject: bool) -> list[pd.DataFrame]:
        start, end = _as_date(start), _as_date(end)
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        frames: list[pd.DataFrame] = []
        current, day_idx = start, 0
        # Ids keep counting across days; the temporal field is pinned per day anyway.
        seq_state: dict = {} if self.continue_ids else None
        while current <= end:
            day_seed = seed + day_idx if seed is not None else None
            df = self._generate_for_date(current, rows_per_day, day_seed, seq_state)
            if inject:
                df = inject_anomalies(self._schema, df, once=False)
            frames.append(df)
            current += timedelta(days=1)
            day_idx += 1
        return frames


class MultiPipeline:
    """Generates several related datasets from one config.

    Parents are generated first; a ``type: reference`` field draws from its parent's clean
    values, so every link is valid until a ``broken_reference`` anomaly breaks some on purpose.
    """

    def __init__(self, config: MultiDatasetSchema):
        self._config = config

    def _bound(self, schema: DatasetSchema, clean: dict[str, pd.DataFrame]) -> tuple[DatasetSchema, dict]:
        fields, parent_values = [], {}
        for f in schema.fields:
            if isinstance(f.distribution, Reference):
                values = clean[f.distribution.dataset][f.distribution.column].dropna().unique().tolist()
                f = f.model_copy(update={"distribution": f.distribution.bind(values)})
                parent_values[f.name] = values
            fields.append(f)
        return schema.model_copy(update={"fields": fields}), parent_values

    def run(self, n_rows: Union[int, dict[str, int]] = 1000, seed: int = 42) -> dict[str, pd.DataFrame]:
        """n_rows is one number for every dataset, or a dict of dataset name to rows."""
        clean: dict[str, pd.DataFrame] = {}
        out: dict[str, pd.DataFrame] = {}
        for i, schema in enumerate(self._config.ordered()):
            bound, parent_values = self._bound(schema, clean)
            n = n_rows[schema.name] if isinstance(n_rows, dict) else n_rows
            df = generate_data(bound, n_rows=n, seed=seed + i)
            clean[schema.name] = df.copy()
            out[schema.name] = inject_anomalies(bound, df, parent_values=parent_values)
        return out

    def run_date_range(
        self,
        start: Union[date, str],
        end: Union[date, str],
        rows_per_day: Union[int, dict[str, int]] = 1000,
        seed: int = 42,
    ) -> dict[str, pd.DataFrame]:
        """Date-range mode for every dataset that has a temporal field; the others use run().

        rows_per_day is per dataset when given as a dict; a dataset without a temporal field
        reads its entry as a total row count.
        """
        clean: dict[str, pd.DataFrame] = {}
        out: dict[str, pd.DataFrame] = {}
        for i, schema in enumerate(self._config.ordered()):
            bound, parent_values = self._bound(schema, clean)
            n = rows_per_day[schema.name] if isinstance(rows_per_day, dict) else rows_per_day
            pipe = Pipeline(bound, continue_ids=True)
            if any(f.temporal for f in bound.fields):
                df = pd.concat(pipe._range_frames(start, end, n, seed + i * 100000, inject=False), ignore_index=True)
            else:
                df = generate_data(bound, n_rows=n, seed=seed + i)
            clean[schema.name] = df.copy()
            df = inject_anomalies(bound, df, once=False)
            out[schema.name] = inject_anomalies(bound, df, once=True, parent_values=parent_values)
        return out
