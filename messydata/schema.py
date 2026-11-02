from __future__ import annotations

from datetime import date
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, model_validator

from .distributions import ContinuousDistribution, Distribution, Reference

# Anomalies that change rows over time instead of affecting a share of rows.
TIME_ANOMALIES = {"volume_drop", "stale"}


class FieldSpec(BaseModel):
    name: str
    dtype: str = "object"
    distribution: Distribution
    unique_per_id: bool = False
    nullable: bool = True
    temporal: bool = False


class AnomalySpec(BaseModel):
    name: Literal[
        "missing_values", "duplicate_values", "invalid_category", "invalid_date", "outliers",
        "broken_reference", "volume_drop", "stale",
    ]
    prob: float = 1.0
    rate: Optional[float] = None
    # Exact number of rows (or cells, for missing_values) to affect, instead of a rate.
    # Applied once to the whole output, also in date-range mode, so the count stays exact.
    count: Optional[int] = None
    columns: Union[Literal["any"], list[str]] = "any"
    distribution: Optional[Distribution] = None  # used by outliers
    date: Optional[str] = None   # volume_drop: the day that loses rows
    keep: Optional[float] = None  # volume_drop: fraction of that day's rows kept
    after: Optional[str] = None   # stale: no rows dated after this day

    @model_validator(mode="after")
    def _check_params(self):
        if self.name == "volume_drop":
            if self.date is None or self.keep is None:
                raise ValueError("volume_drop needs date and keep")
            date.fromisoformat(self.date)
        elif self.name == "stale":
            if self.after is None:
                raise ValueError("stale needs after")
            date.fromisoformat(self.after)
        elif (self.rate is None) == (self.count is None):
            raise ValueError(f"{self.name} needs exactly one of rate or count")
        return self

    @property
    def applies_once(self) -> bool:
        """True when the anomaly must run on the final output rather than per generation pass."""
        return self.count is not None or self.name in TIME_ANOMALIES or self.name == "broken_reference"


class DatasetSchema(BaseModel):
    name: str
    primary_key: str = "id"
    records_per_primary_key: ContinuousDistribution
    fields: list[FieldSpec]
    anomalies: list[AnomalySpec] = []

    def references(self) -> list[Reference]:
        return [f.distribution for f in self.fields if isinstance(f.distribution, Reference)]

    @classmethod
    def from_yaml(cls, path: str) -> DatasetSchema:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        # Strip legacy use_case wrapper if present
        if "use_case" in raw:
            raw = raw["use_case"]
        return cls.model_validate(raw)


class MultiDatasetSchema(BaseModel):
    """Several related datasets in one config, linked by ``type: reference`` fields."""

    name: str
    datasets: list[DatasetSchema]

    @model_validator(mode="after")
    def _check_references(self):
        by_name = {d.name: d for d in self.datasets}
        if len(by_name) != len(self.datasets):
            raise ValueError("dataset names must be unique")
        for d in self.datasets:
            for ref in d.references():
                parent = by_name.get(ref.dataset)
                if parent is None:
                    raise ValueError(f"{d.name} references unknown dataset {ref.dataset}")
                if ref.column not in {f.name for f in parent.fields}:
                    raise ValueError(f"{d.name} references unknown column {ref.dataset}.{ref.column}")
        self.ordered()  # raises on a cycle
        return self

    def ordered(self) -> list[DatasetSchema]:
        """Datasets with every parent before the datasets that reference it."""
        by_name = {d.name: d for d in self.datasets}
        done: list[str] = []
        visiting: set[str] = set()

        def visit(name: str):
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"reference cycle through {name}")
            visiting.add(name)
            for ref in by_name[name].references():
                visit(ref.dataset)
            visiting.discard(name)
            done.append(name)

        for d in self.datasets:
            visit(d.name)
        return [by_name[n] for n in done]

    @classmethod
    def from_yaml(cls, path: str) -> MultiDatasetSchema:
        with open(path, "r") as f:
            return cls.model_validate(yaml.safe_load(f))


def load_config(path: str) -> Union[DatasetSchema, MultiDatasetSchema]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, dict) and "datasets" in raw:
        return MultiDatasetSchema.model_validate(raw)
    if "use_case" in raw:
        raw = raw["use_case"]
    return DatasetSchema.model_validate(raw)
