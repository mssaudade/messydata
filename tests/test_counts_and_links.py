"""Exact-count anomalies, linked datasets, broken references and time anomalies."""

import pandas as pd
import pytest
from pydantic import ValidationError

from messydata import (
    AnomalySpec, DatasetSchema, FieldSpec, MultiDatasetSchema, MultiPipeline, Pipeline,
    Reference, Sequential, Uniform, WeightedChoice,
)

ONE = Uniform(min=1, max=1)


def _orders(anomalies=()):
    return DatasetSchema(
        name="orders", primary_key="order_id", records_per_primary_key=ONE,
        fields=[
            FieldSpec(name="order_id", dtype="int32", unique_per_id=True, nullable=False,
                      distribution=Sequential(start=1)),
            FieldSpec(name="ordered_at", temporal=True, nullable=False,
                      distribution=Sequential(start="2026-01-01")),
            FieldSpec(name="status", distribution=WeightedChoice(values=["a", "b"], weights=[0.5, 0.5])),
        ],
        anomalies=list(anomalies),
    )


def _payments(anomalies=()):
    return DatasetSchema(
        name="payments", primary_key="payment_id", records_per_primary_key=ONE,
        fields=[
            FieldSpec(name="payment_id", dtype="int32", unique_per_id=True, nullable=False,
                      distribution=Sequential(start=1)),
            FieldSpec(name="order_id", nullable=False, distribution=Reference(dataset="orders", column="order_id")),
        ],
        anomalies=list(anomalies),
    )


def test_count_duplicates_is_exact_in_date_range():
    df = Pipeline(_orders([AnomalySpec(name="duplicate_values", count=7)])).run_date_range(
        "2026-01-01", "2026-01-30", rows_per_day=20)
    assert len(df) == 30 * 20 + 7
    assert df["order_id"].duplicated().sum() == 7


def test_count_missing_is_exact_and_skips_non_nullable():
    df = Pipeline(_orders([AnomalySpec(name="missing_values", count=12)])).run(n_rows=500)
    assert df["status"].isna().sum() == 12
    assert df["order_id"].notna().all() and df["ordered_at"].notna().all()


def test_rate_missing_any_skips_non_nullable():
    df = Pipeline(_orders([AnomalySpec(name="missing_values", rate=0.3)])).run(n_rows=500)
    assert df["order_id"].notna().all()


def test_rate_or_count_required():
    with pytest.raises(ValidationError):
        AnomalySpec(name="duplicate_values")
    with pytest.raises(ValidationError):
        AnomalySpec(name="duplicate_values", rate=0.1, count=3)


def test_references_are_valid_until_broken():
    cfg = MultiDatasetSchema(name="shop", datasets=[_payments(), _orders()])
    out = MultiPipeline(cfg).run(n_rows={"orders": 300, "payments": 200})
    assert set(out["payments"]["order_id"]).issubset(set(out["orders"]["order_id"]))


def test_broken_reference_count_is_exact():
    cfg = MultiDatasetSchema(name="shop", datasets=[
        _orders(), _payments([AnomalySpec(name="broken_reference", count=4)])])
    out = MultiPipeline(cfg).run_date_range("2026-01-01", "2026-01-10",
                                            rows_per_day={"orders": 30, "payments": 250})
    orphans = ~out["payments"]["order_id"].isin(out["orders"]["order_id"])
    assert orphans.sum() == 4


def test_unknown_reference_rejected():
    bad = _payments()
    bad.fields[1] = FieldSpec(name="order_id", distribution=Reference(dataset="nope", column="order_id"))
    with pytest.raises(ValidationError):
        MultiDatasetSchema(name="x", datasets=[_orders(), bad])


def test_volume_drop_and_stale():
    df = Pipeline(_orders([
        AnomalySpec(name="volume_drop", date="2026-01-05", keep=0.25),
        AnomalySpec(name="stale", after="2026-01-08"),
    ])).run_date_range("2026-01-01", "2026-01-10", rows_per_day=40)
    per_day = pd.to_datetime(df["ordered_at"]).dt.strftime("%Y-%m-%d").value_counts()
    assert per_day["2026-01-05"] == 10
    assert per_day["2026-01-04"] == 40
    assert "2026-01-09" not in per_day and "2026-01-10" not in per_day


def test_same_seed_same_output():
    cfg = MultiDatasetSchema(name="shop", datasets=[
        _orders([AnomalySpec(name="duplicate_values", count=3)]),
        _payments([AnomalySpec(name="broken_reference", count=2)])])
    a = MultiPipeline(cfg).run(n_rows=100, seed=7)
    b = MultiPipeline(cfg).run(n_rows=100, seed=7)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_example_shop_config(tmp_path):
    from click.testing import CliRunner
    from messydata.cli import main
    res = CliRunner().invoke(main, ["generate", "examples/shop_config.yaml", "--start-date", "2026-08-04",
                                    "--end-date", "2026-11-01", "--rows", "60", "--output", str(tmp_path)])
    assert res.exit_code == 0, res.output
    orders = pd.read_csv(tmp_path / "orders.csv")
    payments = pd.read_csv(tmp_path / "payments.csv")
    customers = pd.read_csv(tmp_path / "customers.csv")
    assert orders["order_id"].duplicated().sum() == 7
    assert (~payments["order_id"].isin(orders["order_id"])).sum() == 4
    assert customers["email"].isna().sum() == 12
