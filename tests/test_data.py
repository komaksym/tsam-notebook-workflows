from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsam_workflows.config import DatasetSpec
from tsam_workflows.data import (
    daily_period_timesteps,
    expected_hourly_index,
    filter_countries,
    load_dataset,
    normalize_country_args,
    set_snapshot_index,
    validate_df,
)


def test_expected_hourly_index_handles_leap_year() -> None:
    index = expected_hourly_index(2024, "h")

    assert len(index) == 8784
    assert index[0] == pd.Timestamp("2024-01-01 00:00")
    assert index[-1] == pd.Timestamp("2024-12-31 23:00")


def test_load_dataset_normalizes_raw_country_headers(tmp_path: Path) -> None:
    path = tmp_path / "demand.csv"
    path.write_text(
        "snapshot;DE;FR\n"
        "01.01.2025 00:00;10;20\n"
        "01.01.2025 01:00;11;21\n",
        encoding="utf-8",
    )
    spec = DatasetSpec(path=path, separator=";", feature="demand", feature_group="demand")

    df = load_dataset("demand", spec, 2025, "snapshot")

    assert list(df.columns) == ["snapshot", "DE_demand_2025", "FR_demand_2025"]


def test_load_dataset_accepts_canonical_headers(tmp_path: Path) -> None:
    path = tmp_path / "solar.csv"
    path.write_text(
        "snapshot,DE_solar_2025\n"
        "01.01.2025 00:00,0.0\n"
        "01.01.2025 01:00,0.1\n",
        encoding="utf-8",
    )
    spec = DatasetSpec(
        path=path,
        separator=",",
        feature="solar",
        feature_group="capacity_factors",
        unit_interval=True,
    )

    df = load_dataset("solar", spec, 2025, "snapshot")

    assert list(df.columns) == ["snapshot", "DE_solar_2025"]


def test_load_dataset_rejects_duplicate_raw_headers(tmp_path: Path) -> None:
    path = tmp_path / "demand.csv"
    path.write_text(
        "snapshot;DE;DE\n"
        "01.01.2025 00:00;10;20\n",
        encoding="utf-8",
    )
    spec = DatasetSpec(path=path, separator=";", feature="demand", feature_group="demand")

    with pytest.raises(ValueError, match="duplicate raw CSV headers"):
        load_dataset("demand", spec, 2025, "snapshot")


def test_load_dataset_rejects_malformed_feature_header(tmp_path: Path) -> None:
    path = tmp_path / "demand.csv"
    path.write_text(
        "snapshot,Germany\n"
        "01.01.2025 00:00,10\n",
        encoding="utf-8",
    )
    spec = DatasetSpec(path=path, separator=",", feature="demand", feature_group="demand")

    with pytest.raises(ValueError, match="malformed feature column"):
        load_dataset("demand", spec, 2025, "snapshot")


def test_validate_df_rejects_missing_hour() -> None:
    index = expected_hourly_index(2025, "h").delete(5)
    df = pd.DataFrame({"DE_demand_2025": range(len(index))}, index=index)

    with pytest.raises(ValueError, match="regular sampling frequency"):
        validate_df(df, "feature_data", 2025)


def test_set_snapshot_index_uses_configured_timestamp_format() -> None:
    df = pd.DataFrame(
        {
            "time": ["2025-01-01 00:00:00", "2025-01-01 00:30:00"],
            "DE_demand_2025": [1.0, 2.0],
        }
    )

    indexed = set_snapshot_index(df, "time", "%Y-%m-%d %H:%M:%S")

    assert indexed.index.equals(
        pd.date_range("2025-01-01", periods=2, freq="30min", name="time")
    )


def test_validate_df_infers_complete_half_hour_frequency() -> None:
    index = pd.date_range(
        "2025-01-01",
        "2026-01-01",
        freq="30min",
        inclusive="left",
    )
    df = pd.DataFrame({"DE_demand_2025": range(len(index))}, index=index)

    frequency = validate_df(df, "feature_data", 2025)

    assert frequency == pd.Timedelta(minutes=30)
    assert daily_period_timesteps(frequency) == 48


def test_daily_period_timesteps_rejects_frequency_that_cannot_form_days() -> None:
    with pytest.raises(ValueError, match="does not divide evenly into 24 hours"):
        daily_period_timesteps(pd.Timedelta(minutes=7))


def test_filter_countries_normalizes_deduplicates_and_keeps_asymmetric_features() -> None:
    df = pd.DataFrame(
        {
            "DE_demand_2025": [1.0],
            "FR_demand_2025": [2.0],
            "DE_solar_2025": [0.5],
        }
    )

    filtered, selected = filter_countries(df, ["de", "DE"])

    assert selected == ("DE",)
    assert list(filtered.columns) == ["DE_demand_2025", "DE_solar_2025"]


def test_filter_countries_rejects_unknown_country() -> None:
    df = pd.DataFrame({"DE_demand_2025": [1.0]})

    with pytest.raises(ValueError, match="Unknown countries"):
        filter_countries(df, ["XX"])


def test_normalize_country_args_rejects_all_mixed_with_codes() -> None:
    with pytest.raises(ValueError, match="'all' cannot be combined"):
        normalize_country_args(["all", "DE"])
