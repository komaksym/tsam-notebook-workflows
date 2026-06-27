from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import tsam

from conftest import write_feature_csv
from tsam_workflows.config import (
    DatasetSpec,
    GroupedWorkflowConfig,
    SUPPORTED_CLUSTER_METHODS,
)
from tsam_workflows.grouped import (
    build_hourly_metadata,
    build_representative_days,
    run_grouped_workflow,
    sort_group_id,
    validate_group_feasibility,
)


def test_build_hourly_metadata_assigns_month_day_type_and_group() -> None:
    index = pd.date_range("2025-01-03", periods=48, freq="h")
    features = pd.DataFrame({"DE_demand_2025": range(48)}, index=index)

    metadata = build_hourly_metadata(features, {"Saturday", "Sunday"})

    assert metadata.loc["2025-01-03 00:00", "day_type"] == "working"
    assert metadata.loc["2025-01-04 00:00", "day_type"] == "non-working"
    assert metadata.loc["2025-01-04 00:00", "group_id"] == "2025_1_non-working"


def test_sort_group_id_uses_calendar_order() -> None:
    groups = ["2025_10_working", "2025_2_non-working", "2025_2_working"]

    assert sorted(groups, key=sort_group_id) == [
        "2025_2_working",
        "2025_2_non-working",
        "2025_10_working",
    ]


def test_validate_group_feasibility_rejects_impossible_cluster_counts(hourly_2025: pd.DatetimeIndex) -> None:
    features = pd.DataFrame({"DE_demand_2025": range(len(hourly_2025))}, index=hourly_2025)
    metadata = build_hourly_metadata(features, {"Saturday", "Sunday"})
    daily = metadata.assign(date=metadata.index.normalize()).drop_duplicates("date")

    with pytest.raises(ValueError, match="at least as many days"):
        validate_group_feasibility(daily, {"working": 30, "non-working": 2})


@pytest.mark.parametrize("method", SUPPORTED_CLUSTER_METHODS)
def test_supported_cluster_methods_use_medoid_representation(method: str) -> None:
    cluster = tsam.ClusterConfig(method=method, representation="medoid")

    assert cluster.method == method
    assert cluster.representation == "medoid"


def test_run_grouped_workflow_preserves_output_shapes_and_schemas(
    tmp_path: Path,
    hourly_2025: pd.DatetimeIndex,
) -> None:
    demand_path = tmp_path / "demand.csv"
    solar_path = tmp_path / "solar.csv"
    demand = [float(ts.month * 10 + ts.hour) for ts in hourly_2025]
    solar = [0.0 if ts.hour < 6 or ts.hour > 18 else 0.5 for ts in hourly_2025]
    write_feature_csv(demand_path, hourly_2025, {"DE": demand}, sep=";")
    write_feature_csv(solar_path, hourly_2025, {"DE_solar_2025": solar}, sep=",")
    specs = {
        "demand": DatasetSpec(demand_path, ";", "demand", "demand"),
        "solar": DatasetSpec(solar_path, ",", "solar", "capacity_factors", True),
    }
    config = GroupedWorkflowConfig(
        year=2025,
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        countries=("DE",),
        working_clusters=1,
        non_working_clusters=1,
        cluster_method="hierarchical",
    )

    result = run_grouped_workflow(config, specs)

    assert len(result.tsam_results_by_group) == 24
    assert len(result.day_assignments_df) == 365
    assert len(result.representative_days) == 24
    assert len(result.reduced_hourly_df) == 24 * 24
    assert list(result.representative_days.columns) == [
        "selected_medoid_date",
        "representative_id",
        "group_id",
        "month",
        "day_type",
        "cluster_id",
        "cluster_weight",
    ]


def test_build_representative_days_schema() -> None:
    reduced = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "representative_id": ["2025_1_working_c0", "2025_1_working_c0"],
            "group_id": ["2025_1_working", "2025_1_working"],
            "month": [1, 1],
            "day_type": ["working", "working"],
            "cluster_id": [0, 0],
            "cluster_weight": [3, 3],
        }
    )

    representative_days = build_representative_days(reduced)

    assert representative_days.loc[0, "selected_medoid_date"] == pd.Timestamp("2025-01-01")
    assert len(representative_days) == 1

