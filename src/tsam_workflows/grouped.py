"""Grouped month/day-type TSAM aggregation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import tsam

from tsam_workflows.config import ClusterMethod, DatasetSpec, GroupedWorkflowConfig
from tsam_workflows.data import (
    build_dataset_coverage,
    daily_period_timesteps,
    filter_countries,
    join_datasets,
    load_datasets,
    set_snapshot_index,
    validate_df,
    validate_loaded_columns,
    validate_unit_interval,
)

DAY_TYPE_SORT_ORDER: dict[str, int] = {"working": 0, "non-working": 1}
DAILY_PERIOD_DURATION_HOURS = 24
NUMERICAL_TOLERANCE = 1e-9
REPRESENTATIVE_DAY_COLUMNS: list[str] = [
    "selected_medoid_date",
    "representative_id",
    "group_id",
    "month",
    "day_type",
    "cluster_id",
    "cluster_weight",
]


@dataclass(frozen=True)
class GroupedWorkflowResult:
    """Outputs and diagnostics produced by the grouped TSAM workflow."""

    reduced_hourly_df: pd.DataFrame
    representative_days: pd.DataFrame
    day_assignments_df: pd.DataFrame
    tsam_results_by_group: dict[str, Any]
    dataset_coverage: pd.DataFrame
    group_accuracy: pd.DataFrame
    feature_columns_by_country_and_group: dict[str, dict[str, list[str]]]
    feature_group_by_feature: dict[str, str]
    selected_countries: tuple[str, ...]
    group_ids: list[str]
    sampling_frequency: pd.Timedelta
    period_timesteps: int
    preserve_column_means: bool


def build_hourly_metadata(
    feature_data: pd.DataFrame,
    non_working_weekdays: set[str] | frozenset[str],
) -> pd.DataFrame:
    """Add calendar and clustering-group metadata to hourly features."""
    result = feature_data.copy()
    snapshot_index = result.index
    if not isinstance(snapshot_index, pd.DatetimeIndex):
        raise TypeError("feature_data index must be a DatetimeIndex")

    result["date"] = snapshot_index.date
    result["month"] = snapshot_index.month
    result["day_of_month"] = snapshot_index.day
    result["weekday"] = snapshot_index.day_name()
    result["day_type"] = "working"
    result.loc[result["weekday"].isin(non_working_weekdays), "day_type"] = (
        "non-working"
    )
    result["group_id"] = (
        pd.Series(snapshot_index.year, index=snapshot_index).astype(str)
        + "_"
        + result["month"].astype(str)
        + "_"
        + result["day_type"]
    )
    return result


def build_daily_metadata(hourly_data_with_metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one metadata row per original day."""
    return (
        hourly_data_with_metadata.assign(
            date=hourly_data_with_metadata.index.normalize()
        )
        .drop_duplicates(subset="date")
        .set_index("date")
    )


def validate_group_feasibility(
    daily_metadata: pd.DataFrame,
    requested_clusters_by_day_type: Mapping[str, int],
) -> None:
    """Validate the 12 month x day-type groups and cluster counts."""
    month_order = range(1, 13)
    expected_group_count = len(month_order) * len(requested_clusters_by_day_type)
    actual_group_count = daily_metadata["group_id"].nunique()
    if actual_group_count != expected_group_count:
        raise ValueError(
            f"Expected {expected_group_count} groups, got {actual_group_count}"
        )

    month_day_type_counts = (
        daily_metadata.groupby(["month", "day_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=month_order,
            columns=requested_clusters_by_day_type.keys(),
            fill_value=0,
        )
    )
    enough = month_day_type_counts.ge(
        pd.Series(requested_clusters_by_day_type),
        axis="columns",
    )
    if not enough.all().all():
        raise ValueError(
            "Each month/day-type group must have at least as many days as "
            "requested clusters"
        )


def sort_group_id(group_id: str) -> tuple[int, int, int]:
    """Sort groups by year, month, then working/non-working day type."""
    year, month, day_type = group_id.split("_", maxsplit=2)
    return int(year), int(month), DAY_TYPE_SORT_ORDER[day_type]


def add_group_day_number(group_data_with_metadata: pd.DataFrame) -> pd.DataFrame:
    """Add a 1-based day number inside one month/day-type group."""
    result = group_data_with_metadata.copy()
    result["group_day_number"] = (
        result["day_of_month"].ne(result["day_of_month"].shift()).cumsum()
    )
    return result


def slice_group_data(
    group_id: str,
    data_with_metadata: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice one month/day-type group and return TSAM features plus metadata."""
    group_data_with_metadata = data_with_metadata.loc[
        data_with_metadata["group_id"] == group_id
    ].copy()
    if group_data_with_metadata.empty:
        raise ValueError(f"{group_id}: group has no rows")
    group_data_with_metadata = add_group_day_number(group_data_with_metadata)
    group_features = group_data_with_metadata.loc[:, feature_columns]
    return group_features, group_data_with_metadata


def get_n_clusters_for_group(
    group_data_with_metadata: pd.DataFrame,
    requested_clusters_by_day_type: Mapping[str, int],
) -> int:
    """Return requested cluster count for one month/day-type group."""
    day_types = group_data_with_metadata["day_type"].unique()
    if len(day_types) != 1:
        raise ValueError(f"Expected one day_type per group, got: {day_types}")
    day_type = str(day_types[0])
    try:
        return requested_clusters_by_day_type[day_type]
    except KeyError as exc:
        raise ValueError(f"Unknown day_type: {day_type}") from exc


def collect_representative_day_data(
    aggregation_result: Any,
    group_data_with_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build representative profiles and original-day assignments for one group.

    The exported representative values come from TSAM, while the selected
    medoid date remains as provenance for the calendar day behind each cluster.
    """
    representative_day_chunks: list[pd.DataFrame] = []
    representative_day_indices = aggregation_result.clustering.cluster_centers
    cluster_weights_by_id = aggregation_result.cluster_weights

    day_assignments = (
        aggregation_result.assignments.assign(
            date=aggregation_result.assignments.index.normalize()
        )
        .drop_duplicates("date")
        .set_index("date")
    )
    day_assignments = day_assignments.loc[:, ["cluster_idx"]].rename(
        columns={"cluster_idx": "cluster_id"}
    )
    day_assignments["cluster_weight"] = day_assignments["cluster_id"].map(
        lambda cluster_id: cluster_weights_by_id[cluster_id]
    )
    group_id = str(group_data_with_metadata["group_id"].iloc[0])
    day_assignments["group_id"] = group_id
    day_assignments["representative_id"] = (
        day_assignments["group_id"].astype(str)
        + "_c"
        + day_assignments["cluster_id"].astype(str)
    )

    cluster_ids = aggregation_result.period_index
    for cluster_id, representative_day_index in zip(
        cluster_ids,
        representative_day_indices,
        strict=True,
    ):
        representative_day_hours = group_data_with_metadata.loc[
            group_data_with_metadata["group_day_number"]
            == representative_day_index + 1
        ].copy()
        if representative_day_hours.empty:
            raise ValueError(
                f"{group_id}: representative day {representative_day_index} is empty"
            )
        representative_values = aggregation_result.cluster_representatives.xs(
            cluster_id,
            level=0,
        )
        if len(representative_values) != len(representative_day_hours):
            raise ValueError(
                f"{group_id}: representative {cluster_id} has "
                f"{len(representative_values)} timesteps; expected "
                f"{len(representative_day_hours)}"
            )
        missing_columns = set(representative_values.columns).difference(
            representative_day_hours.columns
        )
        if missing_columns:
            raise ValueError(
                f"{group_id}: representative columns missing from source metadata: "
                f"{sorted(missing_columns)}"
            )
        for column in representative_values.columns:
            representative_day_hours[column] = representative_values[column].to_numpy()
        representative_day_hours["cluster_id"] = cluster_id
        representative_day_hours["cluster_weight"] = cluster_weights_by_id[
            cluster_id
        ]
        representative_day_hours["representative_id"] = f"{group_id}_c{cluster_id}"
        representative_day_chunks.append(representative_day_hours)

    return pd.concat(representative_day_chunks), day_assignments


def build_cluster_config(method: ClusterMethod) -> tsam.ClusterConfig:
    """Build the TSAM medoid-representation cluster configuration."""
    return tsam.ClusterConfig(method=method, representation="medoid")


def run_aggregation_all_groups(
    group_ids: list[str],
    data_with_metadata: pd.DataFrame,
    feature_columns: list[str],
    config: GroupedWorkflowConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one 24-hour TSAM aggregation per group and combine the outputs."""
    reduced_hourly_chunks: list[pd.DataFrame] = []
    day_assignment_chunks: list[pd.DataFrame] = []
    tsam_results_by_group: dict[str, Any] = {}
    cluster = build_cluster_config(config.cluster_method)

    for group_id in group_ids:
        group_features, group_data_with_metadata = slice_group_data(
            group_id,
            data_with_metadata,
            feature_columns,
        )
        n_clusters = get_n_clusters_for_group(
            group_data_with_metadata,
            config.clusters_by_day_type,
        )
        aggregation_result = tsam.aggregate(
            data=group_features,
            n_clusters=n_clusters,
            period_duration=DAILY_PERIOD_DURATION_HOURS,
            preserve_column_means=config.preserve_column_means,
            cluster=cluster,
            numerical_tolerance=NUMERICAL_TOLERANCE,
        )
        tsam_results_by_group[group_id] = aggregation_result

        group_reduced_hourly_data, group_day_assignments = (
            collect_representative_day_data(
                aggregation_result,
                group_data_with_metadata,
            )
        )
        reduced_hourly_chunks.append(group_reduced_hourly_data)
        day_assignment_chunks.append(group_day_assignments)

    return (
        pd.concat(reduced_hourly_chunks).sort_index(),
        pd.concat(day_assignment_chunks).sort_index(),
        tsam_results_by_group,
    )


def build_representative_days(reduced_hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Build one provenance row per exported representative profile."""
    return (
        reduced_hourly_df.drop_duplicates(subset="representative_id", keep="first")
        .rename(columns={"date": "selected_medoid_date"})
        .loc[:, REPRESENTATIVE_DAY_COLUMNS]
        .sort_values("selected_medoid_date")
        .reset_index(drop=True)
    )


def build_group_accuracy(
    representative_days: pd.DataFrame,
    tsam_results_by_group: Mapping[str, Any],
) -> pd.DataFrame:
    """Build one weighted accuracy row per TSAM group."""
    group_accuracy_metrics = pd.DataFrame.from_dict(
        {
            group_id: {
                "weighted_rmse": result.accuracy.weighted_rmse,
                "weighted_rmse_duration": result.accuracy.weighted_rmse_duration,
            }
            for group_id, result in tsam_results_by_group.items()
        },
        orient="index",
    ).rename_axis("group_id")
    group_metadata = representative_days.groupby("group_id").agg(
        month=("month", "first"),
        day_type=("day_type", "first"),
        n_days=("cluster_weight", "sum"),
        n_clusters=("cluster_id", "nunique"),
    )
    return group_accuracy_metrics.join(group_metadata).sort_values(
        "weighted_rmse",
        ascending=False,
    )


def build_feature_columns_by_country_and_group(
    columns: list[str],
    feature_group_by_feature: Mapping[str, str],
) -> dict[str, dict[str, list[str]]]:
    """Build the chart lookup from canonical ``COUNTRY_feature_YEAR`` columns."""
    lookup: dict[str, dict[str, list[str]]] = {}
    for col in columns:
        parts = col.split("_")
        if len(parts) != 3 or not parts[2].isdigit():
            raise ValueError(f"Malformed canonical feature column: {col}")
        country, feature, _ = parts
        if feature not in feature_group_by_feature:
            raise ValueError(f"Unknown feature in column: {col}")
        group = feature_group_by_feature[feature]
        lookup.setdefault(country, {}).setdefault(group, []).append(col)
    return lookup


def _load_feature_data(
    config: GroupedWorkflowConfig,
    dataset_specs: Mapping[str, DatasetSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], pd.Timedelta]:
    """Load, validate, join, and country-filter configured feature datasets.

    Returns the TSAM-ready feature table, dataset coverage diagnostics, and the
    normalized selected country codes plus the shared sampling frequency.
    Validation happens before joining so malformed source files fail with
    dataset-specific error messages.
    """
    loaded = load_datasets(dataset_specs, config.year, config.snapshot_column)
    for name, df in loaded.items():
        validate_loaded_columns(name, df, config.snapshot_column)

    indexed: dict[str, pd.DataFrame] = {}
    sampling_frequency: pd.Timedelta | None = None
    for name, df in loaded.items():
        indexed_df = set_snapshot_index(
            df,
            config.snapshot_column,
            config.timestamp_format,
        )
        dataset_frequency = validate_df(indexed_df, name, config.year)
        if sampling_frequency is None:
            sampling_frequency = dataset_frequency
        elif dataset_frequency != sampling_frequency:
            raise ValueError(
                f"{name}: sampling frequency {dataset_frequency} differs from "
                f"{sampling_frequency}"
            )
        if dataset_specs[name].unit_interval:
            validate_unit_interval(indexed_df, name)
        indexed[name] = indexed_df

    dataset_coverage = build_dataset_coverage(indexed)
    feature_data = join_datasets(indexed)
    feature_data, selected_countries = filter_countries(feature_data, config.countries)
    if sampling_frequency is None:
        raise ValueError("At least one dataset must be configured")
    return feature_data, dataset_coverage, selected_countries, sampling_frequency


def run_grouped_workflow(
    config: GroupedWorkflowConfig,
    dataset_specs: Mapping[str, DatasetSpec],
) -> GroupedWorkflowResult:
    """Run the grouped workflow and return reusable notebook/CLI outputs."""
    if config.working_clusters < 1 or config.non_working_clusters < 1:
        raise ValueError("Cluster counts must be positive integers")

    feature_data, dataset_coverage, selected_countries, sampling_frequency = (
        _load_feature_data(
        config,
        dataset_specs,
        )
    )
    period_timesteps = daily_period_timesteps(sampling_frequency)
    feature_columns = feature_data.columns.tolist()
    hourly_data_with_metadata = build_hourly_metadata(
        feature_data,
        config.non_working_weekdays,
    )
    daily_metadata = build_daily_metadata(hourly_data_with_metadata)
    validate_group_feasibility(daily_metadata, config.clusters_by_day_type)
    group_ids = sorted(
        hourly_data_with_metadata["group_id"].unique(),
        key=sort_group_id,
    )
    reduced_hourly_df, day_assignments_df, tsam_results_by_group = (
        run_aggregation_all_groups(
            group_ids,
            hourly_data_with_metadata,
            feature_columns,
            config,
        )
    )
    representative_days = build_representative_days(reduced_hourly_df)
    group_accuracy = build_group_accuracy(representative_days, tsam_results_by_group)
    feature_group_by_feature = {
        spec.feature: spec.feature_group for spec in dataset_specs.values()
    }
    feature_lookup = build_feature_columns_by_country_and_group(
        feature_columns,
        feature_group_by_feature,
    )

    return GroupedWorkflowResult(
        reduced_hourly_df=reduced_hourly_df,
        representative_days=representative_days,
        day_assignments_df=day_assignments_df,
        tsam_results_by_group=tsam_results_by_group,
        dataset_coverage=dataset_coverage,
        group_accuracy=group_accuracy,
        feature_columns_by_country_and_group=feature_lookup,
        feature_group_by_feature=feature_group_by_feature,
        selected_countries=selected_countries,
        group_ids=group_ids,
        sampling_frequency=sampling_frequency,
        period_timesteps=period_timesteps,
        preserve_column_means=config.preserve_column_means,
    )
