"""Configuration types and default dataset specs for TSAM workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ClusterMethod = Literal[
    "averaging",
    "kmeans",
    "kmedoids",
    "kmaxoids",
    "hierarchical",
    "contiguous",
]

SUPPORTED_CLUSTER_METHODS: tuple[ClusterMethod, ...] = (
    "averaging",
    "kmeans",
    "kmedoids",
    "kmaxoids",
    "hierarchical",
    "contiguous",
)

DEFAULT_NON_WORKING_WEEKDAYS: frozenset[str] = frozenset({"Saturday", "Sunday"})


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one single-feature input CSV."""

    path: Path
    separator: str
    feature: str
    feature_group: str
    unit_interval: bool = False


@dataclass(frozen=True)
class ChartSelection:
    """Selectors controlling optional offline chart expansion."""

    groups: tuple[str, ...] | None = None
    countries: tuple[str, ...] | None = None
    feature_groups: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GroupedWorkflowConfig:
    """Runtime options for the grouped TSAM workflow."""

    year: int
    data_dir: Path
    output_dir: Path
    countries: tuple[str, ...] | None = None
    working_clusters: int = 5
    non_working_clusters: int = 2
    cluster_method: ClusterMethod = "hierarchical"
    chart_selection: ChartSelection = field(default_factory=ChartSelection)
    overwrite: bool = False
    hourly_frequency: str = "h"
    snapshot_column: str = "snapshot"
    period_duration_hours: int = 24
    numerical_tolerance: float = 1e-9
    preserve_column_means: bool = False
    non_working_weekdays: frozenset[str] = DEFAULT_NON_WORKING_WEEKDAYS

    @property
    def clusters_by_day_type(self) -> dict[str, int]:
        """Return TSAM cluster counts keyed by workflow day type."""
        return {
            "working": self.working_clusters,
            "non-working": self.non_working_clusters,
        }


def default_dataset_specs(data_dir: Path) -> dict[str, DatasetSpec]:
    """Return the checked-in grouped-workflow dataset configuration."""
    return {
        "demand": DatasetSpec(
            path=data_dir / "Demand_ENTSO_E.csv",
            separator=";",
            feature="demand",
            feature_group="demand",
        ),
        "solar_cf": DatasetSpec(
            path=data_dir / "solar_capacity_factors.csv",
            separator=";",
            feature="solar",
            feature_group="capacity_factors",
            unit_interval=True,
        ),
        "onwind_cf": DatasetSpec(
            path=data_dir / "onwind_capacity_factors.csv",
            separator=",",
            feature="onwind",
            feature_group="capacity_factors",
            unit_interval=True,
        ),
        "ror_cf": DatasetSpec(
            path=data_dir / "ror_capacity_factors.csv",
            separator=",",
            feature="ror",
            feature_group="capacity_factors",
            unit_interval=True,
        ),
        "hydro_inflow": DatasetSpec(
            path=data_dir / "hydro_inflow_scaled_deduped_2025.csv",
            separator=",",
            feature="hydro",
            feature_group="hydro",
        ),
    }
