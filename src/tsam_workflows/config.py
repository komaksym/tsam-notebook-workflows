"""Configuration types and default dataset specs for TSAM workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal, cast

import yaml

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
WEEKDAYS: frozenset[str] = frozenset(
    {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }
)
DEFAULT_TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M"

COUNTRY_NAMES: dict[str, str] = {
    "AL": "Albania",
    "AT": "Austria",
    "BA": "Bosnia and Herzegovina",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CH": "Switzerland",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GR": "Greece",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MD": "Moldova",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "NL": "Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RS": "Serbia",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "UA": "Ukraine",
    "UK": "United Kingdom",
    "XK": "Kosovo",
}


def country_options(codes: list[str]) -> list[tuple[str, str]]:
    """Return the original notebook's sorted ``Full name (CODE)`` options."""
    missing_country_names = set(codes).difference(COUNTRY_NAMES)
    if missing_country_names:
        raise ValueError(
            f"Missing country names for: {sorted(missing_country_names)}"
        )
    return [(f"{COUNTRY_NAMES[code]} ({code})", code) for code in sorted(codes)]


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one single-feature input CSV."""

    path: Path
    separator: str
    feature: str
    feature_group: str
    unit_interval: bool = False


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
    overwrite: bool = False
    snapshot_column: str = "snapshot"
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT
    preserve_column_means: bool = False
    non_working_weekdays: frozenset[str] = DEFAULT_NON_WORKING_WEEKDAYS

    @property
    def clusters_by_day_type(self) -> dict[str, int]:
        """Return TSAM cluster counts keyed by workflow day type."""
        return {
            "working": self.working_clusters,
            "non-working": self.non_working_clusters,
        }


@dataclass(frozen=True)
class GroupedConfigFile:
    """Validated portable settings loaded from one grouped-workflow YAML file."""

    dataset_specs: dict[str, DatasetSpec]
    snapshot_column: str = "snapshot"
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT
    year: int = 2025
    countries: tuple[str, ...] | None = None
    working_clusters: int = 5
    non_working_clusters: int = 2
    cluster_method: ClusterMethod = "hierarchical"
    non_working_weekdays: frozenset[str] = DEFAULT_NON_WORKING_WEEKDAYS
    preserve_column_means: bool = False


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return cast(Mapping[str, Any], value)


def _reject_unknown_keys(
    values: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown}")


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _parse_yaml_countries(value: object) -> tuple[str, ...] | None:
    from tsam_workflows.data import normalize_country_args

    if isinstance(value, str):
        if value.lower() != "all":
            raise ValueError("workflow.countries must be ALL or a list of codes")
        return None
    if not isinstance(value, list) or not value or not all(
        isinstance(country, str) for country in value
    ):
        raise ValueError("workflow.countries must be ALL or a list of codes")
    return normalize_country_args(value)


def _parse_dataset_specs(
    values: object,
    config_dir: Path,
) -> dict[str, DatasetSpec]:
    datasets = _require_mapping(values, "datasets")
    if not datasets:
        raise ValueError("At least one dataset must be configured")

    specs: dict[str, DatasetSpec] = {}
    allowed = {"path", "separator", "feature", "feature_group", "unit_interval"}
    for name, raw_spec in datasets.items():
        spec = _require_mapping(raw_spec, f"datasets.{name}")
        _reject_unknown_keys(spec, allowed, f"datasets.{name}")
        missing = sorted({"path", "separator", "feature", "feature_group"} - set(spec))
        if missing:
            raise ValueError(f"datasets.{name} missing required keys: {missing}")

        relative_path = Path(_require_string(spec["path"], f"datasets.{name}.path"))
        path = (
            relative_path
            if relative_path.is_absolute()
            else (config_dir / relative_path).resolve()
        )
        separator = _require_string(spec["separator"], f"datasets.{name}.separator")
        feature = _require_string(spec["feature"], f"datasets.{name}.feature")
        if re.fullmatch(r"[A-Za-z0-9]+", feature) is None:
            raise ValueError(f"datasets.{name}.feature must contain only letters and digits")
        feature_group = _require_string(
            spec["feature_group"], f"datasets.{name}.feature_group"
        )
        unit_interval = spec.get("unit_interval", False)
        if not isinstance(unit_interval, bool):
            raise ValueError(f"datasets.{name}.unit_interval must be a boolean")
        specs[name] = DatasetSpec(
            path=path,
            separator=separator,
            feature=feature,
            feature_group=feature_group,
            unit_interval=unit_interval,
        )
    return specs


def load_grouped_config_file(path: Path) -> GroupedConfigFile:
    """Load and strictly validate one optional grouped-workflow YAML file."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    root = _require_mapping(raw, "config")
    _reject_unknown_keys(root, {"input", "datasets", "workflow"}, "top-level")
    if "datasets" not in root:
        raise ValueError("Config is missing required 'datasets' mapping")

    input_values = _require_mapping(root.get("input", {}), "input")
    _reject_unknown_keys(
        input_values,
        {"snapshot_column", "timestamp_format"},
        "input",
    )
    snapshot_column = _require_string(
        input_values.get("snapshot_column", "snapshot"),
        "input.snapshot_column",
    )
    timestamp_format = _require_string(
        input_values.get("timestamp_format", DEFAULT_TIMESTAMP_FORMAT),
        "input.timestamp_format",
    )

    workflow = _require_mapping(root.get("workflow", {}), "workflow")
    allowed_workflow = {
        "year",
        "countries",
        "working_clusters",
        "non_working_clusters",
        "cluster_method",
        "non_working_weekdays",
        "preserve_column_means",
    }
    _reject_unknown_keys(workflow, allowed_workflow, "workflow")

    year = _require_positive_int(workflow.get("year", 2025), "workflow.year")
    countries = _parse_yaml_countries(workflow.get("countries", "ALL"))
    working_clusters = _require_positive_int(
        workflow.get("working_clusters", 5), "workflow.working_clusters"
    )
    non_working_clusters = _require_positive_int(
        workflow.get("non_working_clusters", 2),
        "workflow.non_working_clusters",
    )
    method_value = workflow.get("cluster_method", "hierarchical")
    if method_value not in SUPPORTED_CLUSTER_METHODS:
        raise ValueError(
            "workflow.cluster_method must be one of "
            f"{list(SUPPORTED_CLUSTER_METHODS)}"
        )
    cluster_method = cast(ClusterMethod, method_value)

    weekdays_value = workflow.get(
        "non_working_weekdays", sorted(DEFAULT_NON_WORKING_WEEKDAYS)
    )
    if not isinstance(weekdays_value, list) or not all(
        isinstance(day, str) for day in weekdays_value
    ):
        raise ValueError("workflow.non_working_weekdays must be a list")
    unknown_weekdays = sorted(set(weekdays_value).difference(WEEKDAYS))
    if unknown_weekdays:
        raise ValueError(f"Unknown non-working weekdays: {unknown_weekdays}")
    non_working_weekdays = frozenset(weekdays_value)

    preserve_column_means = workflow.get("preserve_column_means", False)
    if not isinstance(preserve_column_means, bool):
        raise ValueError("workflow.preserve_column_means must be a boolean")

    return GroupedConfigFile(
        dataset_specs=_parse_dataset_specs(root["datasets"], path.parent),
        snapshot_column=snapshot_column,
        timestamp_format=timestamp_format,
        year=year,
        countries=countries,
        working_clusters=working_clusters,
        non_working_clusters=non_working_clusters,
        cluster_method=cluster_method,
        non_working_weekdays=non_working_weekdays,
        preserve_column_means=preserve_column_means,
    )


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
