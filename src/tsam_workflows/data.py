"""CSV loading, column normalization, validation, and country filtering."""

from __future__ import annotations

import re
from os import PathLike
from collections.abc import Mapping, Sequence

import pandas as pd

from tsam_workflows.config import DEFAULT_TIMESTAMP_FORMAT, DatasetSpec

CANONICAL_COLUMN_RE = re.compile(
    r"(?P<country>[A-Z]{2})_(?P<feature>[A-Za-z0-9]+)_(?P<year>\d{4})"
)


def find_duplicate_headers(path: str | PathLike[str], sep: str) -> list[str]:
    """Return CSV header names that appear more than once."""
    from pathlib import Path

    header = Path(path).read_text(encoding="utf-8").splitlines()[0].split(sep)
    return sorted({col for col in header if header.count(col) > 1})


def normalize_feature_columns(
    name: str,
    df: pd.DataFrame,
    spec: DatasetSpec,
    year: int,
    snapshot_column: str,
) -> pd.DataFrame:
    """Normalize source headers to ``country_feature_year``."""
    if snapshot_column not in df.columns:
        raise ValueError(f"{name}: missing {snapshot_column!r} column")

    feature_columns = [col for col in df.columns if col != snapshot_column]
    if not feature_columns:
        raise ValueError(f"{name}: no feature columns")

    rename: dict[str, str] = {}
    for col in feature_columns:
        if re.fullmatch(r"[A-Z]{2}", col):
            country = col
        else:
            match = CANONICAL_COLUMN_RE.fullmatch(col)
            if match is None:
                raise ValueError(f"{name}: malformed feature column {col!r}")
            if match["feature"] != spec.feature or int(match["year"]) != year:
                raise ValueError(
                    f"{name}: expected feature {spec.feature!r} and year {year} "
                    f"in column {col!r}"
                )
            country = match["country"]

        rename[col] = f"{country}_{spec.feature}_{year}"

    normalized = df.rename(columns=rename).copy()
    duplicate_cols = normalized.columns[
        normalized.columns.duplicated()
    ].tolist()
    if duplicate_cols:
        raise ValueError(
            f"{name}: duplicate normalized columns: {duplicate_cols}"
        )

    mangled_cols = [
        col for col in normalized.columns if re.search(r"\.\d+$", col)
    ]
    if mangled_cols:
        raise ValueError(f"{name}: pandas-mangled columns: {mangled_cols}")

    return normalized


def load_dataset(
    name: str,
    spec: DatasetSpec,
    year: int,
    snapshot_column: str,
) -> pd.DataFrame:
    """Load and normalize one configured dataset."""
    if not spec.path.is_file():
        raise FileNotFoundError(f"{name}: source file not found: {spec.path}")
    if spec.path.stat().st_size == 0:
        raise ValueError(f"{name}: dataset is empty")

    duplicate_headers = find_duplicate_headers(spec.path, spec.separator)
    if duplicate_headers:
        raise ValueError(
            f"{name}: duplicate raw CSV headers: {duplicate_headers}"
        )

    df = pd.read_csv(spec.path, sep=spec.separator)
    if df.empty:
        raise ValueError(f"{name}: dataset is empty")

    return normalize_feature_columns(name, df, spec, year, snapshot_column)


def load_datasets(
    specs: Mapping[str, DatasetSpec],
    year: int,
    snapshot_column: str,
) -> dict[str, pd.DataFrame]:
    """Load all configured datasets."""
    if not specs:
        raise ValueError("At least one dataset must be configured")
    return {
        name: load_dataset(name, spec, year, snapshot_column)
        for name, spec in specs.items()
    }


def expected_hourly_index(year: int, frequency: str) -> pd.DatetimeIndex:
    """Return the complete timestamp index for one calendar year."""
    return pd.date_range(
        start=pd.Timestamp(year=year, month=1, day=1),
        end=pd.Timestamp(year=year + 1, month=1, day=1),
        freq=frequency,
        inclusive="left",
    )


def set_snapshot_index(
    df: pd.DataFrame,
    snapshot_column: str = "snapshot",
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
) -> pd.DataFrame:
    """Return a copy indexed by timestamps parsed with ``timestamp_format``."""
    if snapshot_column not in df.columns:
        raise KeyError(f"{snapshot_column!r} is not a column in the DataFrame")
    result = df.copy()
    result[snapshot_column] = pd.to_datetime(
        result[snapshot_column],
        format=timestamp_format,
        errors="raise",
    )
    return result.set_index(snapshot_column, drop=True)


def infer_sampling_frequency(
    index: pd.DatetimeIndex,
    name: str,
) -> pd.Timedelta:
    """Infer the one regular positive timestep used by a dataset index."""
    if len(index) < 2:
        raise ValueError(f"{name}: at least two timestamps are required")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{name}: timestamps must be unique and increasing")

    intervals = index[1:] - index[:-1]
    unique_intervals = intervals.unique()
    if len(unique_intervals) != 1:
        raise ValueError(f"{name}: timestamps do not have a regular sampling frequency")
    frequency = pd.Timedelta(unique_intervals[0])
    if frequency <= pd.Timedelta(0):
        raise ValueError(f"{name}: sampling frequency must be positive")
    return frequency


def daily_period_timesteps(frequency: pd.Timedelta) -> int:
    """Return the number of regular samples in one 24-hour TSAM period."""
    day = pd.Timedelta(days=1)
    if frequency <= pd.Timedelta(0) or day % frequency != pd.Timedelta(0):
        raise ValueError(
            f"Sampling frequency {frequency} does not divide evenly into 24 hours"
        )
    return int(day / frequency)


def validate_loaded_columns(
    name: str,
    df: pd.DataFrame,
    snapshot_column: str,
) -> None:
    """Validate columns before snapshot indexing can hide naming issues."""
    if snapshot_column not in df.columns:
        raise ValueError(f"{name}: missing {snapshot_column!r} column")

    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"{name}: duplicate loaded columns: {duplicate_cols}")

    mangled_cols = [col for col in df.columns if re.search(r"\.\d+$", col)]
    if mangled_cols:
        raise ValueError(f"{name}: pandas-mangled columns: {mangled_cols}")


def validate_df(
    df: pd.DataFrame,
    name: str,
    year_to_check: int,
) -> pd.Timedelta:
    """Validate complete-year TSAM input and return its regular timestep."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: index must be a DatetimeIndex")
    if df.index.tz is not None:
        raise ValueError(f"{name}: index must be timezone-naive")
    frequency = infer_sampling_frequency(df.index, name)
    expected_index = expected_hourly_index(year_to_check, frequency)
    if len(df) != len(expected_index):
        raise ValueError(
            f"{name}: expected {len(expected_index)} rows, got {len(df)}"
        )
    if df.index.has_duplicates:
        raise ValueError(f"{name}: index has duplicate timestamps")
    if not df.index.equals(expected_index):
        missing = expected_index.difference(df.index)
        extra = df.index.difference(expected_index)
        raise ValueError(
            f"{name}: index does not exactly match complete {year_to_check} "
            f"at frequency {frequency}. "
            f"Missing examples: {missing[:5].tolist()}. "
            f"Extra examples: {extra[:5].tolist()}."
        )
    if df.isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{name}: contains NaNs in columns {cols}")
    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_cols:
        raise TypeError(
            f"{name}: non-numeric columns found: {non_numeric_cols}"
        )
    return frequency


def validate_unit_interval(df: pd.DataFrame, name: str) -> None:
    """Validate that capacity-factor columns stay inside the physical 0..1 range."""
    is_in_range = df.ge(0.0) & df.le(1.0)
    if not is_in_range.all().all():
        invalid_cols = df.columns[~is_in_range.all()].tolist()
        raise ValueError(
            f"{name}: values outside the 0.0-1.0 range in {invalid_cols}"
        )


def join_datasets(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join validated datasets with identical indexes."""
    if not datasets:
        raise ValueError("At least one dataset is required")

    reference_name, reference = next(iter(datasets.items()))
    for name, df in datasets.items():
        if not df.index.equals(reference.index):
            raise ValueError(f"{name}: index differs from {reference_name}")

    joined = pd.concat(datasets.values(), axis=1)
    duplicate_cols = joined.columns[joined.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"Joined data has duplicate columns: {duplicate_cols}")
    return joined


def countries_from_columns(columns: Sequence[str]) -> set[str]:
    """Extract country prefixes from canonical feature columns."""
    countries: set[str] = set()
    for col in columns:
        match = CANONICAL_COLUMN_RE.fullmatch(col)
        if match is None:
            raise ValueError(f"Malformed canonical feature column: {col}")
        countries.add(match["country"])
    return countries


def normalize_country_args(
    values: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Normalize CLI country arguments, returning ``None`` for all countries."""
    if not values:
        return None

    lowered = [value.lower() for value in values]
    if "all" in lowered:
        if len(values) > 1:
            raise ValueError("'all' cannot be combined with country codes")
        return None

    normalized: list[str] = []
    for value in values:
        country = value.upper()
        if not re.fullmatch(r"[A-Z]{2}", country):
            raise ValueError(f"Invalid country code: {value}")
        if country not in normalized:
            normalized.append(country)
    return tuple(normalized)


def filter_countries(
    df: pd.DataFrame,
    requested: Sequence[str] | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Filter canonical feature columns by requested country codes."""
    selected = normalize_country_args(requested)
    available = countries_from_columns(df.columns)
    if selected is None:
        return df.copy(), tuple(sorted(available))

    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"Unknown countries: {', '.join(unknown)}")

    columns = [
        col for col in df.columns if col.split("_", maxsplit=1)[0] in selected
    ]
    if not columns:
        raise ValueError("Country selection removed every feature column")
    return df.loc[:, columns].copy(), selected


def build_dataset_coverage(
    datasets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build country coverage diagnostics for each normalized dataset."""
    country_sets = {
        name: countries_from_columns(df.columns)
        for name, df in datasets.items()
    }
    all_countries = (
        set().union(*country_sets.values()) if country_sets else set()
    )
    return pd.DataFrame(
        [
            {
                "dataset": name,
                "country_count": len(countries),
                "countries": ", ".join(sorted(countries)),
                "missing_from_union": ", ".join(
                    sorted(all_countries - countries)
                )
                or "-",
            }
            for name, countries in country_sets.items()
        ]
    )
