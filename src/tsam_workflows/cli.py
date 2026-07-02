"""Command-line interface for reusable TSAM workflows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from tsam_workflows.charts import ChartExportError, export_charts
from tsam_workflows.config import (
    GroupedConfigFile,
    GroupedWorkflowConfig,
    DatasetSpec,
    SUPPORTED_CLUSTER_METHODS,
    default_dataset_specs,
    load_grouped_config_file,
)
from tsam_workflows.data import normalize_country_args
from tsam_workflows.grouped import GroupedWorkflowResult, run_grouped_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser and grouped-workflow subcommand parser."""
    parser = argparse.ArgumentParser(prog="tsam-workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)
    grouped = subparsers.add_parser("grouped", help="Run grouped TSAM workflow")
    grouped.add_argument(
        "--config",
        type=Path,
        help="Optional YAML workflow configuration",
    )
    grouped.add_argument(
        "--data-dir",
        type=Path,
        help="Built-in dataset directory (cannot be combined with --config)",
    )
    grouped.add_argument("--output-dir", type=Path, required=True)
    grouped.add_argument("--year", type=int, help="Override configured year")
    grouped.add_argument(
        "--countries",
        nargs="+",
        help="Override configured countries with ALL or country codes",
    )
    grouped.add_argument(
        "--working-clusters",
        type=int,
        help="Override configured working-day cluster count",
    )
    grouped.add_argument(
        "--non-working-clusters",
        type=int,
        help="Override configured non-working-day cluster count",
    )
    grouped.add_argument(
        "--cluster-method",
        choices=SUPPORTED_CLUSTER_METHODS,
    )
    grouped.add_argument("--overwrite", action="store_true")
    return parser


def resolve_grouped_inputs(
    args: argparse.Namespace,
) -> tuple[GroupedWorkflowConfig, dict[str, DatasetSpec]]:
    """Merge defaults, optional YAML, and explicit CLI grouped settings."""
    if args.config is not None:
        if args.data_dir is not None:
            raise ValueError("--data-dir cannot be used with --config")
        config_path = args.config.resolve()
        loaded = load_grouped_config_file(config_path)
        data_dir = config_path.parent
    else:
        data_dir = args.data_dir if args.data_dir is not None else Path("data")
        loaded = GroupedConfigFile(dataset_specs=default_dataset_specs(data_dir))

    countries = (
        normalize_country_args(args.countries)
        if args.countries is not None
        else loaded.countries
    )
    config = GroupedWorkflowConfig(
        year=args.year if args.year is not None else loaded.year,
        data_dir=data_dir,
        output_dir=args.output_dir,
        countries=countries,
        working_clusters=(
            args.working_clusters
            if args.working_clusters is not None
            else loaded.working_clusters
        ),
        non_working_clusters=(
            args.non_working_clusters
            if args.non_working_clusters is not None
            else loaded.non_working_clusters
        ),
        cluster_method=(
            args.cluster_method
            if args.cluster_method is not None
            else loaded.cluster_method
        ),
        overwrite=args.overwrite,
        snapshot_column=loaded.snapshot_column,
        timestamp_format=loaded.timestamp_format,
        preserve_column_means=loaded.preserve_column_means,
        non_working_weekdays=loaded.non_working_weekdays,
    )
    return config, loaded.dataset_specs


def config_from_args(args: argparse.Namespace) -> GroupedWorkflowConfig:
    """Convert parsed grouped-command inputs into workflow configuration."""
    config, _ = resolve_grouped_inputs(args)
    return config


def _is_non_empty_dir(path: Path) -> bool:
    """Return whether ``path`` is an existing directory with any entries."""
    return path.is_dir() and any(path.iterdir())


def _ensure_output_available(path: Path, overwrite: bool) -> None:
    """Fail before running if the target output path cannot be published to."""
    if path.exists() and not path.is_dir():
        if not overwrite:
            raise ValueError(f"Output path exists and is not a directory: {path}")
        return
    if _is_non_empty_dir(path) and not overwrite:
        raise ValueError(f"Output directory is not empty: {path}")


def _save_tables(result: GroupedWorkflowResult, output_dir: Path) -> list[Path]:
    """Write the three stable CSV artifacts and return their paths."""
    tables = {
        "reduced_hourly_df.csv": result.reduced_hourly_df.reset_index(),
        "representative_days.csv": result.representative_days,
        "day_assignments_df.csv": result.day_assignments_df.reset_index(),
    }
    files: list[Path] = []
    for filename, table in tables.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        files.append(path)
    return files


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a DataFrame into JSON-compatible row dictionaries."""
    return json.loads(df.to_json(orient="records"))


def _write_manifest(
    config: GroupedWorkflowConfig,
    result: Any,
    specs: Mapping[str, DatasetSpec],
    output_dir: Path,
    artifacts: list[Path],
    skipped: list[dict[str, str]],
) -> Path:
    """Write the run manifest with config, coverage, artifacts, and skips."""
    artifact_names = {
        str(path.relative_to(output_dir)) for path in artifacts if path.exists()
    }
    artifact_names.add("manifest.json")
    manifest = {
        "schema_version": 1,
        "config": {
            "year": config.year,
            "countries": list(config.countries) if config.countries else "all",
            "working_clusters": config.working_clusters,
            "non_working_clusters": config.non_working_clusters,
            "cluster_method": config.cluster_method,
            "preserve_column_means": config.preserve_column_means,
            "sampling_frequency": str(result.sampling_frequency),
            "period_timesteps": result.period_timesteps,
            "snapshot_column": config.snapshot_column,
            "timestamp_format": config.timestamp_format,
            "non_working_weekdays": sorted(config.non_working_weekdays),
        },
        "datasets": {
            name: {
                "path": str(spec.path),
                "separator": spec.separator,
                "feature": spec.feature,
                "feature_group": spec.feature_group,
                "unit_interval": spec.unit_interval,
            }
            for name, spec in sorted(specs.items())
        },
        "dataset_coverage": _json_records(result.dataset_coverage),
        "selected_countries": list(getattr(result, "selected_countries", ())),
        "artifacts": sorted(artifact_names),
        "skipped_chart_combinations": skipped,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _publish_staging(staging_dir: Path, output_dir: Path, overwrite: bool) -> None:
    """Publish staged artifacts while preserving existing directory identities."""
    if output_dir.exists():
        if output_dir.is_dir():
            if any(output_dir.iterdir()) and not overwrite:
                raise ValueError(f"Output directory is not empty: {output_dir}")
            _merge_staging_directory(staging_dir, output_dir)
            return
        elif overwrite:
            output_dir.unlink()
        else:
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    staging_dir.rename(output_dir)


def _remove_artifact(path: Path) -> None:
    """Remove one obsolete artifact regardless of whether it is a file or directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _merge_staging_directory(staging_dir: Path, output_dir: Path) -> None:
    """Move staged files into stable directories and publish the manifest last."""
    staged_names = {path.name for path in staging_dir.iterdir()}
    for output_path in output_dir.iterdir():
        if output_path.name not in staged_names and output_path.name != ".DS_Store":
            _remove_artifact(output_path)

    staged_paths = sorted(
        staging_dir.iterdir(),
        key=lambda path: (path.name == "manifest.json", path.name),
    )
    for staging_path in staged_paths:
        output_path = output_dir / staging_path.name
        if staging_path.is_dir() and not staging_path.is_symlink():
            if output_path.exists() and not (
                output_path.is_dir() and not output_path.is_symlink()
            ):
                _remove_artifact(output_path)
            output_path.mkdir(exist_ok=True)
            _merge_staging_directory(staging_path, output_path)
        else:
            if output_path.exists() and output_path.is_dir():
                _remove_artifact(output_path)
            staging_path.replace(output_path)
    staging_dir.rmdir()


def run_grouped_command(args: argparse.Namespace) -> int:
    """Run the grouped workflow CLI command and publish staged artifacts."""
    config, specs = resolve_grouped_inputs(args)
    _ensure_output_available(config.output_dir, config.overwrite)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.",
            dir=config.output_dir.parent,
        )
    )
    published = False
    try:
        print("[1/4] Loading data and clustering...", file=sys.stderr, flush=True)
        result = run_grouped_workflow(config, specs)
        print("[2/4] Writing CSV artifacts...", file=sys.stderr, flush=True)
        artifacts = _save_tables(result, staging_dir)
        print("[3/4] Generating interactive charts...", file=sys.stderr, flush=True)
        charts_dir = staging_dir / "charts"
        chart_result = export_charts(result, charts_dir)
        artifacts.extend(chart_result.files)
        print("[4/4] Publishing output...", file=sys.stderr, flush=True)
        manifest = _write_manifest(
            config,
            result,
            specs,
            staging_dir,
            artifacts,
            chart_result.skipped,
        )
        artifacts.append(manifest)
        _publish_staging(staging_dir, config.output_dir, config.overwrite)
        published = True
        print(file=sys.stderr, flush=True)
        output_dir = config.output_dir.resolve()
        dashboard = (output_dir / "charts" / "index.html").as_uri()
        print("Grouped workflow completed successfully.")
        print(f"Open the chart dashboard: {dashboard}")
        print(f"Artifacts written to: {output_dir}")
        return 0
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, dispatch the selected command, and return exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "grouped":
            return run_grouped_command(args)
        parser.error(f"Unknown command: {args.command}")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    except (ValueError, OSError, ChartExportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
