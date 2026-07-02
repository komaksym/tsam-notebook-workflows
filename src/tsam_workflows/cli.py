"""Command-line interface for reusable TSAM workflows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from tsam_workflows.charts import ChartExportError, export_charts
from tsam_workflows.config import (
    GroupedWorkflowConfig,
    SUPPORTED_CLUSTER_METHODS,
    default_dataset_specs,
)
from tsam_workflows.data import normalize_country_args
from tsam_workflows.grouped import GroupedWorkflowResult, run_grouped_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser and grouped-workflow subcommand parser."""
    parser = argparse.ArgumentParser(prog="tsam-workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)
    grouped = subparsers.add_parser("grouped", help="Run grouped TSAM workflow")
    grouped.add_argument("--data-dir", type=Path, default=Path("data"))
    grouped.add_argument("--output-dir", type=Path, required=True)
    grouped.add_argument("--year", type=int, default=2025)
    grouped.add_argument("--countries", nargs="+", default=("all",))
    grouped.add_argument("--working-clusters", type=int, default=5)
    grouped.add_argument("--non-working-clusters", type=int, default=2)
    grouped.add_argument(
        "--cluster-method",
        choices=SUPPORTED_CLUSTER_METHODS,
        default="hierarchical",
    )
    grouped.add_argument("--overwrite", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> GroupedWorkflowConfig:
    """Convert parsed grouped-command arguments into workflow configuration."""
    countries = normalize_country_args(args.countries)
    return GroupedWorkflowConfig(
        year=args.year,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        countries=countries,
        working_clusters=args.working_clusters,
        non_working_clusters=args.non_working_clusters,
        cluster_method=args.cluster_method,
        overwrite=args.overwrite,
    )


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
    """Atomically move a completed staging directory into the final location."""
    if output_dir.exists():
        if output_dir.is_dir():
            if any(output_dir.iterdir()) and not overwrite:
                raise ValueError(f"Output directory is not empty: {output_dir}")
            shutil.rmtree(output_dir)
        elif overwrite:
            output_dir.unlink()
        else:
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    staging_dir.rename(output_dir)


def run_grouped_command(args: argparse.Namespace) -> int:
    """Run the grouped workflow CLI command and publish staged artifacts."""
    config = config_from_args(args)
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
        specs = default_dataset_specs(config.data_dir)
        result = run_grouped_workflow(config, specs)
        artifacts = _save_tables(result, staging_dir)
        charts_dir = staging_dir / "charts"
        chart_result = export_charts(result, charts_dir)
        artifacts.extend(chart_result.files)
        manifest = _write_manifest(
            config,
            result,
            staging_dir,
            artifacts,
            chart_result.skipped,
        )
        artifacts.append(manifest)
        _publish_staging(staging_dir, config.output_dir, config.overwrite)
        published = True
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
