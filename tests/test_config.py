from __future__ import annotations

from pathlib import Path

import pytest

from tsam_workflows import cli
from tsam_workflows.config import load_grouped_config_file


def test_load_grouped_config_file_resolves_complete_custom_configuration(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "portable-run"
    config_dir.mkdir()
    path = config_dir / "workflow.yaml"
    path.write_text(
        """\
input:
  snapshot_column: timestamp
  timestamp_format: "%Y-%m-%d %H:%M:%S"
datasets:
  load:
    path: inputs/load.csv
    separator: ";"
    feature: load
    feature_group: demand
    unit_interval: false
  pv:
    path: inputs/pv.csv
    separator: ","
    feature: pv
    feature_group: capacity_factors
    unit_interval: true
workflow:
  year: 2024
  countries: [de, FR]
  working_clusters: 4
  non_working_clusters: 3
  cluster_method: kmedoids
  non_working_weekdays: [Friday, Saturday]
  preserve_column_means: true
""",
        encoding="utf-8",
    )

    loaded = load_grouped_config_file(path)

    assert loaded.snapshot_column == "timestamp"
    assert loaded.timestamp_format == "%Y-%m-%d %H:%M:%S"
    assert loaded.year == 2024
    assert loaded.countries == ("DE", "FR")
    assert loaded.working_clusters == 4
    assert loaded.non_working_clusters == 3
    assert loaded.cluster_method == "kmedoids"
    assert loaded.non_working_weekdays == frozenset({"Friday", "Saturday"})
    assert loaded.preserve_column_means is True
    assert loaded.dataset_specs["load"].path == config_dir / "inputs" / "load.csv"
    assert loaded.dataset_specs["pv"].unit_interval is True


@pytest.mark.parametrize("countries", ["ALL", "all"])
def test_load_grouped_config_file_accepts_explicit_all_countries(
    tmp_path: Path,
    countries: str,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        f"""\
datasets:
  demand:
    path: demand.csv
    separator: ";"
    feature: demand
    feature_group: demand
workflow:
  countries: {countries}
""",
        encoding="utf-8",
    )

    loaded = load_grouped_config_file(path)

    assert loaded.countries is None


def test_load_grouped_config_file_defaults_to_all_countries(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """\
datasets:
  demand:
    path: demand.csv
    separator: ";"
    feature: demand
    feature_group: demand
""",
        encoding="utf-8",
    )

    loaded = load_grouped_config_file(path)

    assert loaded.countries is None
    assert loaded.year == 2025
    assert loaded.non_working_weekdays == frozenset({"Saturday", "Sunday"})
    assert loaded.preserve_column_means is False


def test_load_grouped_config_file_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """\
datasets:
  demand:
    path: demand.csv
    separator: ";"
    feature: demand
    feature_group: demand
workflow:
  numerical_tolerance: 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown workflow keys.*numerical_tolerance"):
        load_grouped_config_file(path)


def test_load_grouped_config_file_rejects_all_mixed_with_codes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """\
datasets:
  demand:
    path: demand.csv
    separator: ";"
    feature: demand
    feature_group: demand
workflow:
  countries: [ALL, DE]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'all' cannot be combined"):
        load_grouped_config_file(path)


def test_cli_config_uses_yaml_then_explicit_cli_overrides(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """\
input:
  snapshot_column: time
  timestamp_format: "%Y-%m-%d %H:%M"
datasets:
  demand:
    path: demand.csv
    separator: ";"
    feature: demand
    feature_group: demand
workflow:
  year: 2024
  countries: [FR]
  working_clusters: 3
  preserve_column_means: true
""",
        encoding="utf-8",
    )
    args = cli.build_parser().parse_args(
        [
            "grouped",
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path / "out"),
            "--year",
            "2025",
            "--countries",
            "ALL",
            "--working-clusters",
            "6",
        ]
    )

    config, specs = cli.resolve_grouped_inputs(args)

    assert config.year == 2025
    assert config.countries is None
    assert config.working_clusters == 6
    assert config.non_working_clusters == 2
    assert config.snapshot_column == "time"
    assert config.timestamp_format == "%Y-%m-%d %H:%M"
    assert config.preserve_column_means is True
    assert specs["demand"].path == tmp_path / "demand.csv"


def test_cli_without_config_preserves_builtin_defaults(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        ["grouped", "--output-dir", str(tmp_path / "out")]
    )

    config, specs = cli.resolve_grouped_inputs(args)

    assert config.year == 2025
    assert config.countries is None
    assert config.working_clusters == 5
    assert config.non_working_clusters == 2
    assert specs["demand"].path == Path("data/Demand_ENTSO_E.csv")


def test_cli_rejects_data_dir_with_yaml_config(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """\
datasets:
  demand:
    path: demand.csv
    separator: ";"
    feature: demand
    feature_group: demand
""",
        encoding="utf-8",
    )
    args = cli.build_parser().parse_args(
        [
            "grouped",
            "--config",
            str(path),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(ValueError, match="--data-dir cannot be used with --config"):
        cli.resolve_grouped_inputs(args)
