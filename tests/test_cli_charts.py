from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from tsam_workflows import charts, cli, config
from tsam_workflows.charts import (
    build_assignment_calendar_figure,
    build_group_accuracy_figure,
    build_representative_weights_figure,
    export_charts,
    plan_drilldown_jobs,
)


class FakePlotter:
    def cluster_weights(self, title: str = "") -> go.Figure:
        return go.Figure(data=[go.Bar(x=["c0"], y=[1])])

    def accuracy(self, title: str = "") -> go.Figure:
        return go.Figure(data=[go.Bar(x=["rmse"], y=[0.1])])

    def cluster_representatives(self, columns: list[str], title: str = "") -> go.Figure:
        raise AssertionError("feature figures must be built in the browser")

    def cluster_members(
        self,
        columns: list[str],
        slider: str = "cluster",
        title: str = "",
    ) -> go.Figure:
        raise AssertionError("feature figures must be built in the browser")

    def compare(self, columns: list[str], title: str = "") -> go.Figure:
        raise AssertionError("feature figures must be built in the browser")

    def residuals(self, columns: list[str], title: str = "") -> go.Figure:
        raise AssertionError("feature figures must be built in the browser")


def fake_result() -> SimpleNamespace:
    columns = ["DE_demand_2025", "DE_solar_2025", "FR_hydro_2025"]
    original_index = pd.date_range("2025-01-01", periods=4, freq="h", name="snapshot")
    original = pd.DataFrame(
        {
            "DE_demand_2025": [1.0, 2.0, 3.0, 4.0],
            "DE_solar_2025": [0.0, 0.2, 0.8, 0.1],
            "FR_hydro_2025": [0.4, 0.5, 0.6, 0.7],
        },
        index=original_index,
    )
    representatives = pd.DataFrame(
        original.to_numpy(),
        columns=columns,
        index=pd.MultiIndex.from_product(
            [[0, 1], [0, 1]], names=["cluster", "timestep"]
        ),
    )
    aggregation = SimpleNamespace(
        plot=FakePlotter(),
        original=original,
        cluster_representatives=representatives,
        cluster_assignments=[0, 1],
        cluster_weights={0: 1, 1: 1},
        n_timesteps_per_period=2,
    )
    representative_days = pd.DataFrame(
        {
            "selected_medoid_date": [pd.Timestamp("2025-01-01")],
            "representative_id": ["2025_1_working_c0"],
            "group_id": ["2025_1_working"],
            "month": [1],
            "day_type": ["working"],
            "cluster_id": [0],
            "cluster_weight": [1],
        }
    )
    day_assignments = pd.DataFrame(
        {
            "cluster_id": [0],
            "cluster_weight": [1],
            "group_id": ["2025_1_working"],
            "representative_id": ["2025_1_working_c0"],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2025-01-01")], name="date"),
    )
    return SimpleNamespace(
        reduced_hourly_df=pd.DataFrame(
            {"representative_id": ["2025_1_working_c0"]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-01-01 00:00")], name="snapshot"),
        ),
        representative_days=representative_days,
        day_assignments_df=day_assignments,
        group_accuracy=pd.DataFrame(
            {
                "weighted_rmse": [0.1],
                "weighted_rmse_duration": [0.2],
                "month": [1],
                "day_type": ["working"],
                "n_days": [1],
                "n_clusters": [1],
            },
            index=pd.Index(["2025_1_working"], name="group_id"),
        ),
        tsam_results_by_group={"2025_1_working": aggregation},
        feature_columns_by_country_and_group={
            "DE": {"demand": ["DE_demand_2025"], "capacity_factors": ["DE_solar_2025"]},
            "FR": {"hydro": ["FR_hydro_2025"]},
        },
        dataset_coverage=pd.DataFrame(
            [{"dataset": "demand", "country_count": 1, "countries": "DE", "missing_from_union": "-"}]
        ),
    )


def calendar_result() -> SimpleNamespace:
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D", name="date")
    day_types = [
        "non-working" if date.weekday() >= 5 else "working" for date in dates
    ]
    assignments = pd.DataFrame(
        {
            "cluster_id": 0,
            "cluster_weight": 1,
            "group_id": [
                f"2025_{date.month}_{day_type}"
                for date, day_type in zip(dates, day_types, strict=True)
            ],
            "representative_id": [
                f"2025_{date.month}_{day_type}_c0"
                for date, day_type in zip(dates, day_types, strict=True)
            ]
        },
        index=dates,
    )
    return SimpleNamespace(day_assignments_df=assignments)


def representative_weight_result() -> SimpleNamespace:
    return SimpleNamespace(
        representative_days=pd.DataFrame(
            {
                "month": [1, 1, 1],
                "day_type": ["working", "non-working", "working"],
                "cluster_id": [1, 0, 0],
                "cluster_weight": [10, 9, 12],
            }
        )
    )


def test_assignment_calendar_uses_month_calendar_heatmaps() -> None:
    figure = build_assignment_calendar_figure(calendar_result())

    heatmaps = [trace for trace in figure.data if trace.type == "heatmap"]
    day_labels = [trace for trace in figure.data if trace.type == "scatter"]

    assert len(heatmaps) == 12
    assert len(day_labels) == 12
    assert all(trace.hoverinfo == "skip" for trace in day_labels)
    assert [annotation.text for annotation in figure.layout.annotations] == [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    assert figure.layout.shapes


def test_assignment_calendar_keeps_original_representative_palette() -> None:
    figure = build_assignment_calendar_figure(calendar_result())
    heatmap = next(trace for trace in figure.data if trace.type == "heatmap")
    colors = [color for _, color in heatmap.colorscale[::2]]

    assert colors[: len(charts.px.colors.qualitative.Plotly)] == list(
        charts.px.colors.qualitative.Plotly
    )


def test_calendar_day_text_uses_contrasting_colors() -> None:
    assert charts._contrast_text_color("#222A2A") == "#ffffff"
    assert charts._contrast_text_color("#FECB52") == "#000000"


def test_approach_notebook_uses_original_chart_defaults() -> None:
    notebook_path = Path(__file__).parents[1] / "src" / "approach_1_ALL.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    config_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and any("NOTEBOOK_WORKING_CLUSTERS =" in line for line in cell["source"])
    )

    assert "NOTEBOOK_WORKING_CLUSTERS = 5" in config_source
    assert "NOTEBOOK_NON_WORKING_CLUSTERS = 2" in config_source
    assert "NOTEBOOK_COUNTRIES = None" in config_source

    all_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "RUN_CHART_OUTPUTS" not in all_code
    assert "RUN_WIDGET_OUTPUTS" not in all_code
    assert "COUNTRY_OPTIONS = country_options(" in all_code


def test_approach_notebook_documents_unsupported_tsam_customization_methods() -> None:
    notebook_path = Path(__file__).parents[1] / "src" / "approach_1_ALL.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_by_heading = {
        cell["source"][0].strip(): "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown" and cell["source"]
    }

    normalization = markdown_by_heading["# Normalization"]
    for detail in (
        "per-column min-max scaling",
        "normalize_column_means=True",
        "weights=",
        "1 / sqrt(group_column_count)",
        "Physical-scale normalization",
        "not supported by the grouped workflow or CLI",
    ):
        assert detail in normalization

    preservation = markdown_by_heading["# Feature Preservation"]
    for detail in (
        'representation="medoid"',
        "preserve_column_means=True",
        "ExtremeConfig",
        "tsam.Distribution",
        "not supported by the grouped workflow or CLI",
    ):
        assert detail in preservation


def test_approach_notebook_has_linked_table_of_contents_after_title() -> None:
    notebook_path = Path(__file__).parents[1] / "src" / "approach_1_ALL.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    toc = "".join(notebook["cells"][1]["source"])
    assert toc == (
        "## Table of Contents\n"
        "\n"
        "- [Method Overview](#Method-Overview)\n"
        "- [Normalization](#Normalization)\n"
        "- [Feature Preservation](#Feature-Preservation)\n"
        "  - [Preservation Objectives](#Preservation-Objectives)\n"
        "  - [Clustering Influence](#Clustering-Influence)\n"
        "  - [Current Baseline](#Current-Baseline)\n"
        "- [Imports And Configuration](#Imports-And-Configuration)\n"
        "- [Run Workflow](#Run-Workflow)\n"
        "- [Output Tables](#Output-Tables)\n"
        "- [Summary Charts](#Summary-Charts)\n"
        "- [Group-Level TSAM Diagnostic Drilldowns](#Group-Level-TSAM-Diagnostic-Drilldowns)\n"
        "- [Optional CSV Export](#Optional-CSV-Export)\n"
    )


def test_country_options_restore_original_notebook_labels() -> None:
    assert config.country_options(["FR", "DE"]) == [
        ("Germany (DE)", "DE"),
        ("France (FR)", "FR"),
    ]

    with pytest.raises(ValueError, match=r"Missing country names for: \['ZZ'\]"):
        config.country_options(["ZZ"])


def test_representative_weights_include_ordered_labels_and_day_counts() -> None:
    figure = build_representative_weights_figure(representative_weight_result())
    heatmap = figure.data[0]

    assert list(heatmap.y) == ["working_c0", "working_c1", "non-working_c0"]
    assert heatmap.text[0][0] == "54.5%"
    assert list(heatmap.customdata[0][0]) == [12, 22]
    assert "Days assigned" in heatmap.hovertemplate


def test_summary_charts_keep_original_plotly_palette() -> None:
    expected = pio.templates["plotly"].layout.colorscale.sequential

    weights = build_representative_weights_figure(representative_weight_result())
    accuracy = build_group_accuracy_figure(fake_result())

    assert weights.layout.coloraxis.colorscale == expected
    assert accuracy.layout.template.layout.colorway == pio.templates["plotly"].layout.colorway


def test_export_charts_writes_summary_and_group_files_by_default(
    tmp_path: Path,
) -> None:
    result = export_charts(fake_result(), tmp_path)

    names = {path.name for path in result.files}
    assert {"index.html", "assignment_calendar.html", "representative_weights.html", "group_accuracy.html"} <= names
    assert "group_diagnostics.html" in names
    assert "drilldown_dashboard.html" in names
    assert not any(name.endswith("_cluster_weights.html") for name in names)
    assert not any(name.endswith("_cluster_accuracy.html") for name in names)
    assert not any(name.startswith("group_2025_1_working_country_") for name in names)
    assert {path.name for path in tmp_path.iterdir()} == {"assets", "index.html"}
    assets_dir = tmp_path / "assets"
    assert (assets_dir / "plotly.min.js").is_file()

    dashboard = (assets_dir / "group_diagnostics.html").read_text(encoding="utf-8")
    assert '<select id="group-select">' in dashboard
    assert '<option value="2025_1_working">2025_1_working</option>' in dashboard
    assert 'id="cluster-weights"' in dashboard
    assert 'id="cluster-accuracy"' in dashboard
    assert "Plotly.react" in dashboard

    drilldowns = (assets_dir / "drilldown_dashboard.html").read_text(encoding="utf-8")
    for selector in ("group-select", "country-select", "feature-select", "chart-select"):
        assert f'id="{selector}"' in drilldowns
    for chart_type in ("representatives", "members", "comparison", "residuals"):
        assert f'value="{chart_type}"' in drilldowns
    for field in (
        '"original"',
        '"representatives"',
        '"assignments"',
        '"weights"',
    ):
        assert field in drilldowns
    assert '"reconstructed"' not in drilldowns
    assert '"DE":"Germany (DE)"' in drilldowns
    assert '"FR":"France (FR)"' in drilldowns
    assert "DE_demand_2025" in drilldowns
    assert "Plotly.react" in drilldowns

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '<nav id="chart-navigation"' in index
    assert "Overview" in index
    assert "Explore" in index
    for filename in (
        "assignment_calendar.html",
        "representative_weights.html",
        "group_accuracy.html",
        "group_diagnostics.html",
        "drilldown_dashboard.html",
    ):
        assert f'href="assets/{filename}" target="chart-frame"' in index
    assert '<iframe id="chart-frame" name="chart-frame"' in index
    assert 'src="assets/assignment_calendar.html"' in index
    assert 'aria-current="page"' in index
    assert 'id="open-chart"' in index
    assert "https://" not in index


def test_plan_drilldown_jobs_returns_every_valid_combination() -> None:
    jobs, skipped = plan_drilldown_jobs(fake_result())

    assert {
        (job.group, job.country, job.feature_group)
        for job in jobs
    } == {
        ("2025_1_working", "DE", "capacity_factors"),
        ("2025_1_working", "DE", "demand"),
        ("2025_1_working", "FR", "hydro"),
    }
    assert skipped == []


def test_cli_rejects_chart_selection_options(tmp_path: Path) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "grouped",
                "--output-dir",
                str(tmp_path / "out"),
                "--chart-groups",
                "all",
            ]
        )


def test_cli_rejects_non_empty_output_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("x", encoding="utf-8")

    status = cli.main(["grouped", "--data-dir", str(tmp_path), "--output-dir", str(output_dir)])

    assert status == 2


def test_publish_overwrite_preserves_open_directory_identity(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True)
    (output_dir / ".DS_Store").write_bytes(b"finder-layout")
    (output_dir / "manifest.json").write_text("old", encoding="utf-8")
    (charts_dir / "obsolete.html").write_text("old", encoding="utf-8")
    output_inode = output_dir.stat().st_ino
    charts_inode = charts_dir.stat().st_ino

    staging_dir = tmp_path / ".out.staging"
    staging_charts = staging_dir / "charts"
    staging_charts.mkdir(parents=True)
    (staging_dir / "manifest.json").write_text("new", encoding="utf-8")
    (staging_charts / "index.html").write_text("new chart", encoding="utf-8")

    cli._publish_staging(staging_dir, output_dir, overwrite=True)

    assert output_dir.stat().st_ino == output_inode
    assert charts_dir.stat().st_ino == charts_inode
    assert (output_dir / ".DS_Store").read_bytes() == b"finder-layout"
    assert (output_dir / "manifest.json").read_text(encoding="utf-8") == "new"
    assert (charts_dir / "index.html").read_text(encoding="utf-8") == "new chart"
    assert not (charts_dir / "obsolete.html").exists()
    assert not staging_dir.exists()


def test_cli_publishes_staged_artifacts_with_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "out"
    result = fake_result()

    monkeypatch.setattr(cli, "run_grouped_workflow", lambda config, specs: result)
    monkeypatch.setattr(
        cli,
        "default_dataset_specs",
        lambda data_dir: {
            "demand": SimpleNamespace(
                path=tmp_path / "demand.csv",
                separator=";",
                feature="demand",
                feature_group="demand",
                unit_interval=False,
            )
        },
    )
    monkeypatch.setattr(
        cli,
        "export_charts",
        lambda workflow_result, charts_dir: SimpleNamespace(
            files=[charts_dir / "index.html"],
            skipped=[],
        ),
    )

    status = cli.main(
        [
            "grouped",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--countries",
            "de",
            "--working-clusters",
            "1",
            "--non-working-clusters",
            "1",
        ]
    )

    assert status == 0
    assert (output_dir / "reduced_hourly_df.csv").is_file()
    assert (output_dir / "representative_days.csv").is_file()
    assert (output_dir / "day_assignments_df.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "chart_selection" not in manifest["config"]
    assert not list(tmp_path.glob(".out.*"))
    captured = capsys.readouterr()
    assert captured.err == (
        "[1/4] Loading data and clustering...\n"
        "[2/4] Writing CSV artifacts...\n"
        "[3/4] Generating interactive charts...\n"
        "[4/4] Publishing output...\n\n"
    )
    assert captured.out == (
        "Grouped workflow completed successfully.\n"
        f"Open the chart dashboard: {(output_dir / 'charts' / 'index.html').resolve().as_uri()}\n"
        f"Artifacts written to: {output_dir.resolve()}\n"
    )
