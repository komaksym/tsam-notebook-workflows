from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest

from tsam_workflows import cli
from tsam_workflows.charts import ChartExportError, export_charts, plan_drilldown_jobs
from tsam_workflows.config import ChartSelection


class FakePlotter:
    def cluster_weights(self, title: str = "") -> go.Figure:
        return go.Figure(data=[go.Bar(x=["c0"], y=[1])])

    def accuracy(self, title: str = "") -> go.Figure:
        return go.Figure(data=[go.Bar(x=["rmse"], y=[0.1])])

    def cluster_representatives(self, columns: list[str], title: str = "") -> go.Figure:
        return go.Figure(data=[go.Scatter(x=[0, 1], y=[1, 2], name=columns[0])])

    def cluster_members(
        self,
        columns: list[str],
        slider: str = "cluster",
        title: str = "",
    ) -> go.Figure:
        return go.Figure(data=[go.Scatter(x=[0, 1], y=[2, 1], name=columns[0])])

    def compare(self, columns: list[str], title: str = "") -> go.Figure:
        return go.Figure(data=[go.Scatter(x=[0, 1], y=[1, 1], name=columns[0])])

    def residuals(self, columns: list[str], title: str = "") -> go.Figure:
        return go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 0], name=columns[0])])


def fake_result() -> SimpleNamespace:
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
        tsam_results_by_group={"2025_1_working": SimpleNamespace(plot=FakePlotter())},
        feature_columns_by_country_and_group={
            "DE": {"demand": ["DE_demand_2025"], "capacity_factors": ["DE_solar_2025"]},
            "FR": {"hydro": ["FR_hydro_2025"]},
        },
        dataset_coverage=pd.DataFrame(
            [{"dataset": "demand", "country_count": 1, "countries": "DE", "missing_from_union": "-"}]
        ),
    )


def test_export_charts_writes_offline_summary_files(tmp_path: Path) -> None:
    result = export_charts(fake_result(), tmp_path, ChartSelection())

    names = {path.name for path in result.files}
    assert {"index.html", "assignment_calendar.html", "representative_weights.html", "group_accuracy.html"} <= names
    assert (tmp_path / "plotly.min.js").is_file()


def test_export_charts_with_group_selector_writes_group_charts(tmp_path: Path) -> None:
    result = export_charts(
        fake_result(),
        tmp_path,
        ChartSelection(groups=("2025_1_working",)),
    )

    names = {path.name for path in result.files}
    assert "group_2025_1_working_cluster_weights.html" in names
    assert "group_2025_1_working_cluster_accuracy.html" in names


def test_plan_drilldown_jobs_records_skipped_combinations() -> None:
    result = fake_result()
    selection = ChartSelection(
        groups=("2025_1_working",),
        countries=("DE",),
        feature_groups=("hydro",),
    )

    with pytest.raises(ChartExportError, match="No requested drilldown"):
        plan_drilldown_jobs(result, selection)


def test_cli_rejects_non_empty_output_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("x", encoding="utf-8")

    status = cli.main(["grouped", "--data-dir", str(tmp_path), "--output-dir", str(output_dir)])

    assert status == 2


def test_cli_publishes_staged_artifacts_with_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        lambda workflow_result, charts_dir, selection: SimpleNamespace(
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
    assert not list(tmp_path.glob(".out.*"))
