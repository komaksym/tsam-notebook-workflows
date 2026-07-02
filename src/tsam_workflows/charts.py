"""Plotly chart builders and offline HTML export for grouped TSAM results."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, unlabel_rgb
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

from tsam_workflows.drilldown_dashboard import write_drilldown_dashboard

PLOTLY_TEMPLATE = "ggplot2"
PLOTLY_SUMMARY_TEMPLATE = "plotly"
MONTH_ORDER: list[str] = [
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
MONTH_NAME_BY_NUMBER = dict(zip(range(1, 13), MONTH_ORDER, strict=True))
CALENDAR_ROWS = 4
CALENDAR_COLS = 3
WEEKDAY_NUMBERS = list(range(7))
WEEKEND_WEEKDAY_NUMBERS = {5, 6}
WEEKDAY_LABELS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
REPRESENTATIVE_PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Dark24
)
REPRESENTATIVE_DAY_TYPE_SORT_ORDER = {"working": 0, "non-working": 1}


class ChartExportError(ValueError):
    """Raised when requested chart selectors cannot produce valid charts."""


@dataclass(frozen=True)
class DrilldownJob:
    """One concrete group/country/feature combination to export."""

    group: str
    country: str
    feature_group: str
    columns: list[str]


@dataclass(frozen=True)
class ChartExportResult:
    """Files written by chart export plus intentionally skipped combinations."""

    files: list[Path]
    skipped: list[dict[str, str]]


def _slug(value: str) -> str:
    """Return a filesystem-safe label fragment for deterministic HTML filenames."""
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _available_groups(result: Any) -> list[str]:
    """Return group IDs in workflow order, falling back to result mapping order."""
    group_ids = getattr(result, "group_ids", None)
    if group_ids is not None:
        return list(group_ids)
    return list(result.tsam_results_by_group)


def style_tsam_figure(fig: go.Figure, title: str) -> go.Figure:
    """Apply shared notebook/export sizing to TSAM-provided figures."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={"text": title, "x": 0.5, "xanchor": "center", "font_size": 20},
        autosize=True,
        width=None,
        height=720,
        margin={"l": 40, "r": 30, "t": 120, "b": 70},
        legend={"yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
    )
    fig.update_annotations(font_size=13)
    return fig


def _prepare_calendar_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Add numeric color IDs and calendar coordinates to day assignments."""
    calendar = assignments.loc[:, ["representative_id"]].copy()
    calendar.index = pd.DatetimeIndex(calendar.index).normalize()
    representative_ids = calendar["representative_id"].drop_duplicates()
    color_by_representative = {
        representative_id: color_id
        for color_id, representative_id in enumerate(representative_ids)
    }
    calendar["representative_color_id"] = calendar["representative_id"].map(
        color_by_representative
    )
    calendar["day"] = calendar.index.day
    calendar["weekday_num"] = calendar.index.weekday
    calendar["week_row"] = 0
    for month in range(1, 13):
        month_mask = calendar.index.month == month
        first_weekday = calendar.loc[month_mask].index.min().weekday()
        calendar.loc[month_mask, "week_row"] = (
            calendar.loc[month_mask, "day"] + first_weekday - 1
        ) // 7
    return calendar


def _build_representative_colorscale(
    color_ids: list[int],
) -> tuple[list[tuple[float, str]], float, float]:
    """Build a discrete Plotly colorscale for representative IDs."""
    zmin = min(color_ids) - 0.5
    zmax = max(color_ids) + 0.5
    colorscale: list[tuple[float, str]] = []
    for index, color_id in enumerate(color_ids):
        color = REPRESENTATIVE_PALETTE[index % len(REPRESENTATIVE_PALETTE)]
        colorscale.append(((color_id - 0.5 - zmin) / (zmax - zmin), color))
        colorscale.append(((color_id + 0.5 - zmin) / (zmax - zmin), color))
    return colorscale, zmin, zmax


def _contrast_text_color(background: str) -> str:
    """Return the higher-contrast text color for a Plotly palette color."""
    rgb = hex_to_rgb(background) if background.startswith("#") else unlabel_rgb(background)
    channels = [channel / 255 for channel in rgb]
    linear_channels = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    luminance = sum(
        weight * channel
        for weight, channel in zip(
            (0.2126, 0.7152, 0.0722),
            linear_channels,
            strict=True,
        )
    )
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "#000000" if black_contrast >= white_contrast else "#ffffff"


def _build_calendar_month_trace(
    month_data: pd.DataFrame,
    colorscale: list[tuple[float, str]],
    zmin: float,
    zmax: float,
) -> go.Heatmap:
    """Build one calendar-month heatmap colored by representative ID."""
    color_grid = month_data.pivot(
        index="week_row",
        columns="weekday_num",
        values="representative_color_id",
    ).reindex(columns=WEEKDAY_NUMBERS)
    day_grid = month_data.pivot(
        index="week_row",
        columns="weekday_num",
        values="day",
    ).reindex(columns=WEEKDAY_NUMBERS)
    day_text = day_grid.map(lambda value: "" if pd.isna(value) else str(int(value)))
    representative_grid = month_data.pivot(
        index="week_row",
        columns="weekday_num",
        values="representative_id",
    ).reindex(columns=WEEKDAY_NUMBERS)
    hover_text = representative_grid.copy()
    for weekday_num, weekday_label in enumerate(WEEKDAY_LABELS):
        hover_text[weekday_num] = representative_grid[weekday_num].map(
            lambda representative_id: (
                ""
                if pd.isna(representative_id)
                else f"{weekday_label}<br>Representative: {representative_id}"
            )
        )

    return go.Heatmap(
        z=color_grid.to_numpy(),
        x=WEEKDAY_NUMBERS,
        y=color_grid.index,
        text=day_text.to_numpy(),
        customdata=hover_text.to_numpy(),
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        hovertemplate="Day %{text}<br>%{customdata}<extra></extra>",
        xgap=2,
        ygap=2,
        showscale=False,
    )


def _build_calendar_day_labels(month_data: pd.DataFrame) -> go.Scatter:
    """Overlay calendar day numbers with text colors matched to each cell."""
    colors = [
        _contrast_text_color(
            REPRESENTATIVE_PALETTE[
                int(color_id) % len(REPRESENTATIVE_PALETTE)
            ]
        )
        for color_id in month_data["representative_color_id"]
    ]
    return go.Scatter(
        x=month_data["weekday_num"],
        y=month_data["week_row"],
        mode="text",
        text=month_data["day"].astype(str),
        textfont={"size": 13, "color": colors},
        hoverinfo="skip",
        showlegend=False,
    )


def _add_weekend_borders(
    fig: go.Figure,
    monthly_assignments: list[pd.DataFrame],
    subplot_positions: list[tuple[int, int]],
) -> None:
    """Outline weekend cells without replacing representative colors."""
    for month_data, (row, col) in zip(
        monthly_assignments,
        subplot_positions,
        strict=True,
    ):
        weekend_days = month_data[
            month_data["weekday_num"].isin(WEEKEND_WEEKDAY_NUMBERS)
        ]
        for _, day in weekend_days.iterrows():
            fig.add_shape(
                type="rect",
                x0=day["weekday_num"] - 0.5,
                x1=day["weekday_num"] + 0.5,
                y0=day["week_row"] - 0.5,
                y1=day["week_row"] + 0.5,
                line={"color": "rgba(35, 35, 35, 0.75)", "width": 2},
                fillcolor="rgba(0, 0, 0, 0)",
                layer="above",
                row=row,
                col=col,
            )


def build_assignment_calendar_figure(result: Any) -> go.Figure:
    """Build a 12-month calendar colored by assigned representative day."""
    calendar = _prepare_calendar_assignments(result.day_assignments_df)
    color_ids = sorted(calendar["representative_color_id"].unique().astype(int))
    colorscale, zmin, zmax = _build_representative_colorscale(color_ids)
    monthly_assignments = [
        calendar[calendar.index.month == month] for month in range(1, 13)
    ]
    heatmaps = [
        _build_calendar_month_trace(month, colorscale, zmin, zmax)
        for month in monthly_assignments
    ]
    day_labels = [
        _build_calendar_day_labels(month) for month in monthly_assignments
    ]
    subplot_positions = [
        (row, col)
        for row in range(1, CALENDAR_ROWS + 1)
        for col in range(1, CALENDAR_COLS + 1)
    ]
    fig = make_subplots(
        rows=CALENDAR_ROWS,
        cols=CALENDAR_COLS,
        subplot_titles=MONTH_ORDER,
    )
    for heatmap, labels, (row, col) in zip(
        heatmaps,
        day_labels,
        subplot_positions,
        strict=True,
    ):
        fig.add_trace(heatmap, row=row, col=col)
        fig.add_trace(labels, row=row, col=col)
    _add_weekend_borders(fig, monthly_assignments, subplot_positions)
    year = int(calendar.index.year.min())
    fig.update_layout(
        title={
            "text": f"{year} representative-day assignment calendar",
            "x": 0.5,
            "font_size": 20,
            "y": 0.99,
        },
        template=PLOTLY_TEMPLATE,
        autosize=True,
        height=900,
        plot_bgcolor="white",
        margin={"l": 20, "r": 20, "t": 120, "b": 20},
    )
    fig.update_annotations(yshift=25)
    fig.update_xaxes(
        side="top",
        tickmode="array",
        tickvals=WEEKDAY_NUMBERS,
        ticktext=WEEKDAY_LABELS,
        showticklabels=True,
        ticks="",
        showline=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        ticks="",
        showline=False,
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    return fig


def _sort_representative_group(representative_group: str) -> tuple[int, int]:
    """Sort representative rows by day type and numeric cluster ID."""
    day_type, cluster_label = representative_group.rsplit("_c", maxsplit=1)
    return (
        REPRESENTATIVE_DAY_TYPE_SORT_ORDER.get(day_type, 99),
        int(cluster_label),
    )


def build_representative_weights_figure(result: Any) -> go.Figure:
    """Build a heatmap of representative-day shares within month/day-type groups."""
    summary = result.representative_days.copy()
    summary["month_name"] = summary["month"].map(MONTH_NAME_BY_NUMBER)
    summary["representative_group"] = (
        summary["day_type"] + "_c" + summary["cluster_id"].astype(str)
    )
    summary["group_days"] = summary.groupby(["month", "day_type"])[
        "cluster_weight"
    ].transform("sum")
    summary["group_share_pct"] = (
        summary["cluster_weight"].div(summary["group_days"]).mul(100)
    )

    # The percentage denominator is the original days in the same month/day-type
    # group, not the whole month. That keeps working/non-working groups separate.
    representative_order = sorted(
        summary["representative_group"].unique(),
        key=_sort_representative_group,
    )
    matrix = summary.pivot(
        index="representative_group",
        columns="month_name",
        values="group_share_pct",
    ).reindex(index=representative_order, columns=MONTH_ORDER)
    assigned_days = summary.pivot(
        index="representative_group",
        columns="month_name",
        values="cluster_weight",
    ).reindex(index=representative_order, columns=MONTH_ORDER)
    group_days = summary.pivot(
        index="representative_group",
        columns="month_name",
        values="group_days",
    ).reindex(index=representative_order, columns=MONTH_ORDER)
    text = matrix.map(lambda value: "" if pd.isna(value) else f"{value:.1f}%")
    hover_data = [
        [
            [assigned_count, group_count]
            for assigned_count, group_count in zip(
                assigned_row,
                group_row,
                strict=True,
            )
        ]
        for assigned_row, group_row in zip(
            assigned_days.to_numpy(),
            group_days.to_numpy(),
            strict=True,
        )
    ]
    fig = px.imshow(
        matrix,
        aspect="auto",
        labels={
            "x": "Month",
            "y": "Representative group",
            "color": "% of month/day-type group",
        },
        template=PLOTLY_SUMMARY_TEMPLATE,
    )
    fig.update_traces(
        customdata=hover_data,
        text=text.to_numpy(),
        texttemplate="%{text}" if matrix.size <= 240 else "",
        hovertemplate=(
            "Month: %{x}<br>"
            "Representative group: %{y}<br>"
            "Share of month/day-type group: %{z:.1f}%<br>"
            "Days assigned: %{customdata[0]:.0f}<br>"
            "Days in month/day-type group: %{customdata[1]:.0f}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        title={
            "text": "Share of each month/day-type group assigned to each representative",
            "x": 0.5,
        },
        autosize=True,
        height=max(500, 120 + 42 * max(1, len(matrix))),
        margin={"l": 20, "r": 20, "t": 80, "b": 40},
        coloraxis_showscale=False,
    )
    fig.update_xaxes(title_font_size=18, tickfont={"size": 14})
    fig.update_yaxes(
        title_font_size=18,
        tickfont={"size": 14 if len(matrix) <= 10 else 11},
    )
    return fig


def build_group_accuracy_figure(result: Any) -> go.Figure:
    """Build the weighted-RMSE overview across month/day-type groups."""
    group_accuracy_plot = result.group_accuracy.reset_index().sort_values(
        "weighted_rmse",
        ascending=True,
    )
    fig = px.bar(
        group_accuracy_plot,
        x="weighted_rmse",
        y="group_id",
        color="day_type",
        orientation="h",
        custom_data=[
            "month",
            "day_type",
            "n_days",
            "n_clusters",
            "weighted_rmse_duration",
        ],
        labels={
            "weighted_rmse": "Weighted RMSE",
            "group_id": "Group",
            "day_type": "Day type",
        },
        template=PLOTLY_SUMMARY_TEMPLATE,
    )
    fig.update_traces(
        hovertemplate=(
            "Group: %{y}<br>"
            "Month: %{customdata[0]}<br>"
            "Day type: %{customdata[1]}<br>"
            "Original days: %{customdata[2]:.0f}<br>"
            "Clusters: %{customdata[3]:.0f}<br>"
            "Weighted RMSE: %{x:.4f}<br>"
            "Weighted duration RMSE: %{customdata[4]:.4f}"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        title={"text": "Weighted RMSE by group", "x": 0.5},
        autosize=True,
        height=700,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def _write_figure(fig: go.Figure, path: Path) -> Path:
    """Write one interactive HTML figure and return its path."""
    # "directory" writes one shared plotly.min.js next to the HTML files, so the
    # exported chart folder stays offline-capable without duplicating Plotly.
    fig.write_html(path, include_plotlyjs="directory", config={"responsive": True})
    return path


def plan_group_jobs(result: Any) -> list[str]:
    """Return group IDs that need group-only TSAM charts."""
    return _available_groups(result)


def plan_drilldown_jobs(
    result: Any,
) -> tuple[list[DrilldownJob], list[dict[str, str]]]:
    """Return every valid group/country/feature chart combination."""
    jobs: list[DrilldownJob] = []
    feature_lookup = result.feature_columns_by_country_and_group
    for group in _available_groups(result):
        for country in sorted(feature_lookup):
            for feature_group, columns in sorted(feature_lookup[country].items()):
                jobs.append(DrilldownJob(group, country, feature_group, columns))
    return jobs, []


def _write_group_diagnostics(result: Any, output_dir: Path) -> Path:
    """Write one offline dashboard for all group-level TSAM diagnostics."""
    figures: dict[str, dict[str, Any]] = {}
    groups = plan_group_jobs(result)
    for group in groups:
        plotter = result.tsam_results_by_group[group].plot
        weights = style_tsam_figure(
            plotter.cluster_weights(title=""),
            f"Cluster weights: {group}",
        )
        accuracy = style_tsam_figure(
            plotter.accuracy(title=""),
            f"Cluster accuracy: {group}",
        )
        figures[group] = {
            "cluster_weights": weights.to_plotly_json(),
            "cluster_accuracy": accuracy.to_plotly_json(),
        }

    options = "\n".join(
        f'<option value="{html.escape(group, quote=True)}">'
        f"{html.escape(group)}</option>"
        for group in groups
    )
    payload = json.dumps(
        figures,
        cls=PlotlyJSONEncoder,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    path = output_dir / "group_diagnostics.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>TSAM group diagnostics</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}"
        "label{font-weight:600;margin-right:8px;}select{font:inherit;padding:6px;}"
        ".chart{min-height:720px;}</style></head><body>"
        "<h1>TSAM group diagnostics</h1>"
        '<label for="group-select">Group</label>'
        f'<select id="group-select">{options}</select>'
        '<div id="cluster-weights" class="chart"></div>'
        '<div id="cluster-accuracy" class="chart"></div>'
        '<script src="plotly.min.js"></script><script>'
        f"const figures={payload};"
        "const groupSelect=document.getElementById('group-select');"
        "const config={responsive:true};"
        "function renderGroup(group){"
        "const selected=figures[group];"
        "Plotly.react('cluster-weights',selected.cluster_weights.data,"
        "selected.cluster_weights.layout,config);"
        "Plotly.react('cluster-accuracy',selected.cluster_accuracy.data,"
        "selected.cluster_accuracy.layout,config);"
        "}"
        "groupSelect.addEventListener('change',event=>renderGroup(event.target.value));"
        "renderGroup(groupSelect.value);"
        "</script></body></html>",
        encoding="utf-8",
    )
    return path


def _write_index(output_dir: Path, files: list[Path]) -> Path:
    """Write a minimal local navigation page for exported chart HTML files."""
    links = []
    for path in sorted(files, key=lambda item: item.name):
        if path.name == "index.html":
            continue
        label = html.escape(path.stem.replace("_", " ").title())
        links.append(f'<li><a href="{html.escape(path.name)}">{label}</a></li>')
    index = output_dir / "index.html"
    index.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>TSAM workflow charts</title></head><body>"
        "<h1>TSAM workflow charts</h1><ul>"
        + "\n".join(links)
        + "</ul></body></html>",
        encoding="utf-8",
    )
    return index


def export_charts(
    result: Any,
    output_dir: Path,
) -> ChartExportResult:
    """Export every summary and valid drilldown chart as offline Plotly HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write_figure(
            build_assignment_calendar_figure(result),
            assets_dir / "assignment_calendar.html",
        ),
        _write_figure(
            build_representative_weights_figure(result),
            assets_dir / "representative_weights.html",
        ),
        _write_figure(
            build_group_accuracy_figure(result),
            assets_dir / "group_accuracy.html",
        ),
    ]

    files.append(_write_group_diagnostics(result, assets_dir))

    files.append(write_drilldown_dashboard(result, assets_dir))
    skipped: list[dict[str, str]] = []

    index = _write_index(output_dir, files)
    files.append(index)
    plotly_bundle = assets_dir / "plotly.min.js"
    if plotly_bundle.exists():
        files.append(plotly_bundle)
    return ChartExportResult(files=files, skipped=skipped)
