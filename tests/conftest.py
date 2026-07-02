from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def hourly_2025() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left")


def write_feature_csv(
    path: Path,
    index: pd.DatetimeIndex,
    columns: dict[str, list[float]],
    *,
    sep: str,
) -> None:
    rows = ["snapshot" + sep + sep.join(columns)]
    values = list(columns.values())
    for row_idx, timestamp in enumerate(index):
        formatted = timestamp.strftime("%d.%m.%Y %H:%M")
        row_values = [str(col_values[row_idx]) for col_values in values]
        rows.append(formatted + sep + sep.join(row_values))
    path.write_text("\n".join(rows), encoding="utf-8")

