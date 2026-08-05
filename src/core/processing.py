"""CSV ingestion and grouping logic for expenses."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


def load_and_clean_csv(path: Path | str) -> pd.DataFrame:
    """Load the input CSV and normalize the expected columns."""
    df = _read_expenses_csv(path)

    if str(df.columns[0]).strip() == "" or str(df.columns[0]).startswith("Unnamed"):
        df = df.iloc[:, 1:]

    df.columns = [str(column).strip() for column in df.columns]
    df = df.apply(
        lambda column: column.str.strip() if column.dtype == "object" else column
    )

    for column in ("F.Valor", "Divisa", "Divisa.1"):
        if column in df.columns:
            df = df.drop(columns=[column])

    expected_columns = {"Fecha", "Concepto", "Movimiento", "Importe"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas en el CSV: {sorted(missing_columns)}"
        )

    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce")
    return df.dropna(subset=["Fecha", "Importe"]).copy()


def _read_expenses_csv(path: Path | str) -> pd.DataFrame:
    """Read CSVs tolerating spaces before quoted fields."""
    csv_path = Path(path)
    with csv_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("El CSV está vacío") from exc

        rows: list[list[str]] = []
        malformed_rows: list[tuple[int, int, int]] = []
        expected_width = len(header)

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != expected_width:
                malformed_rows.append((line_number, len(row), expected_width))
                continue
            rows.append(row)

    if malformed_rows:
        details = ", ".join(
            f"línea {line_number} ({width}/{expected_width})"
            for line_number, width, expected_width in malformed_rows[:5]
        )
        raise ValueError(
            "Se encontraron filas con un número inesperado de columnas: "
            f"{details}"
        )

    return pd.DataFrame(rows, columns=header)


def build_filter_mask(
    df: pd.DataFrame,
    *,
    group_col: str,
    pattern: str | None,
    importe_range: tuple[float | None, float | None] | None,
    date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None,
) -> pd.Series:
    """Build a reusable mask for date, amount and regex filters."""
    if group_col not in df.columns:
        raise ValueError(f"Columna no encontrada: {group_col}")

    mask = pd.Series(True, index=df.index, dtype=bool)

    if date_range is not None:
        date_min, date_max = date_range
        if date_min is not None:
            mask &= df["Fecha"] >= date_min
        if date_max is not None:
            mask &= df["Fecha"] <= date_max

    if importe_range is not None:
        value_min, value_max = importe_range
        if value_min is not None:
            mask &= df["Importe"] >= value_min
        if value_max is not None:
            mask &= df["Importe"] <= value_max

    if pattern:
        try:
            compiled = re.compile(pattern, flags=re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Regex inválido: {exc}") from exc

        mask &= df[group_col].astype(str).str.contains(compiled, na=False)

    return mask


def filter_and_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    pattern: str | None,
    importe_range: tuple[float | None, float | None] | None,
    date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None,
) -> pd.DataFrame:
    """Filter movements and return an aggregated summary."""
    mask = build_filter_mask(
        df,
        group_col=group_col,
        pattern=pattern,
        importe_range=importe_range,
        date_range=date_range,
    )
    filtered = df.loc[mask].copy()

    if filtered.empty:
        return pd.DataFrame(columns=[group_col, "count", "importe_total"])

    return (
        filtered.groupby(group_col)
        .agg(count=("Importe", "size"), importe_total=("Importe", "sum"))
        .reset_index()
        .sort_values(["importe_total", "count"], ascending=[False, False])
    )
