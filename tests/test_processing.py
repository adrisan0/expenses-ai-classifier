"""Tests for CSV loading, filtering and grouping."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.processing import build_filter_mask, filter_and_group, load_and_clean_csv


def test_load_and_clean_csv_normalizes_and_drops_invalid_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "expenses.csv"
    csv_path.write_text(
        (
            "Unnamed: 0,Fecha,Concepto,Movimiento,Importe,F.Valor,Divisa\n"
            "0,01/01/2026,Mercadona,Compra,-10.50,01/01/2026,EUR\n"
            "1,02/01/2026,Nomina,Ingreso,1200.00,02/01/2026,EUR\n"
            "2,,Sin fecha,Compra,-20.00,03/01/2026,EUR\n"
        ),
        encoding="utf-8",
    )

    dataframe = load_and_clean_csv(csv_path)

    assert list(dataframe.columns) == ["Fecha", "Concepto", "Movimiento", "Importe"]
    assert len(dataframe) == 2
    assert pd.api.types.is_datetime64_any_dtype(dataframe["Fecha"])
    assert dataframe["Importe"].tolist() == [-10.5, 1200.0]


def test_load_and_clean_csv_supports_embedded_commas_after_spaces(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "expenses.csv"
    csv_path.write_text(
        (
            ",F.Valor,Fecha,Concepto,Movimiento,Importe,Divisa,Disponible,"
            "Divisa.1,Comentario\n"
            '0,06/12/2024,09/12/2024, "Rest, artesana",Pago con tarjeta,'
            "-12.60,EUR,714.40,EUR,460332\n"
        ),
        encoding="utf-8",
    )

    dataframe = load_and_clean_csv(csv_path)

    assert len(dataframe) == 1
    assert dataframe.loc[0, "Concepto"] == "Rest, artesana"
    assert dataframe.loc[0, "Movimiento"] == "Pago con tarjeta"
    assert dataframe.loc[0, "Importe"] == -12.6


def test_load_and_clean_csv_rejects_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "expenses.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV está vacío"):
        load_and_clean_csv(csv_path)


def test_load_and_clean_csv_rejects_inconsistent_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "expenses.csv"
    csv_path.write_text(
        (
            "Fecha,Concepto,Movimiento,Importe\n"
            "01/01/2026,Mercadona,Compra,-10.50\n"
            "\n"
            "02/01/2026,Bizum,Transferencia,-5.00,EXTRA\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="número inesperado de columnas"):
        load_and_clean_csv(csv_path)


def test_build_filter_mask_applies_ranges_and_regex() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "Concepto": ["Mercadona", "Bizum", "Nomina"],
            "Movimiento": ["Compra", "Transferencia", "Ingreso"],
            "Importe": [-30.0, -15.0, 1200.0],
        }
    )

    mask = build_filter_mask(
        dataframe,
        group_col="Concepto",
        pattern="merc",
        importe_range=(-40.0, -10.0),
        date_range=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")),
    )

    assert mask.tolist() == [True, False, False]


def test_build_filter_mask_rejects_invalid_regex() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01"]),
            "Concepto": ["Mercadona"],
            "Movimiento": ["Compra"],
            "Importe": [-10.0],
        }
    )

    with pytest.raises(ValueError, match="Regex inválido"):
        build_filter_mask(
            dataframe,
            group_col="Concepto",
            pattern="[",
            importe_range=(None, None),
            date_range=(None, None),
        )


def test_filter_and_group_returns_aggregated_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
            "Concepto": ["Mercadona", "Mercadona", "Bizum", "Nomina"],
            "Movimiento": ["Compra", "Compra", "Transferencia", "Ingreso"],
            "Importe": [-10.0, -20.0, -15.0, 1200.0],
        }
    )

    grouped = filter_and_group(
        dataframe,
        group_col="Concepto",
        pattern="",
        importe_range=(None, None),
        date_range=(None, None),
    )

    assert list(grouped.columns) == ["Concepto", "count", "importe_total"]
    assert grouped.iloc[0].to_dict() == {
        "Concepto": "Nomina",
        "count": 1,
        "importe_total": 1200.0,
    }
    mercadona = grouped[grouped["Concepto"] == "Mercadona"].iloc[0]
    assert mercadona["count"] == 2
    assert mercadona["importe_total"] == -30.0
