"""Tests for the public entrypoint and headless helpers."""
from __future__ import annotations

from pathlib import Path

import main as root_main
import pandas as pd

from src import app


def test_root_main_reexports_app_main() -> None:
    assert root_main.main is app.main


def test_bootstrap_dataframe_applies_saved_categories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "expenses.csv"
    categories_path = tmp_path / "movimientos_categorias.csv"
    tree_path = tmp_path / "category_tree.json"
    csv_path.write_text(
        "Fecha,Concepto,Movimiento,Importe\n"
        "01/01/2026,Mercadona,Compra,-10.5\n",
        encoding="utf-8",
    )
    categories_path.write_text(
        "Fecha,Concepto,Movimiento,Importe,Grupo\n"
        "2026-01-01,Mercadona,Compra,-10.5,Supermercado\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "initialize_runtime", lambda: None)

    dataframe = app.bootstrap_dataframe(
        csv_path,
        categories_file=categories_path,
        tree_file=tree_path,
    )

    assert dataframe.loc[0, "Grupo"] == "Supermercado"
    assert dataframe.loc[0, "CategoriaPath"] == "Alimentacion > Supermercado"


def test_headless_summary_groups_rows_by_concept() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Concepto": ["Mercadona", "Mercadona"],
            "Movimiento": ["Compra", "Compra"],
            "Importe": [-10.0, -5.0],
        }
    )

    summary = app.headless_summary(dataframe)

    assert len(summary) == 1
    assert summary.iloc[0].to_dict() == {
        "Concepto": "Mercadona",
        "count": 2,
        "importe_total": -15.0,
    }


def test_initialize_runtime_calls_setup_steps(monkeypatch) -> None:
    called: list[str] = []

    monkeypatch.setattr(app, "ensure_directories", lambda: called.append("dirs"))
    monkeypatch.setattr(app, "load_dotenv", lambda: called.append("env"))
    monkeypatch.setattr(app, "configure_logging", lambda: called.append("log"))
    monkeypatch.setattr(app, "migrate_legacy_files", lambda: called.append("legacy"))

    app.initialize_runtime()

    assert called == ["dirs", "env", "log", "legacy"]


def test_main_runs_headless_summary_when_gui_is_unavailable(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01"]),
            "Concepto": ["Mercadona"],
            "Movimiento": ["Compra"],
            "Importe": [-10.0],
        }
    )
    captured: list[pd.DataFrame] = []

    monkeypatch.setattr(app, "bootstrap_dataframe", lambda: dataframe)
    monkeypatch.setattr(app, "is_gui_available", lambda: False)
    monkeypatch.setattr(app, "print_headless_summary", captured.append)

    app.main()

    assert len(captured) == 1
    assert captured[0] is dataframe


def test_main_runs_gui_when_available(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01"]),
            "Concepto": ["Mercadona"],
            "Movimiento": ["Compra"],
            "Importe": [-10.0],
        }
    )
    captured: list[pd.DataFrame] = []

    monkeypatch.setattr(app, "bootstrap_dataframe", lambda: dataframe)
    monkeypatch.setattr(app, "is_gui_available", lambda: True)
    monkeypatch.setattr(app, "run_gui", captured.append)

    app.main()

    assert len(captured) == 1
    assert captured[0] is dataframe
