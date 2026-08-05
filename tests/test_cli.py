"""Tests for explicit CLI workflows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import app
from src import cli
from src.core.taxonomy import build_assignments_from_leaf_series


SIMPLE_TREE = {
    "name": "Root",
    "children": [
        {
            "name": "Alimentacion",
            "children": [{"name": "Supermercado"}],
        },
        {
            "name": "Otros",
            "children": [{"name": "Revision manual"}],
        },
    ],
}


def test_execute_cli_summary_exports_grouped_output(
    tmp_path: Path,
    capsys,
) -> None:
    export_path = tmp_path / "summary.csv"
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01"]),
            "Concepto": ["Mercadona"],
            "Movimiento": ["Compra"],
            "Importe": [-10.0],
            "Grupo": ["Supermercado"],
        }
    )
    args = cli.parse_args(
        [
            "--summary",
            "--group-by",
            "CategoriaLeaf",
            "--export-summary",
            str(export_path),
        ]
    )

    handled = cli.execute_cli(
        args,
        dataframe=dataframe,
        category_tree=SIMPLE_TREE,
    )

    captured = capsys.readouterr()
    assert handled is True
    assert export_path.exists()
    assert "Supermercado" in captured.out


def test_execute_cli_categorize_tree_persists_assignments(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    categories_file = tmp_path / "movimientos_categorias.csv"
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01"]),
            "Concepto": ["Mercadona"],
            "Movimiento": ["Compra"],
            "Importe": [-10.0],
            "Grupo": [pd.NA],
        }
    )
    args = cli.parse_args(["--categorize-tree"])

    def fake_categorize_transactions_by_tree(
        df: pd.DataFrame,
        *,
        tree: dict[str, object],
        root_path: str | None,
        cache_path: Path,
        status_callback,
    ) -> pd.DataFrame:
        assert root_path is None
        assert len(df) == 1
        status_callback("fake status")
        return build_assignments_from_leaf_series(
            pd.Series(["Supermercado"], index=df.index, dtype="object"),
            tree,
            source="ia_arbol:Alimentacion > Supermercado",
            confidence=0.91,
            reason="Test",
        )

    monkeypatch.setattr(
        cli,
        "categorize_transactions_by_tree",
        fake_categorize_transactions_by_tree,
    )

    handled = cli.execute_cli(
        args,
        dataframe=dataframe,
        category_tree=SIMPLE_TREE,
        categories_file=categories_file,
        rules_file=tmp_path / "missing_rules.json",
    )

    captured = capsys.readouterr()
    assert handled is True
    assert categories_file.exists()
    assert "categorizo 1 movimientos" in captured.out
    assert "Supermercado" in categories_file.read_text(encoding="utf-8")


def test_execute_cli_ai_audit_prints_metrics_and_review_queue(capsys) -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Concepto": ["Mercadona", "Desconocido"],
            "Movimiento": ["Compra", "Cargo"],
            "Importe": [-10.0, -40.0],
            "Grupo": ["Supermercado", "Revision manual"],
            "CategoriaLeaf": ["Supermercado", "Revision manual"],
            "CategoriaPath": [
                "Alimentacion > Supermercado",
                "Otros > Revision manual",
            ],
            "CategoriaFuente": ["ia_arbol:Root", "ia_arbol:fallback"],
            "CategoriaConfianza": [0.9, 0.0],
        }
    )
    args = cli.parse_args(["--ai-audit", "--review-only"])

    handled = cli.execute_cli(
        args,
        dataframe=dataframe,
        category_tree=SIMPLE_TREE,
    )

    captured = capsys.readouterr()
    assert handled is True
    assert "Auditoria IA" in captured.out
    assert "revision" in captured.out
    assert "Desconocido" in captured.out
    assert "Mercadona" not in captured.out


def test_main_list_tree_does_not_bootstrap_dataframe(monkeypatch, capsys) -> None:
    called: list[str] = []

    monkeypatch.setattr(app, "initialize_runtime", lambda: called.append("init"))
    monkeypatch.setattr(app, "load_category_tree", lambda: SIMPLE_TREE)
    monkeypatch.setattr(
        app,
        "execute_cli",
        lambda args, dataframe, category_tree: print("CLI_TREE"),
    )
    monkeypatch.setattr(
        app,
        "bootstrap_dataframe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    app.main(["--list-tree"])

    captured = capsys.readouterr()
    assert called == ["init"]
    assert "CLI_TREE" in captured.out
