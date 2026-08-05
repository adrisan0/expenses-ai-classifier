"""Tests for disk persistence helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.infra.storage import (
    load_categories_from_disk,
    migrate_legacy_files,
    save_categories_to_disk,
)


def test_save_and_load_categories_round_trip(tmp_path: Path) -> None:
    categories_file = tmp_path / "movimientos_categorias.csv"
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Concepto": ["Mercadona", "Bizum"],
            "Movimiento": ["Compra", "Transferencia"],
            "Importe": [-12.5, -4.5],
            "Grupo": ["Supermercado", ""],
            "CategoriaLeaf": ["Supermercado", pd.NA],
            "CategoriaPath": ["Alimentacion > Supermercado", pd.NA],
            "CategoriaNivel1": ["Alimentacion", pd.NA],
            "CategoriaNivel2": ["Supermercado", pd.NA],
            "CategoriaNivel3": [pd.NA, pd.NA],
            "CategoriaFuente": ["regla_local", pd.NA],
            "CategoriaConfianza": [0.95, pd.NA],
            "CategoriaMotivoIA": ["Ruta IA: Alimentacion > Supermercado", pd.NA],
            "CategoriaTrazaIA": ['[{"nodo":"Root"}]', pd.NA],
        }
    )

    save_categories_to_disk(dataframe, categories_file=categories_file)
    loaded = load_categories_from_disk(
        dataframe.drop(
            columns=[
                "Grupo",
                "CategoriaLeaf",
                "CategoriaPath",
                "CategoriaNivel1",
                "CategoriaNivel2",
                "CategoriaNivel3",
                "CategoriaFuente",
                "CategoriaConfianza",
                "CategoriaMotivoIA",
                "CategoriaTrazaIA",
            ]
        ),
        categories_file=categories_file,
    )

    assert categories_file.exists()
    assert loaded.loc[0, "Grupo"] == "Supermercado"
    assert loaded.loc[0, "CategoriaPath"] == "Alimentacion > Supermercado"
    assert loaded.loc[0, "CategoriaFuente"] == "regla_local"
    assert loaded.loc[0, "CategoriaTrazaIA"] == '[{"nodo":"Root"}]'
    assert pd.isna(loaded.loc[1, "Grupo"])


def test_migrate_legacy_files_moves_existing_file(tmp_path: Path) -> None:
    old_path = tmp_path / "old.csv"
    new_path = tmp_path / "nested" / "new.csv"
    old_path.write_text("demo", encoding="utf-8")

    migrate_legacy_files({old_path: new_path})

    assert not old_path.exists()
    assert new_path.read_text(encoding="utf-8") == "demo"
