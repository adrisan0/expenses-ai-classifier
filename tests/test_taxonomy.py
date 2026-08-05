"""Tests for taxonomy normalization and assignment helpers."""
from __future__ import annotations

import pandas as pd

from src.core.taxonomy import (
    build_assignments_from_leaf_series,
    ensure_category_columns,
    merge_missing_leaves,
)


def test_ensure_category_columns_enriches_legacy_group_labels() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01"]),
            "Concepto": ["Mercadona"],
            "Movimiento": ["Compra"],
            "Importe": [-20.0],
            "Grupo": ["Supermercado"],
        }
    )
    tree = {
        "name": "Root",
        "children": [
            {
                "name": "Alimentacion",
                "children": [{"name": "Supermercado"}],
            }
        ],
    }

    enriched = ensure_category_columns(dataframe, tree)

    assert enriched.loc[0, "CategoriaLeaf"] == "Supermercado"
    assert enriched.loc[0, "CategoriaPath"] == "Alimentacion > Supermercado"
    assert enriched.loc[0, "CategoriaNivel1"] == "Alimentacion"
    assert enriched.loc[0, "CategoriaNivel2"] == "Supermercado"


def test_merge_missing_leaves_adds_legacy_branch_once() -> None:
    tree = {"name": "Root", "children": [{"name": "Otros", "children": []}]}

    merged, changed = merge_missing_leaves(
        tree,
        ["Categoria nueva", "Categoria nueva", "Optica"],
    )

    assert changed is True
    legacy_children = [
        child["name"]
        for child in merged["children"]
        if child["name"] == "Legado"
        for child in child["children"]
    ]
    assert legacy_children == ["Categoria nueva", "Óptica"]


def test_build_assignments_from_leaf_series_sets_levels_and_metadata() -> None:
    tree = {
        "name": "Root",
        "children": [
            {
                "name": "Transferencias y Personal",
                "children": [{"name": "Bizum enviado"}],
            }
        ],
    }
    leaf_series = pd.Series(["Bizum enviado"], index=[4], dtype="object")

    assignments = build_assignments_from_leaf_series(
        leaf_series,
        tree,
        source="ia_arbol:Transferencias y Personal > Bizum enviado",
        confidence=0.91,
        reason="Ruta IA: Transferencias y Personal > Bizum enviado",
    )

    assert assignments.loc[4, "Grupo"] == "Bizum enviado"
    assert (
        assignments.loc[4, "CategoriaPath"]
        == "Transferencias y Personal > Bizum enviado"
    )
    assert assignments.loc[4, "CategoriaNivel1"] == "Transferencias y Personal"
    assert assignments.loc[4, "CategoriaNivel2"] == "Bizum enviado"
    assert assignments.loc[4, "CategoriaFuente"].startswith("ia_arbol:")
