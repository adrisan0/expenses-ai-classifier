"""Tests for local regex rules."""
from __future__ import annotations

import pandas as pd

from src.core.rules import build_group_column, parse_category_rules


def test_parse_category_rules_supports_multiple_separators() -> None:
    rules = parse_category_rules(
        "# comentario\n"
        "Supermercado = mercadona|alcampo\n"
        "Transferencias: bizum|transferencia\n"
        "Invalida\n"
    )

    assert [name for name, _ in rules] == ["Supermercado", "Transferencias"]


def test_build_group_column_preserves_existing_groups_and_fills_missing() -> None:
    dataframe = pd.DataFrame(
        {
            "Concepto": ["Mercadona", "Bizum", "Restaurante"],
            "Movimiento": ["Compra", "Transferencia", "Cena"],
            "Grupo": ["Compras", pd.NA, ""],
        }
    )
    rules = parse_category_rules(
        "Supermercado = mercadona\n"
        "Transferencias = bizum|transferencia\n"
        "Ocio = restaurante|cena\n"
    )

    group_column = build_group_column(dataframe, rules)

    assert group_column.tolist() == ["Compras", "Transferencias", "Ocio"]
