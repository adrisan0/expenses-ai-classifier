"""Tests for pure dashboard helpers."""
from __future__ import annotations

import pandas as pd

from src.ui.dashboard import (
    build_daily_consumption_heatmap,
    build_taxonomy_nodes,
    format_memory_entries_for_display,
    format_memory_entries_for_prompt,
    humanize_group_label,
    normalize_memory_entries,
    normalize_ui_state,
    summarize_subset,
    summarize_taxonomy_path,
)
from src.core.insights import MonthlyFlowItem


TREE = {
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


def test_build_taxonomy_nodes_counts_paths_and_pending_rows() -> None:
    frame = pd.DataFrame(
        {
            "Grupo": ["Supermercado", pd.NA, "Revision manual"],
            "CategoriaPath": [
                "Alimentacion > Supermercado",
                pd.NA,
                "Otros > Revision manual",
            ],
        }
    )

    nodes, pending_total = build_taxonomy_nodes(TREE, frame, selected_path="Otros")

    node_by_path = {node.path: node for node in nodes}
    assert pending_total == 1
    assert node_by_path["Root"].count == 3
    assert node_by_path["Alimentacion"].count == 1
    assert node_by_path["Alimentacion > Supermercado"].count == 1
    assert node_by_path["Otros"].count == 1
    assert node_by_path["Otros"].selected is True
    assert node_by_path["Otros > Revision manual"].is_leaf is True
    assert node_by_path["Otros"].has_children is True


def test_build_taxonomy_nodes_supports_collapsed_and_search_views() -> None:
    frame = pd.DataFrame(
        {
            "Grupo": ["Supermercado", "Revision manual"],
            "CategoriaPath": [
                "Alimentacion > Supermercado",
                "Otros > Revision manual",
            ],
        }
    )

    collapsed, _ = build_taxonomy_nodes(
        TREE,
        frame,
        selected_path="Root",
        expanded_paths={"Root"},
    )
    searched, _ = build_taxonomy_nodes(
        TREE,
        frame,
        selected_path="Root",
        expanded_paths={"Root"},
        search_query="revision",
    )

    assert [node.path for node in collapsed] == ["Root", "Alimentacion", "Otros"]
    assert [node.path for node in searched] == [
        "Root",
        "Otros",
        "Otros > Revision manual",
    ]


def test_normalize_ui_state_keeps_only_supported_values() -> None:
    payload = {
        "group_by": "CategoriaNivel1",
        "pattern": "uber",
        "date_from": "2025-01-01",
        "date_to": "2025-12-31",
        "amount_min": "-100000.00",
        "amount_max": "100000.00",
        "time_display": "seasonal",
        "visual_month": "2025-11",
        "chart_type": "cashflow",
        "ai_scope": "selection",
        "current_tab": "vision",
        "vision_section": "analysis",
        "category_mode": True,
        "timeline_show_accumulated_saving": True,
        "timeline_show_accumulated_investment": True,
        "timeline_show_expense": False,
        "timeline_show_salary": True,
        "timeline_show_balance": False,
        "timeline_show_accumulated_expense": True,
        "pending_only": False,
        "review_only": True,
        "scope_filter": False,
        "selected_node_path": "Ingresos",
        "bad_flag": "oops",
        "chart_type_bad": "unknown",
    }

    normalized = normalize_ui_state(payload)

    assert normalized["group_by"] == "CategoriaNivel1"
    assert normalized["time_display"] == "seasonal"
    assert normalized["chart_type"] == "cashflow"
    assert normalized["ai_scope"] == "selection"
    assert normalized["current_tab"] == "vision"
    assert normalized["vision_section"] == "analysis"
    assert normalized["category_mode"] is True
    assert normalized["timeline_show_accumulated_saving"] is True
    assert normalized["timeline_show_accumulated_investment"] is True
    assert normalized["timeline_show_expense"] is False
    assert normalized["timeline_show_salary"] is True
    assert normalized["timeline_show_balance"] is False
    assert normalized["timeline_show_accumulated_expense"] is True
    assert "bad_flag" not in normalized
    assert "chart_type_bad" not in normalized


def test_summarize_subset_uses_real_income_consistently() -> None:
    frame = pd.DataFrame(
        {
            "Concepto": [
                "Mercadona",
                "Traspaso ahorro",
                "Nomina",
                "Bizum clase",
                "Transferencia dudosa",
                "Bizum cena",
            ],
            "Importe": [-50.0, -300.0, 1200.0, 30.0, 45.0, 25.0],
            "CategoriaNivel1": [
                "Alimentacion",
                "Ahorro",
                "Ingreso",
                "Ingreso",
                "Ingreso",
                "Transferencias",
            ],
            "Movimiento": [
                "TARJETA",
                "Cuenta remunerada pibank",
                "TRANSFERENCIA",
                "BIZUM CLASE MATEMATICAS",
                "TRANSFERENCIA",
                "BIZUM",
            ],
        }
    )

    summary = summarize_subset(frame, "CategoriaNivel1")
    by_group = summary.set_index("CategoriaNivel1")

    assert humanize_group_label("CategoriaNivel1") == "categoria principal"
    assert by_group.loc["Alimentacion", "gasto_consumo"] == 50.0
    assert by_group.loc["Ahorro", "gasto_consumo"] == 0.0
    assert by_group.loc["Ahorro", "ahorro"] == 300.0
    assert by_group.loc["Ingreso", "importe_total"] == 1275.0
    assert by_group.loc["Ingreso", "ingresos"] == 1230.0
    assert by_group.loc["Transferencias", "ingresos"] == 0.0


def test_summarize_taxonomy_path_reports_financial_weight() -> None:
    frame = pd.DataFrame(
        {
            "Concepto": ["Mercadona", "Traspaso ahorro", "Nomina", "Bizum cena"],
            "Movimiento": [
                "TARJETA",
                "Cuenta remunerada pibank",
                "TRANSFERENCIA",
                "BIZUM",
            ],
            "Importe": [-50.0, -300.0, 1200.0, 25.0],
            "CategoriaPath": [
                "Alimentacion > Supermercado",
                "Ahorro > Cuenta ahorro",
                "Ingresos > Nomina",
                "Transferencias > Bizum recibido",
            ],
        }
    )

    root_summary = summarize_taxonomy_path(frame, "Root")
    saving_summary = summarize_taxonomy_path(frame, "Ahorro")

    assert root_summary.count == 4
    assert root_summary.consumption == 50.0
    assert root_summary.income == 1200.0
    assert root_summary.saving == 300.0
    assert saving_summary.count == 1
    assert saving_summary.saving == 300.0


def test_memory_entries_are_formalized_for_deepseek_prompt() -> None:
    entries = normalize_memory_entries(
        {
            "entries": [
                {
                    "text": "Cena de grupo pagada por mi el 2026-03-10.",
                    "source": "usuario",
                    "origin": "formulario_memoria",
                    "created_at": "2026-05-24T10:00:00+00:00",
                }
            ]
        }
    )

    prompt_context = format_memory_entries_for_prompt(entries)
    display_context = format_memory_entries_for_display(entries)

    assert entries[0]["source"] == "usuario"
    assert "fuente=usuario" in prompt_context
    assert "origen=formulario_memoria" in prompt_context
    assert "Cena de grupo" in display_context


def test_memory_entries_keep_legacy_confirmed_context() -> None:
    entries = normalize_memory_entries(
        {"confirmed_context": "Viaje confirmado a Lisboa."}
    )

    assert entries == [
        {
            "text": "Viaje confirmado a Lisboa.",
            "source": "usuario",
            "origin": "memoria_legacy",
            "created_at": "",
        }
    ]





def test_daily_consumption_heatmap_uses_only_real_consumption() -> None:
    frame = pd.DataFrame(
        {
            "Fecha": [
                "2026-05-01",
                "2026-05-01",
                "2026-05-02",
                "2026-05-03",
                "2026-05-04",
            ],
            "Concepto": [
                "Restaurante",
                "Supermercado",
                "Nomina",
                "Pibank ahorro",
                "Broker inversion",
            ],
            "Movimiento": [
                "TARJETA",
                "TARJETA",
                "TRANSFERENCIA NOMINA",
                "TRASPASO",
                "COMPRA",
            ],
            "Importe": [-40.0, -10.0, 1200.0, -300.0, -200.0],
        }
    )

    items = build_daily_consumption_heatmap(frame, max_days=7)
    positive = [item for item in items if item.amount > 0]

    assert len(items) == 4
    assert len(positive) == 1
    assert positive[0].date.strftime("%Y-%m-%d") == "2026-05-01"
    assert positive[0].amount == 50.0
    assert positive[0].count == 2
