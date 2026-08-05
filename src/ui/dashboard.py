"""Flet dashboard for exploring and categorizing expense movements."""
from __future__ import annotations

import asyncio
import html
import json
import logging
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.insights import (
    build_average_period_health,
    MonthlyHealth,
    MonthlyFlowItem,
    SavingsOpportunity,
    SectorSpendItem,
    SpendBreakdownItem,
    TimeDisplayMode,
    build_bizum_intent_summary,
    build_cashflow_nature_summary,
    build_deepseek_agent_context,
    build_audit_metrics,
    classify_cashflow_nature,
    filter_dataframe_to_month,
    build_monthly_health,
    build_monthly_flow,
    build_personal_finance_summary,
    build_prioritized_review_queue,
    build_sector_spend,
    build_savings_opportunities,
    build_spend_breakdown,
    build_time_display_mode,
    scale_spend_breakdown,
    summarize_ai_trace,
)
from src.core.processing import build_filter_mask
from src.core.review import (
    DEFAULT_REVIEW_CONFIDENCE,
    build_pending_mask,
    build_review_mask,
)
from src.core.rules import CompiledRule, build_group_column, parse_category_rules
from src.core.taxonomy import (
    CATEGORY_COLUMNS,
    apply_category_assignments,
    build_assignments_from_leaf_series,
    ensure_category_columns,
    find_node_by_path,
    load_category_tree,
    merge_missing_leaves,
    save_category_tree,
)
from src.infra.deepseek import ask_deepseek_about_expenses, categorize_transactions_by_tree
from src.infra.storage import save_categories_to_disk
from src.support.paths import (
    ANALYSIS_CONTEXT_FILE,
    CATEGORIES_FILE,
    CATEGORY_TREE_FILE,
    DIR_CACHE,
    DIR_EXPORTS,
    EXPORT_FILE,
    LLM_CACHE_FILE,
    RULES_FILE,
    TREE_LLM_CACHE_FILE,
    UI_STATE_FILE,
)

logger = logging.getLogger(__name__)

GUI_IMPORT_ERROR = ""

try:  # pragma: no cover - depends on local GUI support
    import flet as ft
except Exception as exc:  # pragma: no cover - depends on local GUI support
    GUI_IMPORT_ERROR = str(exc)
    ft = None


RULES_PLACEHOLDER = (
    "# Ejemplos\n"
    "Supermercado = mercadona|alcampo\n"
    "Transferencias = bizum|transferencia\n"
    "Ocio = cine|restaurante\n"
)

DETAIL_ROW_LIMIT = 150
SUMMARY_ROW_LIMIT = 80
TIME_DISPLAY_OPTIONS = {
    "total",
    "daily",
    "monthly_average",
    "monthly_selected",
    "seasonal",
    "yearly",
}
CHART_TYPE_OPTIONS = {
    "sectors",
    "categories",
    "timeline",
    "income",
    "cashflow",
    "opportunities",
    "heatmap",
}
TAB_OPTIONS = {"vision", "rules", "tree"}
VISION_SECTION_OPTIONS = {"overview", "analysis", "movements"}
AI_SCOPE_OPTIONS = {"filtered_view", "selection"}
GROUP_OPTIONS = {
    "Concepto",
    "Movimiento",
    "CategoriaNivel1",
    "CategoriaLeaf",
    "CategoriaPath",
}
GROUP_LABELS = {
    "Concepto": "concepto",
    "Movimiento": "tipo de movimiento",
    "CategoriaNivel1": "categoria principal",
    "CategoriaLeaf": "categoria",
    "CategoriaPath": "ruta de categoria",
}


from src.ui.theme import PALETTE, setup_page
from src.ui.components.charts import build_spend_breakdown_chart, build_monthly_chart, build_value_bar_chart, build_daily_heatmap_chart


@dataclass(frozen=True, slots=True)
class TaxonomyNodeView:
    path: str
    label: str
    depth: int
    count: int
    share_text: str
    is_leaf: bool
    expanded: bool
    has_children: bool
    selected: bool


@dataclass(frozen=True, slots=True)
class TaxonomyNodeSummary:
    """Financial summary for a taxonomy path in the current view."""

    path: str
    count: int
    pending: int
    consumption: float
    income: float
    saving: float
    investment: float
    net: float


@dataclass(frozen=True, slots=True)
class DailyConsumptionHeatmapItem:
    """Daily consumption weight for the visual savings heatmap."""

    date: pd.Timestamp
    amount: float
    count: int
    ratio: float


def normalize_ui_state(payload: Any) -> dict[str, Any]:
    """Return a safe persisted UI-state payload."""
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, Any] = {}
    string_fields = {
        "group_by",
        "pattern",
        "date_from",
        "date_to",
        "amount_min",
        "amount_max",
        "visual_month",
        "selected_node_path",
    }
    for field in string_fields:
        value = payload.get(field)
        if value is None:
            continue
        normalized[field] = str(value)

    time_display = str(payload.get("time_display", "")).strip()
    if time_display in TIME_DISPLAY_OPTIONS:
        normalized["time_display"] = time_display

    chart_type = str(payload.get("chart_type", "")).strip()
    if chart_type in CHART_TYPE_OPTIONS:
        normalized["chart_type"] = chart_type

    current_tab = str(payload.get("current_tab", "")).strip()
    if current_tab in TAB_OPTIONS:
        normalized["current_tab"] = current_tab

    vision_section = str(payload.get("vision_section", "")).strip()
    if vision_section in VISION_SECTION_OPTIONS:
        normalized["vision_section"] = vision_section

    ai_scope = str(payload.get("ai_scope", "")).strip()
    if ai_scope in AI_SCOPE_OPTIONS:
        normalized["ai_scope"] = ai_scope

    bool_fields = {
        "category_mode",
        "pending_only",
        "review_only",
        "scope_filter",
        "timeline_show_expense",
        "timeline_show_salary",
        "timeline_show_balance",
        "timeline_show_accumulated_expense",
        "timeline_show_accumulated_saving",
        "timeline_show_accumulated_investment",
    }
    for field in bool_fields:
        value = payload.get(field)
        if isinstance(value, bool):
            normalized[field] = value

    return normalized


def is_gui_available() -> bool:
    """Return whether Flet is available in the current environment."""
    return ft is not None


def gui_unavailable_reason() -> str:
    """Return the reason why the GUI cannot start, if known."""
    return GUI_IMPORT_ERROR or "Flet no esta disponible en este entorno."


def run_gui(df: pd.DataFrame) -> None:
    """Launch the interactive Flet dashboard."""
    if not is_gui_available():
        raise RuntimeError(
            f"{gui_unavailable_reason()} Ejecuta en una maquina con GUI."
        )

    assert ft is not None

    def main(page: ft.Page) -> None:
        dashboard = ExpensesDashboard(page, df)
        dashboard.mount()

    ft.run(main)


def format_count(value: int) -> str:
    """Format integer counts for compact UI presentation."""
    return f"{value:,}".replace(",", ".")


def format_currency(value: float) -> str:
    """Format EUR amounts using local Spanish-style separators."""
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{formatted} EUR"


def format_percent(ratio: float) -> str:
    """Format a ratio as a percentage."""
    return f"{ratio * 100:.1f}%".replace(".", ",")


def humanize_group_label(group_col: str) -> str:
    """Return a readable grouping label for UI copy."""
    return GROUP_LABELS.get(group_col, group_col.replace("_", " ").strip().lower())


def _weekday_name(date: pd.Timestamp) -> str:
    return _weekday_name_from_index(int(date.weekday()))


def _weekday_name_from_index(index: int) -> str:
    names = [
        "Lunes",
        "Martes",
        "Miercoles",
        "Jueves",
        "Viernes",
        "Sabado",
        "Domingo",
    ]
    return names[max(0, min(index, 6))]


def format_confidence(value: Any) -> str:
    """Render a confidence value if present."""
    if pd.isna(value):
        return ""
    try:
        return format_percent(float(value))
    except (TypeError, ValueError):
        return ""


def _empty_chart_svg(message: str) -> str:
    safe_message = html.escape(message)
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='220' "
        "viewBox='0 0 640 220'>"
        "<rect width='640' height='220' rx='18' fill='#101C24'/>"
        f"<text x='320' y='112' fill='#8EA3AD' font-size='16' "
        f"text-anchor='middle' font-family='Arial'>{safe_message}</text>"
        "</svg>"
    )


def build_spend_breakdown_chart(items: list[SpendBreakdownItem]) -> str:
    if not items:
        return _empty_chart_svg("Sin gastos para graficar")
    rows = items[:6]
    max_amount = max((item.amount for item in rows), default=0.0)
    if max_amount <= 0:
        return _empty_chart_svg("Sin gastos para graficar")

    parts = [_chart_svg_start()]
    chart_x = 170
    bar_max = 390
    y = 32
    for index, item in enumerate(rows):
        width = max(4, int((item.amount / max_amount) * bar_max))
        color = "#F06A63" if index == 0 else "#D2A54C" if index == 1 else "#1DB4A5"
        label = _truncate_svg_text(item.label, 22)
        amount = html.escape(format_currency(item.amount))
        parts.append(
            f"<text x='18' y='{y + 17}' fill='#E8F1F2' font-size='13' "
            f"font-weight='700' font-family='Arial'>{html.escape(label)}</text>"
        )
        parts.append(
            f"<rect x='{chart_x}' y='{y}' width='{bar_max}' height='20' rx='8' "
            "fill='#15242D'/>"
        )
        parts.append(
            f"<rect x='{chart_x}' y='{y}' width='{width}' height='20' rx='8' "
            f"fill='{color}'/>"
        )
        parts.append(
            f"<text x='615' y='{y + 16}' fill='#E8F1F2' font-size='12' "
            f"text-anchor='end' font-family='Arial'>{amount}</text>"
        )
        y += 31
    parts.append("</svg>")
    return "".join(parts)


def build_monthly_chart(
    items: list[MonthlyFlowItem],
    *,
    zoom_x: float = 1.0,
    zoom_y: float = 1.0,
    show_expense: bool = True,
    show_salary: bool = True,
    show_balance: bool = True,
    show_accumulated_expense: bool = True,
    show_accumulated_saving: bool = False,
    show_accumulated_investment: bool = False,
) -> str:
    if not items:
        return _empty_chart_svg("Sin meses para graficar")
    if not any(
        [
            show_expense,
            show_salary,
            show_balance,
            show_accumulated_expense,
            show_accumulated_saving,
            show_accumulated_investment,
        ]
    ):
        return _empty_chart_svg("Activa al menos una serie")
    safe_zoom_x = max(float(zoom_x), 1.0)
    safe_zoom_y = max(float(zoom_y), 1.0)
    visible_count = max(3, int(len(items) / safe_zoom_x))
    rows = items[-visible_count:]
    accumulated_expense = 0.0
    accumulated_saving = 0.0
    accumulated_investment = 0.0
    cumulative_values: list[tuple[float, float, float]] = []
    for item in rows:
        accumulated_expense += item.expenses
        accumulated_saving += item.saving
        accumulated_investment += item.investment
        cumulative_values.append(
            (accumulated_expense, accumulated_saving, accumulated_investment)
        )
    max_candidates: list[float] = []
    if show_expense:
        max_candidates.extend(item.expenses for item in rows)
    if show_salary:
        max_candidates.extend(item.salary or 0.0 for item in rows)
    if show_balance:
        max_candidates.extend(item.bank_cashflow for item in rows)
    if show_accumulated_expense:
        max_candidates.extend(value[0] for value in cumulative_values)
    if show_accumulated_saving:
        max_candidates.extend(value[1] for value in cumulative_values)
    if show_accumulated_investment:
        max_candidates.extend(value[2] for value in cumulative_values)
    max_value = max(max_candidates, default=0.0)
    min_candidates = [0.0]
    if show_balance:
        min_candidates.extend(item.bank_cashflow for item in rows)
    min_value = min(min_candidates, default=0.0)
    if max_value <= 0 and min_value >= 0:
        return _empty_chart_svg("Sin importes para graficar")

    parts = [_chart_svg_start()]
    left = 44
    right = 612
    top = 34
    bottom = 178
    chart_width = right - left
    group_width = chart_width / max(len(rows), 1)
    bar_width = max(3, min(18, (group_width - 8) / 2))
    y_max = max(max_value / safe_zoom_y, 1.0)
    y_min = min(min_value / safe_zoom_y, 0.0)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def y_for(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        ratio = (clipped - y_min) / (y_max - y_min)
        return bottom - (ratio * (bottom - top))

    zero_y = y_for(0.0)
    line_points: list[str] = []
    cumulative_expense_points: list[str] = []
    cumulative_saving_points: list[str] = []
    cumulative_investment_points: list[str] = []
    for index, item in enumerate(rows):
        x = left + (index * group_width) + max((group_width - (bar_width * 2) - 4) / 2, 0)
        expense_y = y_for(item.expenses)
        salary_y = y_for(item.salary or 0.0)
        expense_h = max(0.0, zero_y - expense_y)
        salary_h = max(0.0, zero_y - salary_y)
        if show_expense:
            parts.append(
                f"<rect x='{x:.1f}' y='{expense_y:.1f}' width='{bar_width:.1f}' "
                f"height='{expense_h:.1f}' rx='4' fill='#F06A63'/>"
            )
        if show_salary:
            parts.append(
                f"<rect x='{x + bar_width + 4:.1f}' y='{salary_y:.1f}' "
                f"width='{bar_width:.1f}' height='{salary_h:.1f}' rx='4' fill='#D2A54C'/>"
            )
        point_x = left + (index * group_width) + (group_width / 2)
        if show_balance:
            line_points.append(f"{point_x:.1f},{y_for(item.bank_cashflow):.1f}")
        if show_accumulated_expense:
            cumulative_expense_points.append(
                f"{point_x:.1f},{y_for(cumulative_values[index][0]):.1f}"
            )
        if show_accumulated_saving:
            cumulative_saving_points.append(
                f"{point_x:.1f},{y_for(cumulative_values[index][1]):.1f}"
            )
        if show_accumulated_investment:
            cumulative_investment_points.append(
                f"{point_x:.1f},{y_for(cumulative_values[index][2]):.1f}"
            )
        label_step = max(1, int(len(rows) / 8))
        if index % label_step == 0 or index == len(rows) - 1:
            parts.append(
                f"<text x='{point_x:.1f}' y='202' fill='#8EA3AD' font-size='10' "
                f"text-anchor='middle' font-family='Arial'>{html.escape(item.month[2:])}</text>"
            )
    parts.append(
        f"<line x1='{left}' y1='{zero_y:.1f}' x2='{right}' y2='{zero_y:.1f}' "
        "stroke='#425762' stroke-width='1'/>"
    )
    if len(line_points) >= 2:
        parts.append(
            f"<polyline points='{' '.join(line_points)}' fill='none' "
            "stroke='#45C486' stroke-width='2.5' stroke-linecap='round' "
            "stroke-linejoin='round'/>"
        )
    if len(cumulative_expense_points) >= 2:
        parts.append(
            f"<polyline points='{' '.join(cumulative_expense_points)}' fill='none' "
            "stroke='#F06A63' stroke-width='2.5' stroke-linecap='round' "
            "stroke-linejoin='round' stroke-dasharray='6 4'/>"
        )
    if len(cumulative_saving_points) >= 2:
        parts.append(
            f"<polyline points='{' '.join(cumulative_saving_points)}' fill='none' "
            "stroke='#5BC0BE' stroke-width='2.5' stroke-linecap='round' "
            "stroke-linejoin='round' stroke-dasharray='5 5'/>"
        )
    if len(cumulative_investment_points) >= 2:
        parts.append(
            f"<polyline points='{' '.join(cumulative_investment_points)}' fill='none' "
            "stroke='#7C8DFF' stroke-width='2.5' stroke-linecap='round' "
            "stroke-linejoin='round' stroke-dasharray='3 5'/>"
        )
    legend_x = 42
    if show_expense:
        parts.append(
            f"<text x='{legend_x}' y='22' fill='#F06A63' font-size='12' "
            "font-family='Arial'>Gasto</text>"
        )
        legend_x += 66
    if show_salary:
        parts.append(
            f"<text x='{legend_x}' y='22' fill='#D2A54C' font-size='12' "
            "font-family='Arial'>Nomina</text>"
        )
        legend_x += 66
    if show_balance:
        parts.append(
            f"<text x='{legend_x}' y='22' fill='#45C486' font-size='12' "
            "font-family='Arial'>Balance</text>"
        )
        legend_x += 68
    if show_accumulated_expense:
        parts.append(
            f"<text x='{legend_x}' y='22' fill='#F06A63' font-size='12' "
            "font-family='Arial'>Gasto acum.</text>"
        )
        legend_x += 90
    if show_accumulated_saving:
        parts.append(
            f"<text x='{legend_x}' y='22' fill='#5BC0BE' font-size='12' "
            "font-family='Arial'>Ahorro acum.</text>"
        )
        legend_x += 94
    if show_accumulated_investment:
        parts.append(
            f"<text x='{legend_x}' y='22' fill='#7C8DFF' font-size='12' "
            "font-family='Arial'>Inversion acum.</text>"
        )
    parts.append(
        f"<text x='615' y='22' fill='#8EA3AD' font-size='11' "
        f"text-anchor='end' font-family='Arial'>{len(rows)}/{len(items)} meses | "
        f"X {safe_zoom_x:.1f}x Y {safe_zoom_y:.1f}x</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def build_daily_consumption_heatmap(
    dataframe: pd.DataFrame,
    *,
    max_days: int = 112,
) -> list[DailyConsumptionHeatmapItem]:
    """Return daily consumption totals including zero-spend days."""
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return []

    working = dataframe.copy()
    working["_heatmap_date"] = pd.to_datetime(
        working["Fecha"],
        errors="coerce",
    ).dt.normalize()
    amounts = pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(working)
    mask = nature.eq("consumo") & amounts.lt(0) & working["_heatmap_date"].notna()
    if not bool(mask.any()):
        return []

    spend = (
        working.loc[mask]
        .assign(_heatmap_amount=amounts.loc[mask].abs())
        .groupby("_heatmap_date", sort=True)
        .agg(amount=("_heatmap_amount", "sum"), count=("_heatmap_amount", "size"))
    )
    valid_dates = working["_heatmap_date"].dropna()
    end_date = pd.Timestamp(valid_dates.max()).normalize()
    raw_start = pd.Timestamp(valid_dates.min()).normalize()
    bounded_start = max(raw_start, end_date - pd.Timedelta(days=max(max_days - 1, 1)))
    dates = pd.date_range(bounded_start, end_date, freq="D")
    max_amount = float(spend["amount"].max())

    items: list[DailyConsumptionHeatmapItem] = []
    for date in dates:
        if date in spend.index:
            amount = float(spend.loc[date, "amount"])
            count = int(spend.loc[date, "count"])
        else:
            amount = 0.0
            count = 0
        items.append(
            DailyConsumptionHeatmapItem(
                date=pd.Timestamp(date),
                amount=amount,
                count=count,
                ratio=(amount / max_amount) if max_amount > 0 else 0.0,
            )
        )
    return items


def build_daily_heatmap_chart(items: list[DailyConsumptionHeatmapItem]) -> str:
    if not items or not any(item.amount > 0 for item in items):
        return _empty_chart_svg("Sin gasto diario para mapa de calor")

    start_monday = items[0].date - pd.Timedelta(days=int(items[0].date.weekday()))
    max_item = max(items, key=lambda item: item.amount)
    max_week = max(
        int((item.date - start_monday).days // 7)
        for item in items
    )
    cell = 15
    gap = 5
    left = 86
    top = 48
    parts = [_chart_svg_start()]
    parts.append(
        "<text x='18' y='24' fill='#E8F1F2' font-size='13' "
        "font-weight='700' font-family='Arial'>Picos de consumo diario</text>"
    )
    parts.append(
        f"<text x='615' y='24' fill='#8EA3AD' font-size='11' "
        f"text-anchor='end' font-family='Arial'>Pico "
        f"{html.escape(max_item.date.strftime('%Y-%m-%d'))}: "
        f"{html.escape(format_currency(max_item.amount))}</text>"
    )

    for label, weekday in (("L", 0), ("M", 1), ("X", 2), ("J", 3), ("V", 4), ("S", 5), ("D", 6)):
        y = top + weekday * (cell + gap) + 12
        parts.append(
            f"<text x='58' y='{y}' fill='#8EA3AD' font-size='11' "
            f"text-anchor='middle' font-family='Arial'>{label}</text>"
        )

    for item in items:
        week = int((item.date - start_monday).days // 7)
        weekday = int(item.date.weekday())
        x = left + week * (cell + gap)
        y = top + weekday * (cell + gap)
        color = _heatmap_color(item.ratio)
        stroke = "#E8F1F2" if item == max_item else "#243842"
        parts.append(
            f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' rx='4' "
            f"fill='{color}' stroke='{stroke}' stroke-width='1'>"
            f"<title>{html.escape(item.date.strftime('%Y-%m-%d'))}: "
            f"{html.escape(format_currency(item.amount))} | "
            f"{format_count(item.count)} movimientos</title>"
        )
        if item == max_item:
            parts.append(
                "<animate attributeName='stroke-width' values='1;3;1' "
                "dur='1.8s' repeatCount='indefinite'/>"
            )
        parts.append("</rect>")

    month_markers: dict[str, int] = {}
    for item in items:
        label = item.date.strftime("%b").lower()[:3]
        month_markers.setdefault(label, int((item.date - start_monday).days // 7))
    for label, week in month_markers.items():
        x = left + week * (cell + gap)
        if x <= 590:
            parts.append(
                f"<text x='{x}' y='198' fill='#8EA3AD' font-size='10' "
                f"font-family='Arial'>{html.escape(label)}</text>"
            )

    legend_x = min(left + (max_week + 2) * (cell + gap), 500)
    parts.append(
        f"<text x='{legend_x}' y='74' fill='#8EA3AD' font-size='10' "
        "font-family='Arial'>menos</text>"
    )
    for index, ratio in enumerate([0.0, 0.2, 0.4, 0.65, 0.9]):
        parts.append(
            f"<rect x='{legend_x + 42 + index * 20}' y='62' width='14' "
            f"height='14' rx='4' fill='{_heatmap_color(ratio)}' "
            "stroke='#243842' stroke-width='1'/>"
        )
    parts.append(
        f"<text x='{legend_x + 146}' y='74' fill='#8EA3AD' font-size='10' "
        "font-family='Arial'>mas</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _heatmap_color(ratio: float) -> str:
    if ratio <= 0:
        return "#15242D"
    if ratio < 0.18:
        return "#1D3A37"
    if ratio < 0.38:
        return "#31574B"
    if ratio < 0.62:
        return "#806337"
    if ratio < 0.82:
        return "#B76548"
    return "#F06A63"


def build_value_bar_chart(values: list[tuple[str, float, str]]) -> str:
    positive_values = [(label, value, color) for label, value, color in values if value > 0]
    if not positive_values:
        return _empty_chart_svg("Sin valores para graficar")
    max_value = max(value for _, value, _ in positive_values)
    parts = [_chart_svg_start()]
    chart_x = 180
    bar_max = 380
    y = 34
    for label, value, color in positive_values[:5]:
        width = max(4, int((value / max_value) * bar_max))
        parts.append(
            f"<text x='18' y='{y + 17}' fill='#E8F1F2' font-size='13' "
            f"font-weight='700' font-family='Arial'>{html.escape(_truncate_svg_text(label, 22))}</text>"
        )
        parts.append(
            f"<rect x='{chart_x}' y='{y}' width='{bar_max}' height='22' rx='8' "
            "fill='#15242D'/>"
        )
        parts.append(
            f"<rect x='{chart_x}' y='{y}' width='{width}' height='22' rx='8' "
            f"fill='{color}'/>"
        )
        parts.append(
            f"<text x='615' y='{y + 17}' fill='#E8F1F2' font-size='12' "
            f"text-anchor='end' font-family='Arial'>{html.escape(format_currency(value))}</text>"
        )
        y += 35
    parts.append("</svg>")
    return "".join(parts)


def _chart_svg_start() -> str:
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='220' "
        "viewBox='0 0 640 220'>"
        "<rect width='640' height='220' rx='18' fill='#101C24'/>"
        "<line x1='18' y1='184' x2='622' y2='184' stroke='#243842' stroke-width='1'/>"
    )


def _truncate_svg_text(value: str, limit: int) -> str:
    normalized = str(value).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 1, 1)] + "..."


def parse_float(value: str) -> float | None:
    """Parse text into float tolerating local formatting."""
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return float(normalized.replace(".", "").replace(",", "."))
    except ValueError:
        try:
            return float(normalized.replace(",", "."))
        except ValueError:
            return None


def parse_date(value: str) -> pd.Timestamp | None:
    """Parse text into pandas timestamp."""
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return pd.to_datetime(normalized)
    except (TypeError, ValueError):
        return None


def normalize_memory_entries(payload: Any) -> list[dict[str, str]]:
    """Normalize stored memory entries while tolerating legacy payloads."""
    entries: list[dict[str, str]] = []
    if isinstance(payload, dict):
        raw_entries = payload.get("entries")
        if isinstance(raw_entries, list):
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    continue
                text = str(raw_entry.get("text", "")).strip()
                if not text:
                    continue
                entries.append(
                    {
                        "text": text,
                        "source": str(raw_entry.get("source", "usuario")).strip()
                        or "usuario",
                        "origin": str(
                            raw_entry.get("origin", "formulario_memoria")
                        ).strip()
                        or "formulario_memoria",
                        "created_at": str(raw_entry.get("created_at", "")).strip(),
                    }
                )
        legacy_text = str(payload.get("confirmed_context", "")).strip()
        if legacy_text and not entries:
            entries.append(
                {
                    "text": legacy_text,
                    "source": "usuario",
                    "origin": "memoria_legacy",
                    "created_at": "",
                }
            )
    elif isinstance(payload, str):
        legacy_text = payload.strip()
        if legacy_text:
            entries.append(
                {
                    "text": legacy_text,
                    "source": "usuario",
                    "origin": "memoria_legacy",
                    "created_at": "",
                }
            )
    return entries


def format_memory_entries_for_prompt(entries: list[dict[str, str]]) -> str:
    """Build a formal memory block for the DeepSeek prompt context."""
    if not entries:
        return ""
    lines = ["Memoria confirmada registrada por el usuario:"]
    for index, entry in enumerate(entries, start=1):
        created_at = entry.get("created_at") or "sin_fecha_registro"
        lines.append(
            (
                f"{index}. fuente={entry.get('source', 'usuario')}; "
                f"origen={entry.get('origin', 'formulario_memoria')}; "
                f"registrado={created_at}; contenido={entry.get('text', '').strip()}"
            )
        )
    return "\n".join(lines)


def format_memory_entries_for_display(entries: list[dict[str, str]]) -> str:
    """Build a compact readable memory list for the UI."""
    if not entries:
        return "Sin memoria guardada."
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        created_at = entry.get("created_at") or "sin fecha"
        lines.append(
            f"**{index}. Usuario** · {created_at}\n\n{entry.get('text', '').strip()}"
        )
    return "\n\n".join(lines)


def summarize_subset(subset: pd.DataFrame, effective_group: str) -> pd.DataFrame:
    """Build the grouped summary shown in the dashboard."""
    columns = [
        effective_group,
        "count",
        "importe_total",
        "gasto_consumo",
        "ingresos",
        "ahorro",
        "inversion",
        "importe_medio",
    ]
    if subset.empty:
        return pd.DataFrame(columns=columns)

    if effective_group not in subset.columns:
        effective_group = "Concepto" if "Concepto" in subset.columns else effective_group
        columns[0] = effective_group
    if effective_group not in subset.columns or "Importe" not in subset.columns:
        return pd.DataFrame(columns=columns)

    working = subset.copy()
    amounts = pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(working)
    labels = working[effective_group].fillna("Sin categoria").astype(str).str.strip()
    working["_summary_group"] = labels.replace("", "Sin categoria")
    working["_amount"] = amounts
    working["_nature"] = nature

    rows: list[dict[str, Any]] = []
    for label, group in working.groupby("_summary_group", sort=False):
        group_amounts = group["_amount"]
        group_nature = group["_nature"]
        group_finance = build_personal_finance_summary(group)
        net_total = float(group_amounts.sum())
        count = int(len(group))
        rows.append(
            {
                effective_group: str(label),
                "count": count,
                "importe_total": net_total,
                "gasto_consumo": float(
                    group_amounts.loc[
                        (group_amounts < 0) & group_nature.eq("consumo")
                    ].abs().sum()
                ),
                "ingresos": group_finance.real_income_total,
                "ahorro": float(
                    group_amounts.loc[
                        (group_amounts < 0) & group_nature.eq("ahorro")
                    ].abs().sum()
                ),
                "inversion": float(
                    group_amounts.loc[
                        (group_amounts < 0) & group_nature.eq("inversion")
                    ].abs().sum()
                ),
                "importe_medio": net_total / count if count else 0.0,
            }
        )

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["gasto_consumo", "ahorro", "inversion", "ingresos", "count"],
            ascending=[False, False, False, False, False],
        )
        .reset_index(drop=True)
    )


def scale_summary_for_display(summary: pd.DataFrame, divisor: float) -> pd.DataFrame:
    """Scale monetary summary columns for the selected display mode."""
    if summary.empty:
        return summary
    safe_divisor = max(float(divisor), 1.0)
    scaled = summary.copy()
    for column in (
        "importe_total",
        "gasto_consumo",
        "ingresos",
        "ahorro",
        "inversion",
        "importe_medio",
    ):
        if column in scaled.columns:
            scaled[column] = pd.to_numeric(scaled[column], errors="coerce").fillna(0.0)
            scaled[column] = scaled[column] / safe_divisor
    return scaled


def scale_sector_spend_for_display(
    items: list[SectorSpendItem],
    *,
    divisor: float,
) -> list[SectorSpendItem]:
    """Scale sector amounts for the selected display mode."""
    safe_divisor = max(float(divisor), 1.0)
    return [
        SectorSpendItem(
            label=item.label,
            amount=item.amount / safe_divisor,
            average_amount=item.average_amount / safe_divisor,
            delta_amount=(item.amount - item.average_amount) / safe_divisor,
            delta_ratio=(
                (item.amount - item.average_amount) / item.average_amount
                if item.average_amount > 0
                else None
            ),
            count=item.count,
            ratio=item.ratio,
        )
        for item in items
    ]


def build_taxonomy_nodes(
    tree: dict[str, Any],
    frame: pd.DataFrame,
    *,
    selected_path: str,
    expanded_paths: set[str] | None = None,
    search_query: str = "",
) -> tuple[list[TaxonomyNodeView], int]:
    """Flatten the taxonomy tree with counts for the current view."""
    assigned_counts: dict[str, int] = {"Root": len(frame)}
    pending_total = int(build_pending_mask(frame).sum()) if len(frame) else 0

    if "CategoriaPath" in frame.columns:
        categorized = frame.loc[~build_pending_mask(frame), "CategoriaPath"].dropna()
        for raw_path in categorized.astype(str):
            current_parts: list[str] = []
            for part in [part.strip() for part in raw_path.split(" > ") if part.strip()]:
                current_parts.append(part)
                current_path = " > ".join(current_parts)
                assigned_counts[current_path] = assigned_counts.get(current_path, 0) + 1

    total_rows = max(len(frame), 1)
    rows: list[TaxonomyNodeView] = []
    search_text = search_query.strip().lower()

    def walk(node: dict[str, Any], parent_path: str, depth: int) -> None:
        raw_name = str(node.get("name", "")).strip() or "Root"
        if raw_name == "Root":
            current_path = "Root"
            label = "Root"
        elif parent_path in {"", "Root"}:
            current_path = raw_name
            label = raw_name
        else:
            current_path = f"{parent_path} > {raw_name}"
            label = raw_name

        count = assigned_counts.get(current_path, 0)
        children = [child for child in node.get("children", []) if isinstance(child, dict)]
        has_children = bool(children)
        is_leaf = not has_children
        expanded = expanded_paths is None or current_path in expanded_paths
        share_text = (
            f"{format_count(pending_total)} pendientes"
            if current_path == "Root"
            else format_percent(count / total_rows)
        )
        rows.append(
            TaxonomyNodeView(
                path=current_path,
                label=label,
                depth=depth,
                count=count,
                share_text=share_text,
                is_leaf=is_leaf,
                expanded=expanded,
                has_children=has_children,
                selected=current_path == selected_path,
            )
        )
        if search_text or expanded:
            for child in children:
                walk(child, current_path, depth + 1)

    walk(tree, "", 0)
    if search_text:
        matching_paths = {
            node.path
            for node in rows
            if search_text in node.path.lower() or search_text in node.label.lower()
        }
        rows = [
            node
            for node in rows
            if node.path == "Root" and matching_paths
            or any(
                match == node.path or match.startswith(f"{node.path} >")
                for match in matching_paths
            )
        ]
    return rows, pending_total


def summarize_taxonomy_path(frame: pd.DataFrame, path: str) -> TaxonomyNodeSummary:
    """Summarize the financial content of a taxonomy path."""
    if frame.empty:
        return TaxonomyNodeSummary(path, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if path and path != "Root" and "CategoriaPath" in frame.columns:
        category_paths = frame["CategoriaPath"].fillna("").astype(str)
        scoped = frame.loc[
            category_paths.eq(path) | category_paths.str.startswith(f"{path} >")
        ].copy()
    else:
        scoped = frame.copy()

    if scoped.empty or "Importe" not in scoped.columns:
        return TaxonomyNodeSummary(path, len(scoped), 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    amounts = pd.to_numeric(scoped["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(scoped)
    pending = int(build_pending_mask(scoped).sum()) if len(scoped) else 0
    consumption = float(amounts.loc[(amounts < 0) & nature.eq("consumo")].abs().sum())
    income = float(amounts.loc[(amounts > 0) & nature.eq("ingreso")].sum())
    saving = float(amounts.loc[(amounts < 0) & nature.eq("ahorro")].abs().sum())
    investment = float(amounts.loc[(amounts < 0) & nature.eq("inversion")].abs().sum())
    return TaxonomyNodeSummary(
        path=path,
        count=len(scoped),
        pending=pending,
        consumption=consumption,
        income=income,
        saving=saving,
        investment=investment,
        net=float(amounts.sum()),
    )


class ExpensesDashboard:
    """Interactive Flet workspace for local transaction review."""

    def __init__(self, page: ft.Page, dataframe: pd.DataFrame) -> None:
        self.page = page
        self.df = dataframe.copy()
        self.current_subset = pd.DataFrame()
        self.current_summary = pd.DataFrame()
        self.current_group = "Concepto"
        self.current_monthly_health = MonthlyHealth(
            month=None,
            current_income=0.0,
            current_expense=0.0,
            current_margin=0.0,
            savings_rate=None,
            comparison_months=0,
            average_income=0.0,
            average_expense=0.0,
            average_margin=0.0,
            expense_delta=0.0,
            margin_delta=0.0,
            alert_label="Sin movimientos",
            alert_detail="No hay movimientos fechados para calcular salud mensual.",
        )
        self.current_spend_breakdown: list[SpendBreakdownItem] = []
        self.current_sector_spend: list[SectorSpendItem] = []
        self.current_monthly_flow: list[MonthlyFlowItem] = []
        self.current_savings_opportunities: list[SavingsOpportunity] = []
        self.current_time_mode = TimeDisplayMode("total", "Total del rango", 1.0, "", None)
        self.memory_entries = self._load_memory_entries()
        self.confirmed_context = format_memory_entries_for_prompt(self.memory_entries)
        self.focused_group: str | None = None
        self.selected_node_path = "Root"
        self.taxonomy_expanded_paths: set[str] = {"Root"}
        self.selected_detail_indices: set[int] = set()
        self.worker_active = False
        self._status_queue: queue.Queue[str] = queue.Queue()
        self.busy_controls: list[ft.Control] = []
        self.current_tab = "vision"
        self.current_vision_section = "overview"
        self.selected_visual_month: str | None = None
        self.chart_zoom_x = 1.0
        self.chart_zoom_y = 1.0
        self._chart_render_version = 0
        self._chart_render_path: Path | None = None

        self.category_tree = load_category_tree(CATEGORY_TREE_FILE)
        self.category_tree = self._sync_tree_with_dataframe(self.category_tree)
        self.df = ensure_category_columns(self.df, self.category_tree)
        self.has_saved_categories = bool(
            "Grupo" in self.df.columns and self.df["Grupo"].notna().any()
        )

        self.date_min = pd.to_datetime(self.df["Fecha"].min())
        self.date_max = pd.to_datetime(self.df["Fecha"].max())
        self.default_date_from = self.date_max.replace(day=1)
        self.month_min = self.date_min.to_period("M").to_timestamp()
        self.month_max = self.date_max.to_period("M").to_timestamp()
        self.month_count = self._month_index(self.month_max) + 1
        self.amount_min = -100000.0
        self.amount_max = 100000.0

        self._build_controls()

    def mount(self) -> None:
        """Attach the dashboard to the page."""
        self._configure_page()
        self.page.overlay.extend([self.date_from_picker, self.date_to_picker])
        self.page.bottom_appbar = ft.BottomAppBar(
            content=self._build_bottom_date_bar(),
            bgcolor=PALETTE.panel_alt,
            padding=0,
            elevation=10,
        )
        self.page.add(self.shell)
        self._load_rules_from_disk()
        self._load_ui_state()
        self.refresh_view("Panel listo para revision")

    def _configure_page(self) -> None:
        self.page.title = "Expenses Studio"
        setup_page(self.page)
        self.page.padding = 24
        self.page.spacing = 20
        self.page.scroll = ft.ScrollMode.HIDDEN
        self.page.theme_mode = ft.ThemeMode.DARK
        if not self.page.web:
            self.page.window.width = 1680
            self.page.window.height = 940
            self.page.window.min_width = 1280
            self.page.window.min_height = 820

    def _build_controls(self) -> None:
        self.group_dropdown = ft.Dropdown(
            label="Agrupar por",
            value="CategoriaNivel1" if self.has_saved_categories else "Concepto",
            options=[
                ft.dropdown.Option("Concepto", "Concepto"),
                ft.dropdown.Option("Movimiento", "Tipo de movimiento"),
                ft.dropdown.Option("CategoriaNivel1", "Categoria principal"),
                ft.dropdown.Option("CategoriaLeaf", "Categoria final"),
                ft.dropdown.Option("CategoriaPath", "Ruta completa"),
            ],
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            col={"xs": 12, "md": 6, "xl": 2},
        )
        self.pattern_field = self._text_field(
            label="Regex",
            value="",
            hint_text="mercadona|bizum|netflix",
            col={"xs": 12, "md": 6, "xl": 3},
        )
        self.date_from_picker = ft.DatePicker(
            value=self.default_date_from.date(),
            first_date=self.date_min.date(),
            last_date=self.date_max.date(),
            current_date=self.date_max.date(),
            date_picker_mode=ft.DatePickerMode.DAY,
            entry_mode=ft.DatePickerEntryMode.CALENDAR,
            field_label_text="Desde",
            help_text="Selecciona fecha inicial",
            confirm_text="Aplicar",
            cancel_text="Cancelar",
            on_change=self._on_date_from_change,
        )
        self.date_to_picker = ft.DatePicker(
            value=self.date_max.date(),
            first_date=self.date_min.date(),
            last_date=self.date_max.date(),
            current_date=self.date_max.date(),
            date_picker_mode=ft.DatePickerMode.DAY,
            entry_mode=ft.DatePickerEntryMode.CALENDAR,
            field_label_text="Hasta",
            help_text="Selecciona fecha final",
            confirm_text="Aplicar",
            cancel_text="Cancelar",
            on_change=self._on_date_to_change,
        )
        self.date_from_field = self._date_field(
            label="Desde",
            value=self.default_date_from.strftime("%Y-%m-%d"),
            on_click=lambda _event: self.open_date_picker("from"),
            col={"xs": 6, "md": 3, "xl": 2},
        )
        self.date_to_field = self._date_field(
            label="Hasta",
            value=self.date_max.strftime("%Y-%m-%d"),
            on_click=lambda _event: self.open_date_picker("to"),
            col={"xs": 6, "md": 3, "xl": 2},
        )
        self.date_range_text = self._title_text("", size=18)
        self.date_range_hint_text = self._muted_text()
        start_month_index = self._month_index(self.default_date_from)
        end_month_index = self._month_index(self.date_max)
        self.date_range_slider = ft.RangeSlider(
            start_value=start_month_index,
            end_value=end_month_index,
            min=0,
            max=max(self.month_count - 1, 0),
            divisions=max(self.month_count - 1, 1),
            round=0,
            active_color=PALETTE.primary,
            inactive_color=PALETTE.line,
            on_change=self._on_month_range_change,
            on_change_end=self._on_month_range_change_end,
            expand=True,
        )
        self.amount_min_field = self._text_field(
            label="Importe minimo",
            value=f"{self.amount_min:.2f}",
            col={"xs": 6, "md": 3, "xl": 1.5},
        )
        self.amount_max_field = self._text_field(
            label="Importe maximo",
            value=f"{self.amount_max:.2f}",
            col={"xs": 6, "md": 3, "xl": 1.5},
        )
        self.time_display_dropdown = ft.Dropdown(
            label="Mostrar gasto",
            value="total",
            options=[
                ft.dropdown.Option("total", "Total del rango"),
                ft.dropdown.Option("daily", "Diario medio"),
                ft.dropdown.Option("monthly_average", "Mensual medio"),
                ft.dropdown.Option("monthly_selected", "Mes concreto"),
                ft.dropdown.Option("seasonal", "Estacional medio"),
                ft.dropdown.Option("yearly", "Anual medio"),
            ],
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            col={"xs": 12, "md": 6, "xl": 2},
        )
        self.time_display_dropdown.on_select = self._refresh_from_event
        self.visual_month_dropdown = ft.Dropdown(
            label="Mes visualizado",
            value=None,
            options=[],
            visible=False,
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            col={"xs": 12, "md": 6, "xl": 2},
        )
        self.visual_month_dropdown.on_select = self._refresh_from_event

        self.category_mode_checkbox = ft.Checkbox(
            label="Priorizar vista taxonomica",
            value=self.has_saved_categories,
            active_color=PALETTE.primary,
        )
        self.pending_only_checkbox = ft.Checkbox(
            label="Solo pendientes IA",
            value=False,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.review_only_checkbox = ft.Checkbox(
            label="Solo cola de revision",
            value=False,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.scope_filter_checkbox = ft.Checkbox(
            label="Filtrar tabla al nodo IA seleccionado",
            value=False,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.review_threshold_field = self._text_field(
            label="Umbral revision",
            value=f"{DEFAULT_REVIEW_CONFIDENCE:.2f}",
            width=160,
        )
        self.ai_question_field = ft.TextField(
            label="Instruccion para DeepSeek",
            value=(
                "Analiza mi economia: recurrentes mensuales, suscripciones, gastos "
                "frecuentes, Bizums que compensan gastos compartidos y categorias "
                "que deberia corregir."
            ),
            hint_text="ej. donde puedo recortar este mes sin contar ahorro o inversion como gasto?",
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            text_size=14,
            height=56,
            multiline=False,
        )
        self.ai_question_field.on_submit = lambda _event: self.page.run_task(
            self.ask_ai_about_current_view
        )
        self.ai_clarification_field = ft.TextField(
            label="Responder dudas o etiquetar movimientos concretos",
            value="",
            hint_text=(
                "Ej. El Bizum de Ana compensa la cena del 2026-03-10; "
                "Netflix es Suscripciones > Streaming."
            ),
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            text_size=13,
            multiline=True,
            min_lines=2,
            max_lines=4,
        )
        self.ai_scope_dropdown = ft.Dropdown(
            label="Aplicar sobre",
            value="filtered_view",
            options=[
                ft.dropdown.Option("filtered_view", "Vista filtrada"),
                ft.dropdown.Option("selection", "Seleccion actual"),
            ],
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            width=200,
        )
        self.ai_scope_dropdown.on_select = lambda _event: self._save_ui_state()
        self.chart_type_dropdown = ft.Dropdown(
            label="Grafica",
            value="sectors",
            options=[
                ft.dropdown.Option("sectors", "Sectores"),
                ft.dropdown.Option("categories", "Categorias"),
                ft.dropdown.Option("timeline", "Mes a mes"),
                ft.dropdown.Option("income", "Entradas y gastos"),
                ft.dropdown.Option("cashflow", "Naturaleza"),
                ft.dropdown.Option("opportunities", "Ahorro"),
                ft.dropdown.Option("heatmap", "Mapa de calor"),
            ],
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
        )
        self.chart_type_dropdown.on_select = self._refresh_from_event
        self.vision_section_dropdown = ft.Dropdown(
            label="Seccion de Vision",
            value=self.current_vision_section,
            options=[
                ft.dropdown.Option("overview", "Vista general"),
                ft.dropdown.Option("analysis", "Graficas y categorias"),
                ft.dropdown.Option("movements", "Movimientos"),
            ],
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            width=280,
        )
        self.vision_section_dropdown.on_select = self.select_vision_section
        self.timeline_show_accumulated_saving_checkbox = ft.Checkbox(
            label="Ahorro acumulado",
            value=False,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.timeline_show_accumulated_investment_checkbox = ft.Checkbox(
            label="Inversion acumulada",
            value=False,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.timeline_show_expense_checkbox = ft.Checkbox(
            label="Gasto",
            value=True,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.timeline_show_salary_checkbox = ft.Checkbox(
            label="Nomina",
            value=True,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.timeline_show_balance_checkbox = ft.Checkbox(
            label="Balance",
            value=True,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.timeline_show_accumulated_expense_checkbox = ft.Checkbox(
            label="Gasto acumulado",
            value=True,
            active_color=PALETTE.primary,
            on_change=self._refresh_from_event,
        )
        self.confirmed_context_field = ft.TextField(
            label="Anadir memoria confirmada",
            value="",
            hint_text=(
                "Ej. Viaje a Lisboa 2026-03-14 a 2026-03-17; "
                "cena cumple el 2026-04-20; alquiler compartido..."
            ),
            multiline=True,
            min_lines=4,
            max_lines=8,
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            text_size=13,
        )
        self.memory_entries_text = ft.Markdown(
            value=format_memory_entries_for_display(self.memory_entries),
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            auto_follow_links=False,
        )

        self.rules_field = ft.TextField(
            value=RULES_PLACEHOLDER,
            multiline=True,
            min_lines=18,
            max_lines=28,
            expand=True,
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=20,
            border_color=PALETTE.line,
            text_style=ft.TextStyle(size=14, font_family="Consolas"),
        )

        self.dataset_meta_text = self._muted_text()
        self.dataset_meta_text.color = "#A8C1C5"
        self.scope_badge_text = self._badge_text()
        self.selection_badge_text = self._badge_text()
        self.status_text = self._muted_text(value="Panel listo")
        self.rules_count_text = self._muted_text(value="0 reglas activas")
        self.summary_heading_text = self._title_text("Resumen por concepto")
        self.detail_heading_text = self._title_text("Movimientos filtrados")
        self.context_text = self._muted_text()
        self.selection_context_text = self._muted_text(
            value="Mostrando todos los movimientos filtrados"
        )
        self.taxonomy_meta_text = self._muted_text(value="Arbol listo")
        self.taxonomy_scope_text = self._title_text("Mapa economico: Root", size=18)
        self.taxonomy_node_summary_text = self._muted_text(
            value="Selecciona una categoria para ver su peso economico."
        )
        self.taxonomy_node_values = ft.ListView(spacing=10, expand=True, padding=0)
        self.taxonomy_search_field = ft.TextField(
            label="Buscar categoria",
            value="",
            hint_text="ej. supermercado, ahorro, vivienda",
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            text_size=13,
            prefix_icon=ft.Icons.SEARCH,
            on_change=self._render_taxonomy_from_event,
        )
        self.selection_preview_text = self._muted_text(
            value="Sin filas seleccionadas para revision manual."
        )
        self.health_heading_text = self._title_text("Salud mensual", size=20)
        self.health_context_text = self._muted_text(
            value="Ultimo mes con datos comparado contra la media reciente."
        )
        self.health_alert_text = self._muted_text(value="Sin alerta calculada")

        self.metric_visible_value = self._metric_value()
        self.metric_visible_hint = self._muted_text()
        self.metric_review_value = self._metric_value()
        self.metric_review_hint = self._muted_text()
        self.metric_spend_value = self._metric_value()
        self.metric_spend_hint = self._muted_text()
        self.metric_balance_value = self._metric_value()
        self.metric_balance_hint = self._muted_text()
        self.metric_salary_value = self._metric_value()
        self.metric_salary_hint = self._muted_text()
        self.metric_bizum_value = self._metric_value()
        self.metric_bizum_hint = self._muted_text()

        self.progress_bar = ft.ProgressBar(
            visible=False,
            color=PALETTE.primary,
            bgcolor=PALETTE.surface_alt,
            bar_height=6,
            border_radius=10,
        )

        self.summary_list = ft.ListView(spacing=10, expand=True, padding=0)
        self.detail_list = ft.ListView(spacing=10, expand=True, padding=0)
        self.spend_breakdown_list = ft.ListView(spacing=10, expand=True, padding=0)
        self.monthly_flow_list = ft.ListView(spacing=10, expand=True, padding=0)
        self.savings_opportunity_list = ft.ListView(spacing=10, padding=0)
        self.taxonomy_list = ft.ListView(spacing=10, expand=True, padding=0)
        self.selection_preview_column = ft.Column(spacing=10, tight=True)
        self.ai_audit_metrics_text = self._muted_text(value="Auditoria IA pendiente")
        self.ai_trace_text = self._muted_text(
            value="Selecciona una fila para ver la traza IA."
        )
        self.ai_answer_text = ft.Markdown(
            value=(
                "Analiza o recategoriza la vista filtrada o la seleccion usando "
                "contexto compacto y movimientos visibles."
            ),
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK,
            auto_follow_links=False,
        )
        self.chart_heading_text = self._title_text("Sectores principales", size=18)
        self.chart_help_text = self._muted_text(
            value="Gasto por sector segun los filtros actuales."
        )
        self.chart_zoom_text = self._muted_text(value="Zoom X 1,0x | Y 1,0x")
        self.chart_container = ft.Container(content=ft.Container(), height=210, expand=True)
        self.ai_review_queue = ft.ListView(spacing=10, expand=True, padding=0)

        self.refresh_button = ft.OutlinedButton(
            "Actualizar",
            icon=ft.Icons.REFRESH,
            on_click=self._refresh_from_event,
        )
        self.export_button = ft.OutlinedButton(
            "Exportar resumen",
            icon=ft.Icons.DOWNLOAD,
            on_click=self.export_summary,
        )
        self.save_rules_button = ft.OutlinedButton(
            "Guardar reglas",
            icon=ft.Icons.SAVE_OUTLINED,
            on_click=self.save_rules_to_disk,
        )
        self.apply_rules_button = ft.FilledTonalButton(
            "Aplicar reglas",
            icon=ft.Icons.AUTO_FIX_HIGH_OUTLINED,
            on_click=self.apply_rules_to_master_dataframe,
        )
        self.rules_save_secondary_button = ft.OutlinedButton(
            "Guardar reglas",
            icon=ft.Icons.SAVE_OUTLINED,
            on_click=self.save_rules_to_disk,
        )
        self.rules_apply_secondary_button = ft.FilledTonalButton(
            "Aplicar reglas",
            icon=ft.Icons.AUTO_FIX_HIGH_OUTLINED,
            on_click=self.apply_rules_to_master_dataframe,
        )
        self.clear_cache_button = ft.OutlinedButton(
            "Limpiar cache",
            icon=ft.Icons.CLEANING_SERVICES_OUTLINED,
            on_click=self.clear_deepseek_cache,
        )
        self.ask_ai_button = ft.FilledButton(
            "Analizar",
            icon=ft.Icons.PSYCHOLOGY_ALT_OUTLINED,
            on_click=lambda _event: self.page.run_task(self.ask_ai_about_current_view),
        )
        self.recategorize_ai_button = ft.FilledTonalButton(
            "Recategorizar",
            icon=ft.Icons.AUTO_FIX_HIGH_OUTLINED,
            on_click=lambda _event: self.page.run_task(self.recategorize_with_ai_guidance),
        )
        self.apply_ai_clarification_button = ft.FilledTonalButton(
            "Aplicar respuesta",
            icon=ft.Icons.TASK_ALT_OUTLINED,
            on_click=lambda _event: self.page.run_task(self.apply_ai_clarification),
        )
        self.save_context_button = ft.FilledTonalButton(
            "Guardar memoria",
            icon=ft.Icons.LIBRARY_ADD_CHECK_OUTLINED,
            on_click=self.save_confirmed_context,
        )
        self.ai_full_button = ft.FilledButton(
            "IA punta a punta",
            icon=ft.Icons.PSYCHOLOGY_ALT_OUTLINED,
            on_click=lambda _event: self.page.run_task(self.run_tree_categorization, False),
        )
        self.ai_node_button = ft.FilledButton(
            "IA desde nodo",
            icon=ft.Icons.ACCOUNT_TREE_OUTLINED,
            on_click=lambda _event: self.page.run_task(self.run_tree_categorization, True),
        )
        self.taxonomy_ai_node_button = ft.FilledButton(
            "IA desde nodo",
            icon=ft.Icons.ACCOUNT_TREE_OUTLINED,
            on_click=lambda _event: self.page.run_task(self.run_tree_categorization, True),
        )
        self.assign_button = ft.FilledTonalButton(
            "Asignar seleccion",
            icon=ft.Icons.PLAYLIST_ADD_CHECK_CIRCLE_OUTLINED,
            on_click=self.assign_selected_rows_to_node,
        )
        self.taxonomy_assign_button = ft.FilledTonalButton(
            "Asignar seleccion",
            icon=ft.Icons.PLAYLIST_ADD_CHECK_CIRCLE_OUTLINED,
            on_click=self.assign_selected_rows_to_node,
        )
        self.accept_ai_button = ft.FilledTonalButton(
            "Aceptar IA",
            icon=ft.Icons.CHECK_CIRCLE_OUTLINE,
            on_click=self.accept_selected_ai_rows,
        )
        self.manual_review_button = ft.OutlinedButton(
            "Enviar a revision manual",
            icon=ft.Icons.RATE_REVIEW_OUTLINED,
            on_click=self.send_selected_rows_to_manual_review,
        )
        self.clear_focus_button = ft.TextButton(
            "Ver todo",
            on_click=self.clear_group_focus,
        )
        self.vision_tab_button = ft.TextButton(
            "Vision",
            icon=ft.Icons.DASHBOARD_OUTLINED,
            on_click=lambda _event: self.select_tab("vision"),
        )
        self.rules_tab_button = ft.TextButton(
            "Reglas",
            icon=ft.Icons.RULE_OUTLINED,
            on_click=lambda _event: self.select_tab("rules"),
        )
        self.tree_tab_button = ft.TextButton(
            "Economia",
            icon=ft.Icons.SCHEMA_OUTLINED,
            on_click=lambda _event: self.select_tab("tree"),
        )
        self.select_visible_button = ft.TextButton(
            "Seleccionar visibles",
            on_click=self.select_visible_rows,
        )
        self.clear_selection_button = ft.TextButton(
            "Limpiar seleccion",
            on_click=self.clear_selected_rows,
        )
        self.scope_root_button = ft.OutlinedButton(
            "Volver a Root",
            icon=ft.Icons.REPLY_ALL_OUTLINED,
            on_click=self.set_taxonomy_scope_root,
        )
        self.taxonomy_root_button = ft.OutlinedButton(
            "Volver a Root",
            icon=ft.Icons.REPLY_ALL_OUTLINED,
            on_click=self.set_taxonomy_scope_root,
        )
        self.taxonomy_expand_all_button = ft.OutlinedButton(
            "Expandir todo",
            icon=ft.Icons.UNFOLD_MORE,
            on_click=self.expand_taxonomy_tree,
        )
        self.taxonomy_collapse_button = ft.OutlinedButton(
            "Plegar",
            icon=ft.Icons.UNFOLD_LESS,
            on_click=self.collapse_taxonomy_tree,
        )
        self.quick_30_button = ft.TextButton(
            "30 dias",
            on_click=lambda _event: self.select_date_preset(30),
        )
        self.quick_90_button = ft.TextButton(
            "90 dias",
            on_click=lambda _event: self.select_date_preset(90),
        )
        self.quick_all_button = ft.TextButton(
            "Todo",
            on_click=lambda _event: self.select_date_preset(None),
        )
        self.advanced_filters_visible = False
        self.advanced_filters_toggle_button = ft.TextButton(
            "Filtros avanzados",
            icon=ft.Icons.TUNE,
            on_click=self.toggle_advanced_filters,
        )
        self.apply_filters_button = ft.FilledButton(
            "Aplicar filtros",
            icon=ft.Icons.FILTER_ALT_OUTLINED,
            on_click=self._refresh_from_event,
        )
        self.reset_filters_button = ft.OutlinedButton(
            "Resetear",
            icon=ft.Icons.RESTART_ALT_OUTLINED,
            on_click=self.reset_filters,
        )
        self.chart_zoom_x_in_button = ft.IconButton(
            icon=ft.Icons.ZOOM_IN,
            tooltip="Zoom X: ver menos meses con mas detalle",
            on_click=lambda _event: self.adjust_chart_zoom(axis="x", factor=1.5),
        )
        self.chart_zoom_x_out_button = ft.IconButton(
            icon=ft.Icons.ZOOM_OUT,
            tooltip="Alejar X: ver mas meses",
            on_click=lambda _event: self.adjust_chart_zoom(axis="x", factor=1 / 1.5),
        )
        self.chart_zoom_y_in_button = ft.IconButton(
            icon=ft.Icons.VERTICAL_ALIGN_CENTER,
            tooltip="Zoom Y: ampliar escala vertical",
            on_click=lambda _event: self.adjust_chart_zoom(axis="y", factor=1.5),
        )
        self.chart_zoom_y_out_button = ft.IconButton(
            icon=ft.Icons.VERTICAL_ALIGN_BOTTOM,
            tooltip="Alejar Y: reducir escala vertical",
            on_click=lambda _event: self.adjust_chart_zoom(axis="y", factor=1 / 1.5),
        )
        self.chart_zoom_global_in_button = ft.IconButton(
            icon=ft.Icons.OPEN_IN_FULL,
            tooltip="Zoom global en X e Y",
            on_click=lambda _event: self.adjust_chart_zoom(axis="global", factor=1.5),
        )
        self.chart_zoom_reset_button = ft.IconButton(
            icon=ft.Icons.RESTART_ALT,
            tooltip="Resetear zoom de la grafica",
            on_click=self.reset_chart_zoom,
        )

        self.busy_controls = [
            self.refresh_button,
            self.export_button,
            self.save_rules_button,
            self.apply_rules_button,
            self.rules_save_secondary_button,
            self.rules_apply_secondary_button,
            self.clear_cache_button,
            self.ask_ai_button,
            self.recategorize_ai_button,
            self.apply_ai_clarification_button,
            self.save_context_button,
            self.ai_full_button,
            self.ai_node_button,
            self.assign_button,
            self.taxonomy_ai_node_button,
            self.taxonomy_assign_button,
            self.accept_ai_button,
            self.manual_review_button,
            self.apply_filters_button,
            self.reset_filters_button,
            self.chart_zoom_x_in_button,
            self.chart_zoom_x_out_button,
            self.chart_zoom_y_in_button,
            self.chart_zoom_y_out_button,
            self.chart_zoom_global_in_button,
            self.chart_zoom_reset_button,
            self.scope_root_button,
            self.taxonomy_root_button,
            self.advanced_filters_toggle_button,
            self.vision_section_dropdown,
        ]

        self._apply_control_tooltips()

        self.vision_tab = self._build_vision_tab()
        self.rules_tab = self._build_rules_tab()
        self.tree_tab = self._build_taxonomy_tab()
        self.advanced_filters_panel = ft.Container(visible=False)
        self.tab_content = ft.Container()
        self._sync_vision_section_visibility()
        self._sync_tab_controls()
        self._sync_month_slider_from_fields()

        self.shell = ft.Column(
            controls=[self._build_workspace_shell()],
            spacing=0,
            expand=True,
        )

    def _build_workspace_shell(self) -> ft.Control:
        return ft.Row(
            controls=[
                self._build_sidebar(),
                ft.Container(
                    content=ft.ListView(
                        controls=[
                            self.progress_bar,
                            self.tab_content,
                        ],
                        spacing=14,
                        expand=True,
                        padding=ft.padding.only(bottom=18),
                    ),
                    expand=True,
                ),
                self._build_ai_rail(),
            ],
            spacing=18,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    def _build_bottom_date_bar(self) -> ft.Control:
        self.advanced_filters_panel.content = self._build_advanced_filters_content()
        return self._surface_card(
            ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(
                                ft.Icons.DATE_RANGE_OUTLINED,
                                color=PALETTE.primary,
                                size=22,
                            ),
                            ft.Column(
                                controls=[
                                    self.date_range_text,
                                    self.date_range_hint_text,
                                ],
                                spacing=2,
                                expand=True,
                            ),
                            ft.Row(
                                controls=[
                                    self.quick_30_button,
                                    self.quick_90_button,
                                    self.quick_all_button,
                                    self.advanced_filters_toggle_button,
                                ],
                                spacing=4,
                                wrap=True,
                            ),
                        ],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.date_range_slider,
                    self.advanced_filters_panel,
                ],
                spacing=8,
            ),
            padding=14,
            bgcolor=PALETTE.panel_alt,
        )

    def _build_advanced_filters_content(self) -> ft.Control:
        return ft.Container(
            padding=12,
            border_radius=16,
            bgcolor=PALETTE.surface,
            border=ft.border.all(1, PALETTE.line),
            content=ft.Column(
                controls=[
                    ft.ResponsiveRow(
                        controls=[
                            self.group_dropdown,
                            self.pattern_field,
                            self.date_from_field,
                            self.date_to_field,
                            self.amount_min_field,
                            self.amount_max_field,
                            self.time_display_dropdown,
                            self.visual_month_dropdown,
                        ]
                    ),
                    ft.Row(
                        spacing=18,
                        wrap=True,
                        controls=[
                            self.category_mode_checkbox,
                            self.pending_only_checkbox,
                            self.review_only_checkbox,
                            self.scope_filter_checkbox,
                        ],
                    ),
                    ft.Row(
                        controls=[self.apply_filters_button, self.reset_filters_button],
                        spacing=10,
                        wrap=True,
                    ),
                ],
                spacing=10,
            ),
        )

    def _build_sidebar(self) -> ft.Control:
        return ft.Container(
            width=300,
            content=ft.Column(
                controls=[
                    self._build_sidebar_header(),
                    self._build_sidebar_navigation(),
                    self._build_sidebar_actions(),
                    self._build_memory_card(compact=True),
                    self._build_footer(compact=True),
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

    def _build_ai_rail(self) -> ft.Control:
        return ft.Container(
            width=380,
            content=ft.Column(
                controls=[self._build_ai_advisor_card(compact=True)],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

    def _build_sidebar_header(self) -> ft.Control:
        return self._surface_card(
            ft.Column(
                controls=[
                    ft.Text(
                        "Expenses",
                        size=24,
                        weight=ft.FontWeight.W_700,
                        color="#FFFFFF",
                        font_family="Bahnschrift",
                    ),
                    self.dataset_meta_text,
                    ft.Row(
                        controls=[
                            self._hero_badge("Scope", self.scope_badge_text),
                            self._hero_badge("Seleccion", self.selection_badge_text),
                        ],
                        wrap=True,
                    ),
                ],
                spacing=10,
            ),
            padding=16,
            bgcolor=PALETTE.panel_alt,
        )

    def _build_sidebar_navigation(self) -> ft.Control:
        return self._surface_card(
            ft.Column(
                controls=[
                    self.vision_tab_button,
                    self.rules_tab_button,
                    self.tree_tab_button,
                ],
                spacing=6,
            ),
            padding=10,
            bgcolor=PALETTE.surface_alt,
        )

    def _build_sidebar_actions(self) -> ft.Control:
        return self._surface_card(
            ft.Column(
                controls=[
                    ft.Row(
                        controls=[self.refresh_button, self.export_button],
                        spacing=8,
                        wrap=True,
                    ),
                    ft.Row(
                        controls=[self.save_rules_button, self.apply_rules_button],
                        spacing=8,
                        wrap=True,
                    ),
                    ft.Row(
                        controls=[self.ai_full_button, self.ai_node_button],
                        spacing=8,
                        wrap=True,
                    ),
                    self.clear_cache_button,
                ],
                spacing=8,
            ),
            padding=12,
        )

    def _build_hero(self) -> ft.Control:
        return ft.Container(
            padding=32,
            border_radius=28,
            gradient=ft.LinearGradient(
                colors=[PALETTE.hero_start, PALETTE.hero_end],
                begin=ft.Alignment(-1, -1),
                end=ft.Alignment(1, 1),
            ),
            shadow=ft.BoxShadow(
                spread_radius=2,
                blur_radius=15,
                color=ft.colors.with_opacity(0.4, PALETTE.hero_start),
                offset=ft.Offset(0, 4)
            ),
            content=ft.ResponsiveRow(
                controls=[
                    ft.Container(
                        col={"xs": 12, "xl": 8},
                        content=ft.Row(
                            controls=[
                                ft.Container(
                                    content=ft.Icon(ft.Icons.AUTO_AWESOME, color=PALETTE.ai_accent, size=40),
                                    padding=16,
                                    bgcolor=ft.colors.with_opacity(0.15, "#FFFFFF"),
                                    border_radius=16,
                                ),
                                ft.Column(
                                    controls=[
                                        ft.Text(
                                            "Expenses Studio Pro",
                                            size=34,
                                            weight=ft.FontWeight.W_800,
                                            color="#FFFFFF",
                                            font_family="Outfit",
                                        ),
                                        ft.Text(
                                            "Workspace local avanzado para inteligencia financiera y analisis.",
                                            size=15,
                                            color="#E2E8F0",
                                        ),
                                        self.dataset_meta_text,
                                    ],
                                    spacing=4,
                                    tight=True,
                                ),
                            ],
                            spacing=20,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                    ft.Container(
                        col={"xs": 12, "xl": 4},
                        content=ft.Column(
                            controls=[
                                self._hero_badge("Scope activo", self.scope_badge_text),
                                self._hero_badge(
                                    "Seleccion actual",
                                    self.selection_badge_text,
                                ),
                            ],
                            spacing=12,
                            horizontal_alignment=ft.CrossAxisAlignment.END,
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _build_toolbar(self) -> ft.Control:
        return self._surface_card(
            ft.ResponsiveRow(
                controls=[
                    ft.Container(
                        col={"xs": 12, "xl": 8},
                        content=ft.Row(
                            spacing=10,
                            wrap=True,
                            controls=[
                                self.refresh_button,
                                self.export_button,
                                self.save_rules_button,
                                self.apply_rules_button,
                                self.clear_cache_button,
                            ],
                        ),
                    ),
                    ft.Container(
                        col={"xs": 12, "xl": 4},
                        content=ft.Row(
                            spacing=10,
                            wrap=True,
                            alignment=ft.MainAxisAlignment.END,
                            controls=[self.ai_full_button, self.ai_node_button],
                        ),
                    ),
                ]
            ),
            padding=18,
        )

    def _build_vision_tab(self) -> ft.Control:
        self.vision_overview_section = ft.Container(
            content=self._build_vision_overview_section(),
        )
        self.vision_analysis_section = ft.Container(
            content=self._build_vision_analysis_section(),
        )
        self.vision_movements_section = ft.Container(
            content=self._build_vision_movements_section(),
        )
        return ft.Column(
            controls=[
                self._build_vision_navigation_card(),
                self.vision_overview_section,
                self.vision_analysis_section,
                self.vision_movements_section,
            ],
            spacing=14,
        )

    def _build_vision_navigation_card(self) -> ft.Control:
        return self._surface_card(
            ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            self._title_text("Vision", size=20),
                            ft.Text(
                                "Elige una seccion y evita recorrer toda la pagina.",
                                size=13,
                                color=PALETTE.muted,
                            ),
                        ],
                        spacing=4,
                        expand=True,
                    ),
                    self.vision_section_dropdown,
                ],
                spacing=12,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=16,
            bgcolor=PALETTE.surface_alt,
        )

    def _build_vision_overview_section(self) -> ft.Control:
        return ft.Column(
            controls=[
                self._build_health_card(),
                self._build_metrics_grid(),
                self._surface_card(
                    ft.Row(
                        controls=[
                            self.context_text,
                            self.selection_context_text,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        wrap=True,
                    ),
                    padding=16,
                    bgcolor=PALETTE.surface_alt,
                ),
            ],
            spacing=14,
        )

    def _build_metrics_grid(self) -> ft.Control:
        return ft.ResponsiveRow(
            controls=[
                self._metric_card(
                    "Ingresos",
                    self.metric_visible_value,
                    self.metric_visible_hint,
                    PALETTE.primary,
                ),
                self._metric_card(
                    "Gasto",
                    self.metric_review_value,
                    self.metric_review_hint,
                    PALETTE.warning,
                ),
                self._metric_card(
                    "Margen",
                    self.metric_spend_value,
                    self.metric_spend_hint,
                    PALETTE.success,
                ),
                self._metric_card(
                    "Alerta",
                    self.metric_balance_value,
                    self.metric_balance_hint,
                    PALETTE.danger,
                ),
                self._metric_card(
                    "Movimientos",
                    self.metric_salary_value,
                    self.metric_salary_hint,
                    PALETTE.accent,
                ),
                self._metric_card(
                    "Bizums",
                    self.metric_bizum_value,
                    self.metric_bizum_hint,
                    "#7C8DFF",
                ),
            ]
        )

    def _build_vision_analysis_section(self) -> ft.Control:
        return ft.ResponsiveRow(
            controls=[
                ft.Container(
                    col={"xs": 12, "xl": 7},
                    content=self._build_spend_breakdown_card(),
                ),
                ft.Container(
                    col={"xs": 12, "xl": 5},
                    content=self._build_summary_card(),
                ),
            ]
        )

    def _build_vision_movements_section(self) -> ft.Control:
        return ft.Column(
            controls=[
                ft.ResponsiveRow(
                    controls=[
                        ft.Container(
                            col={"xs": 12},
                            content=self._build_detail_card(),
                        ),
                    ]
                ),
            ],
            spacing=14,
        )

    def _build_health_card(self) -> ft.Control:
        return self._surface_card(
            ft.Container(
                padding=24,
                border_radius=18,
                gradient=ft.LinearGradient(
                    colors=[ft.colors.with_opacity(0.1, PALETTE.primary), ft.colors.with_opacity(0.02, PALETTE.surface)],
                    begin=ft.Alignment(-1, -1), end=ft.Alignment(1, 1)
                ),
                content=ft.ResponsiveRow(
                    controls=[
                        ft.Container(
                            col={"xs": 12, "xl": 4},
                            content=ft.Row(
                                controls=[
                                    ft.Container(
                                        content=ft.Icon(ft.Icons.MONITOR_HEART_OUTLINED, color=PALETTE.primary, size=36),
                                        padding=12, bgcolor=ft.colors.with_opacity(0.1, PALETTE.primary), border_radius=14
                                    ),
                                    ft.Column(
                                        controls=[
                                            ft.Text("Health Score", size=13, color=PALETTE.muted, weight=ft.FontWeight.BOLD),
                                            self.health_heading_text,
                                        ],
                                        spacing=2, tight=True,
                                    ),
                                ],
                                spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER
                            ),
                        ),
                        ft.Container(
                            col={"xs": 12, "xl": 8},
                            content=ft.Row(
                                controls=[
                                    ft.Container(width=2, height=40, bgcolor=ft.colors.with_opacity(0.2, PALETTE.line)),
                                    ft.Column(
                                        controls=[
                                            self.health_alert_text,
                                            self.health_context_text,
                                        ],
                                        spacing=4, expand=True,
                                    )
                                ],
                                spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER
                            ),
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ),
            padding=0,
            bgcolor=PALETTE.surface_alt,
        )

    def _build_spend_breakdown_card(self) -> ft.Control:
        return self._surface_card(
            ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.chart_heading_text,
                            self._help_icon(
                                "Cambia el tipo de grafica sin tocar el rango filtrado. "
                                "Sectores compara categorias principales; Categorias baja al "
                                "grupo activo; Mes a mes compara nomina y gasto; Naturaleza "
                                "separa consumo, ahorro e inversion; Ahorro prioriza recortes."
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.chart_help_text,
                    self.chart_type_dropdown,
                    ft.Row(
                        controls=[
                            self._timeline_series_toggle(self.timeline_show_expense_checkbox),
                            self._timeline_series_toggle(self.timeline_show_salary_checkbox),
                            self._timeline_series_toggle(self.timeline_show_balance_checkbox),
                            self._timeline_series_toggle(
                                self.timeline_show_accumulated_expense_checkbox,
                                width=190,
                            ),
                            self._timeline_series_toggle(
                                self.timeline_show_accumulated_saving_checkbox,
                                width=195,
                            ),
                            self._timeline_series_toggle(
                                self.timeline_show_accumulated_investment_checkbox,
                                width=205,
                            ),
                        ],
                        spacing=4,
                        run_spacing=0,
                        wrap=True,
                    ),
                    ft.Row(
                        controls=[
                            ft.Text("X", size=12, color=PALETTE.muted),
                            self.chart_zoom_x_in_button,
                            self.chart_zoom_x_out_button,
                            ft.Text("Y", size=12, color=PALETTE.muted),
                            self.chart_zoom_y_in_button,
                            self.chart_zoom_y_out_button,
                            self.chart_zoom_global_in_button,
                            self.chart_zoom_reset_button,
                            self.chart_zoom_text,
                        ],
                        spacing=2,
                        wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Container(
                        content=self.chart_container,
                        height=220,
                        bgcolor=PALETTE.surface,
                        border_radius=16,
                        border=ft.border.all(1, PALETTE.line),
                        padding=8,
                    ),
                    ft.Divider(height=1, color=PALETTE.line),
                    self.spend_breakdown_list,
                ],
                spacing=12,
                expand=True,
            ),
            height=560,
        )

    def _build_monthly_flow_card(self) -> ft.Control:
        return self._surface_card(
            ft.Column(
                controls=[
                    self._card_header(
                        "Meses recientes",
                        "Detecta transferencias con concepto nomina y las compara por mes.",
                        help_text=(
                            "Usa importes positivos con texto nomina o nómina como entrada "
                            "principal. Los Bizums compartidos se muestran aparte porque no "
                            "son ingreso neto."
                        ),
                    ),
                    ft.Divider(height=1, color=PALETTE.line),
                    self.monthly_flow_list,
                ],
                spacing=12,
                expand=True,
            ),
            height=520,
        )

    def _timeline_series_toggle(
        self,
        checkbox: ft.Checkbox,
        *,
        width: int = 135,
    ) -> ft.Control:
        return ft.Container(
            width=width,
            height=34,
            content=checkbox,
        )

    def _build_ai_advisor_card(self, *, compact: bool = False) -> ft.Control:
        answer_height = 250 if compact else 360
        opportunities: ft.Control = self.savings_opportunity_list
        if compact:
            opportunities = ft.Container(
                content=self.savings_opportunity_list,
                height=120,
            )
        card_height = None
        return self._surface_card(
            ft.Container(
                padding=0,
                border_radius=12,
                content=ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Icon(ft.Icons.AUTO_AWESOME, color=PALETTE.ai_accent, size=24),
                                ft.Text("DeepSeek Copilot", size=18, weight=ft.FontWeight.BOLD, color=PALETTE.ai_accent),
                            ],
                            alignment=ft.MainAxisAlignment.START,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=8
                        ),
                        ft.Text(
                            "Analiza recurrencias, Bizums compartidos y categorias. Puedes pedirle que detecte suscripciones o recategorice movimientos con tus reglas.",
                            size=13, color=PALETTE.muted
                        ),
                        ft.Row(
                            controls=[
                                self.ai_scope_dropdown,
                                self.ask_ai_button,
                                self.recategorize_ai_button,
                            ],
                            spacing=10,
                            wrap=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        self.ai_question_field,
                        ft.Container(
                            content=ft.Column(
                                controls=[self.ai_answer_text],
                                spacing=0,
                                scroll=ft.ScrollMode.AUTO,
                                expand=True,
                            ),
                            padding=16,
                            border_radius=14,
                            bgcolor=ft.colors.with_opacity(0.3, PALETTE.surface),
                            border=ft.border.all(1, ft.colors.with_opacity(0.3, PALETTE.ai_accent)),
                            shadow=ft.BoxShadow(spread_radius=1, blur_radius=10, color=ft.colors.with_opacity(0.1, PALETTE.ai_accent)),
                            height=answer_height,
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    "Dudas y etiquetado",
                                    size=15,
                                    weight=ft.FontWeight.W_600,
                                    color=PALETTE.ink,
                                ),
                                self.ai_clarification_field,
                                ft.Row(
                                    controls=[self.apply_ai_clarification_button],
                                    alignment=ft.MainAxisAlignment.END,
                                ),
                            ],
                            spacing=8,
                        ),
                        ft.Divider(height=1, color=PALETTE.line),
                        ft.Text(
                            "Oportunidades IA",
                            size=15,
                            weight=ft.FontWeight.W_700,
                            color=PALETTE.ink,
                        ),
                        opportunities,
                    ],
                    spacing=14,
                    expand=not compact,
                )
            ), height=card_height
        )

    def _build_memory_card(self, *, compact: bool = False) -> ft.Control:
        if compact:
            return self._surface_card(
                ft.Column(
                    controls=[
                        self._card_header(
                            "Memoria",
                            "Contexto guardado para DeepSeek.",
                            help_text=(
                                "Anade hechos confirmados en lenguaje normal. "
                                "Se guardan con fuente usuario y DeepSeek los usa "
                                "como contexto estable."
                            ),
                        ),
                        self.confirmed_context_field,
                        self.save_context_button,
                        ft.Container(
                            content=ft.Column(
                                controls=[
                                    ft.Text(
                                        "Memoria guardada",
                                        size=13,
                                        weight=ft.FontWeight.W_600,
                                        color=PALETTE.ink,
                                    ),
                                    self.memory_entries_text,
                                ],
                                spacing=8,
                                scroll=ft.ScrollMode.AUTO,
                            ),
                            padding=10,
                            border_radius=12,
                            bgcolor=PALETTE.surface,
                            border=ft.border.all(1, PALETTE.line),
                            height=160,
                        ),
                    ],
                    spacing=12,
                ),
                padding=14,
                bgcolor=PALETTE.surface_alt,
            )

        return self._surface_card(
            ft.ResponsiveRow(
                controls=[
                    ft.Container(
                        col={"xs": 12, "xl": 4},
                        content=self._card_header(
                            "Memoria confirmada",
                            "Viajes, quedadas, fiestas o gastos ya explicados.",
                            help_text=(
                                "Anade hechos confirmados en lenguaje normal. "
                                "Cada entrada queda marcada como fuente usuario y "
                                "DeepSeek la usa como contexto estable al analizar "
                                "patrones, dudas y posibles categorias."
                            ),
                        ),
                    ),
                    ft.Container(
                        col={"xs": 12, "xl": 6},
                        content=self.confirmed_context_field,
                    ),
                    ft.Container(
                        col={"xs": 12, "xl": 2},
                        content=ft.Row(
                            controls=[self.save_context_button],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ),
                    ft.Container(
                        col={"xs": 12},
                        content=ft.Container(
                            content=ft.Column(
                                controls=[
                                    ft.Text(
                                        "Memoria guardada",
                                        size=14,
                                        weight=ft.FontWeight.W_600,
                                        color=PALETTE.ink,
                                    ),
                                    self.memory_entries_text,
                                ],
                                spacing=8,
                                scroll=ft.ScrollMode.AUTO,
                            ),
                            padding=12,
                            border_radius=14,
                            bgcolor=PALETTE.surface,
                            border=ft.border.all(1, PALETTE.line),
                            height=180,
                        ),
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=18,
            bgcolor=PALETTE.surface_alt,
        )

    def _build_filters_card(self, *, compact: bool = False) -> ft.Control:
        filter_controls: list[ft.Control]
        if compact:
            filter_controls = [
                self.group_dropdown,
                self.pattern_field,
                ft.Row(
                    controls=[self.date_from_field, self.date_to_field],
                    spacing=8,
                ),
                ft.Row(
                    controls=[self.amount_min_field, self.amount_max_field],
                    spacing=8,
                ),
                self.time_display_dropdown,
                self.visual_month_dropdown,
            ]
        else:
            filter_controls = [
                ft.ResponsiveRow(
                    controls=[
                        self.group_dropdown,
                        self.pattern_field,
                        self.date_from_field,
                        self.date_to_field,
                        self.amount_min_field,
                        self.amount_max_field,
                        self.time_display_dropdown,
                        self.visual_month_dropdown,
                    ]
                )
            ]

        return self._surface_card(
            ft.Column(
                controls=[
                    self._card_header(
                        "Filtros",
                        "Controles globales: afectan Vision, Economia, resumenes, tablas y acciones IA.",
                    ),
                    *filter_controls,
                    ft.Row(
                        controls=[
                            self.quick_30_button,
                            self.quick_90_button,
                            self.quick_all_button,
                        ],
                        wrap=True,
                    ),
                    ft.Text(
                        "Si eliges 30 dias, toda la aplicacion trabaja sobre esos 30 dias hasta cambiar el rango.",
                        size=12,
                        color=PALETTE.muted,
                    ),
                    ft.Row(
                        spacing=18,
                        wrap=True,
                        controls=[
                            self.category_mode_checkbox,
                            self.pending_only_checkbox,
                            self.review_only_checkbox,
                            self.scope_filter_checkbox,
                        ],
                    ),
                    ft.Row(
                        controls=[self.apply_filters_button, self.reset_filters_button],
                        spacing=10,
                        wrap=True,
                    ),
                ],
                spacing=14,
            )
        )

    def _build_summary_card(self) -> ft.Control:
        return self._surface_card(
            ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.summary_heading_text,
                            self.clear_focus_button,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Divider(height=1, color=PALETTE.line),
                    self.summary_list,
                ],
                spacing=12,
                expand=True,
            ),
            padding=20,
            height=560,
        )

    def _build_detail_card(self) -> ft.Control:
        return self._surface_card(
            ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.detail_heading_text,
                            ft.Row(
                                controls=[
                                    self.select_visible_button,
                                    self.clear_selection_button,
                                ],
                                spacing=0,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Text(
                        "Selecciona filas para revision manual o reasignacion desde Economia.",
                        size=13,
                        color=PALETTE.muted,
                    ),
                    ft.Divider(height=1, color=PALETTE.line),
                    self.detail_list,
                ],
                spacing=12,
                expand=True,
            ),
            padding=20,
            height=520,
        )

    def _build_rules_tab(self) -> ft.Control:
        return ft.Column(
            controls=[
                self._surface_card(
                    ft.Column(
                        controls=[
                            self._card_header(
                                "Reglas locales",
                                (
                                    "Se evaluan antes del recorrido IA. "
                                    "Usa una regla por linea con formato Nombre = regex."
                                ),
                            ),
                            self.rules_count_text,
                            self.rules_field,
                            ft.Row(
                                controls=[
                                    self.rules_save_secondary_button,
                                    self.rules_apply_secondary_button,
                                ],
                                spacing=10,
                                wrap=True,
                            ),
                        ],
                        spacing=14,
                    )
                )
            ],
            spacing=18,
        )

    def _build_taxonomy_tab(self) -> ft.Control:
        return ft.ResponsiveRow(
            controls=[
                ft.Container(
                    col={"xs": 12, "xl": 5},
                    content=self._surface_card(
                        ft.Column(
                            controls=[
                                self._card_header(
                                    "Mapa economico",
                                    (
                                        "Explora categorias por peso economico y usa "
                                        "IA solo cuando aporte revision."
                                    ),
                                ),
                                self.taxonomy_meta_text,
                                self.taxonomy_search_field,
                                ft.Row(
                                    controls=[
                                        self.taxonomy_expand_all_button,
                                        self.taxonomy_collapse_button,
                                    ],
                                    spacing=8,
                                    wrap=True,
                                ),
                                ft.Divider(height=1, color=PALETTE.line),
                                self.taxonomy_list,
                            ],
                            spacing=12,
                            expand=True,
                        ),
                        height=700,
                    ),
                ),
                ft.Container(
                    col={"xs": 12, "xl": 7},
                    content=self._surface_card(
                        ft.Column(
                            controls=[
                                self.taxonomy_scope_text,
                                self.taxonomy_node_summary_text,
                                ft.Container(
                                    content=self.taxonomy_node_values,
                                    height=170,
                                ),
                                ft.Divider(height=1, color=PALETTE.line),
                                self.ai_audit_metrics_text,
                                ft.Row(
                                    controls=[
                                        self.review_threshold_field,
                                        self.taxonomy_root_button,
                                    ],
                                    wrap=True,
                                ),
                                ft.Row(
                                    spacing=10,
                                    wrap=True,
                                    controls=[
                                        self.taxonomy_ai_node_button,
                                        self.taxonomy_assign_button,
                                        self.accept_ai_button,
                                        self.manual_review_button,
                                    ],
                                ),
                                ft.Divider(height=1, color=PALETTE.line),
                                ft.Text(
                                    "Cola priorizada",
                                    size=16,
                                    weight=ft.FontWeight.W_600,
                                    color=PALETTE.ink,
                                ),
                                ft.Container(
                                    content=self.ai_review_queue,
                                    height=230,
                                ),
                                self.ai_trace_text,
                                ft.Divider(height=1, color=PALETTE.line),
                                ft.Text(
                                    "Seleccion actual",
                                    size=16,
                                    weight=ft.FontWeight.W_600,
                                    color=PALETTE.ink,
                                ),
                                self.selection_preview_text,
                                self.selection_preview_column,
                            ],
                            spacing=14,
                            expand=True,
                        ),
                        height=700,
                    ),
                ),
            ]
        )

    def _build_footer(self, *, compact: bool = False) -> ft.Control:
        if compact:
            return self._surface_card(
                ft.Column(
                    controls=[
                        self.status_text,
                        ft.Text(
                            "Panel lateral: filtros, IA y acciones. Centro: lectura esencial.",
                            size=12,
                            color=PALETTE.muted,
                        ),
                    ],
                    spacing=8,
                ),
                padding=14,
                bgcolor=PALETTE.panel_alt,
            )

        return self._surface_card(
            ft.Row(
                controls=[
                    self.status_text,
                    ft.Text(
                        (
                            "Acciones clave: actualizar, exportar resumen, "
                            "guardar reglas, IA completa o desde nodo."
                        ),
                        size=12,
                        color=PALETTE.muted,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                wrap=True,
            ),
            padding=16,
            bgcolor=PALETTE.panel_alt,
        )

    def _build_tab_shell(self) -> ft.Control:
        return ft.Column(
            controls=[
                self._surface_card(
                    ft.Row(
                        controls=[
                            self.vision_tab_button,
                            self.rules_tab_button,
                            self.tree_tab_button,
                        ],
                        spacing=8,
                        wrap=True,
                    ),
                    padding=10,
                    bgcolor=PALETTE.surface_alt,
                ),
                self.tab_content,
            ],
            spacing=14,
            expand=True,
        )

    def _text_field(
        self,
        *,
        label: str,
        value: str,
        hint_text: str | None = None,
        width: float | None = None,
        col: Any = None,
    ) -> ft.TextField:
        return ft.TextField(
            label=label,
            value=value,
            hint_text=hint_text,
            width=width,
            col=col,
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            text_size=14,
            on_submit=self._refresh_from_event,
        )

    def _date_field(
        self,
        *,
        label: str,
        value: str,
        on_click: Any,
        col: Any = None,
    ) -> ft.TextField:
        return ft.TextField(
            label=label,
            value=value,
            col=col,
            read_only=True,
            filled=True,
            fill_color=PALETTE.surface,
            border_radius=16,
            border_color=PALETTE.line,
            text_size=14,
            on_click=on_click,
            suffix_icon=ft.Icons.CALENDAR_MONTH_OUTLINED,
            always_call_on_tap=True,
        )

    def _muted_text(self, value: str = "") -> ft.Text:
        return ft.Text(value=value, size=13, color=PALETTE.muted)

    def _badge_text(self, value: str = "") -> ft.Text:
        return ft.Text(value=value, size=13, weight=ft.FontWeight.W_600, color="#F6FBFB")

    def _title_text(self, value: str, *, size: int = 20) -> ft.Text:
        return ft.Text(
            value=value,
            size=size,
            weight=ft.FontWeight.W_700,
            color=PALETTE.ink,
            font_family="Bahnschrift",
        )

    def _metric_value(self) -> ft.Text:
        return ft.Text(
            size=24,
            weight=ft.FontWeight.W_700,
            color=PALETTE.ink,
            font_family="Bahnschrift",
        )

    def _surface_card(
        self,
        content: ft.Control,
        *,
        padding: int = 20,
        bgcolor: str = PALETTE.panel,
        height: float | None = None,
    ) -> ft.Container:
        return ft.Container(
            content=content,
            padding=padding,
            bgcolor=bgcolor,
            border_radius=24,
            border=ft.border.all(1, ft.colors.with_opacity(0.5, PALETTE.line)),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=8,
                color=ft.colors.with_opacity(0.4, "#000000"),
                offset=ft.Offset(0, 2)
            ),
            height=height,
        )

    def _metric_card(
        self,
        title: str,
        value_text: ft.Text,
        hint_text: ft.Text,
        stripe: str,
    ) -> ft.Container:
        return ft.Container(
            col={"xs": 12, "md": 4, "xl": 2},
            content=self._surface_card(
                ft.Row(
                    controls=[
                        ft.Container(
                            width=6, 
                            border_radius=8, 
                            gradient=ft.LinearGradient(
                                colors=[stripe, ft.colors.with_opacity(0.3, stripe)],
                                begin=ft.Alignment(0, -1), end=ft.Alignment(0, 1)
                            )
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    title,
                                    size=12,
                                    color=PALETTE.muted,
                                    weight=ft.FontWeight.W_700,
                                ),
                                value_text,
                                hint_text,
                            ],
                            spacing=4,
                            expand=True,
                        ),
                    ],
                    spacing=16,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                padding=20,
            ),
        )

    def _card_header(
        self,
        title: str,
        subtitle: str,
        *,
        help_text: str | None = None,
    ) -> ft.Column:
        title_row_controls: list[ft.Control] = [
            ft.Text(
                title,
                size=22,
                weight=ft.FontWeight.W_700,
                color=PALETTE.ink,
                font_family="Bahnschrift",
            )
        ]
        if help_text:
            title_row_controls.append(self._help_icon(help_text))
        return ft.Column(
            controls=[
                ft.Row(
                    controls=title_row_controls,
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Text(subtitle, size=13, color=PALETTE.muted),
            ],
            spacing=4,
            tight=True,
        )

    def _help_icon(self, message: str) -> ft.Control:
        icon = ft.Icon(
            ft.Icons.HELP_OUTLINE,
            color=PALETTE.muted,
            size=18,
        )
        icon.tooltip = message
        return icon

    def _hero_badge(self, label: str, value_text: ft.Text) -> ft.Control:
        return ft.Container(
            bgcolor="#16343A",
            border_radius=18,
            padding=14,
            content=ft.Column(
                controls=[
                    ft.Text(label, size=12, color="#93C9C4"),
                    value_text,
                ],
                spacing=4,
                tight=True,
                horizontal_alignment=ft.CrossAxisAlignment.END,
            ),
        )

    def _sync_tab_controls(self) -> None:
        mapping = {
            "vision": self.vision_tab,
            "rules": self.rules_tab,
            "tree": self.tree_tab,
        }
        self.tab_content.content = mapping[self.current_tab]
        selected_style = {
            "color": PALETTE.panel,
            "bgcolor": PALETTE.primary,
        }
        idle_style = {
            "color": PALETTE.ink,
            "bgcolor": PALETTE.surface,
        }
        for key, button in (
            ("vision", self.vision_tab_button),
            ("rules", self.rules_tab_button),
            ("tree", self.tree_tab_button),
        ):
            style = selected_style if self.current_tab == key else idle_style
            button.style = ft.ButtonStyle(
                color=style["color"],
                bgcolor=style["bgcolor"],
                shape=ft.RoundedRectangleBorder(radius=16),
                padding=ft.padding.symmetric(horizontal=16, vertical=14),
            )

    def _sync_vision_section_visibility(self) -> None:
        if self.current_vision_section not in VISION_SECTION_OPTIONS:
            self.current_vision_section = "overview"
        self.vision_section_dropdown.value = self.current_vision_section
        section_controls = {
            "overview": getattr(self, "vision_overview_section", None),
            "analysis": getattr(self, "vision_analysis_section", None),
            "movements": getattr(self, "vision_movements_section", None),
        }
        for key, control in section_controls.items():
            if control is not None:
                control.visible = key == self.current_vision_section

    def select_vision_section(self, event: ft.Event[ft.Dropdown]) -> None:
        value = str(event.control.value or "overview")
        if value not in VISION_SECTION_OPTIONS:
            value = "overview"
        self.current_vision_section = value
        self._sync_vision_section_visibility()
        self._save_ui_state()
        self.page.update()

    def _sync_tree_with_dataframe(self, tree: dict[str, Any]) -> dict[str, Any]:
        known_leaves: list[str] = []
        for column in ("CategoriaLeaf", "Grupo"):
            if column not in self.df.columns:
                continue
            known_leaves.extend(
                str(value).strip()
                for value in self.df[column].dropna().tolist()
                if str(value).strip()
            )
        merged_tree, tree_changed = merge_missing_leaves(tree, known_leaves)
        if tree_changed:
            save_category_tree(merged_tree, tree_file=CATEGORY_TREE_FILE)
        return merged_tree

    def _load_memory_entries(self) -> list[dict[str, str]]:
        if not ANALYSIS_CONTEXT_FILE.exists():
            return []
        try:
            payload = json.loads(ANALYSIS_CONTEXT_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("No se pudo cargar memoria confirmada: %s", exc)
            return []
        return normalize_memory_entries(payload)

    def save_confirmed_context(self, _event: Any = None) -> None:
        new_context = (self.confirmed_context_field.value or "").strip()
        if new_context:
            self.memory_entries.append(
                {
                    "text": new_context,
                    "source": "usuario",
                    "origin": "formulario_memoria",
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
        self.confirmed_context = format_memory_entries_for_prompt(self.memory_entries)
        ANALYSIS_CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        ANALYSIS_CONTEXT_FILE.write_text(
            json.dumps(
                {
                    "confirmed_context": self.confirmed_context,
                    "entries": self.memory_entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.confirmed_context_field.value = ""
        self.memory_entries_text.value = format_memory_entries_for_display(
            self.memory_entries
        )
        self.refresh_view("Memoria confirmada guardada")

    def _load_ui_state(self) -> None:
        if not UI_STATE_FILE.exists():
            return
        try:
            payload = json.loads(UI_STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("No se pudo cargar el estado de UI: %s", exc)
            return

        state = normalize_ui_state(payload)
        group_by = state.get("group_by")
        if group_by in GROUP_OPTIONS:
            self.group_dropdown.value = str(group_by)
        self.pattern_field.value = str(state.get("pattern", self.pattern_field.value or ""))
        self.date_from_field.value = str(
            state.get("date_from", self.date_from_field.value or "")
        )
        self.date_to_field.value = str(state.get("date_to", self.date_to_field.value or ""))
        self.amount_min_field.value = str(
            state.get("amount_min", self.amount_min_field.value or "")
        )
        self.amount_max_field.value = str(
            state.get("amount_max", self.amount_max_field.value or "")
        )
        self.time_display_dropdown.value = str(
            state.get("time_display", self.time_display_dropdown.value or "total")
        )
        self.chart_type_dropdown.value = str(
            state.get("chart_type", self.chart_type_dropdown.value or "sectors")
        )
        self.ai_scope_dropdown.value = str(
            state.get("ai_scope", self.ai_scope_dropdown.value or "filtered_view")
        )
        self.current_tab = str(state.get("current_tab", self.current_tab))
        self.current_vision_section = str(
            state.get("vision_section", self.current_vision_section)
        )
        self.vision_section_dropdown.value = self.current_vision_section
        self.category_mode_checkbox.value = bool(
            state.get("category_mode", self.category_mode_checkbox.value)
        )
        self.pending_only_checkbox.value = bool(
            state.get("pending_only", self.pending_only_checkbox.value)
        )
        self.review_only_checkbox.value = bool(
            state.get("review_only", self.review_only_checkbox.value)
        )
        self.scope_filter_checkbox.value = bool(
            state.get("scope_filter", self.scope_filter_checkbox.value)
        )
        self.timeline_show_expense_checkbox.value = bool(
            state.get("timeline_show_expense", self.timeline_show_expense_checkbox.value)
        )
        self.timeline_show_salary_checkbox.value = bool(
            state.get("timeline_show_salary", self.timeline_show_salary_checkbox.value)
        )
        self.timeline_show_balance_checkbox.value = bool(
            state.get("timeline_show_balance", self.timeline_show_balance_checkbox.value)
        )
        self.timeline_show_accumulated_expense_checkbox.value = bool(
            state.get(
                "timeline_show_accumulated_expense",
                self.timeline_show_accumulated_expense_checkbox.value,
            )
        )
        self.timeline_show_accumulated_saving_checkbox.value = bool(
            state.get(
                "timeline_show_accumulated_saving",
                self.timeline_show_accumulated_saving_checkbox.value,
            )
        )
        self.timeline_show_accumulated_investment_checkbox.value = bool(
            state.get(
                "timeline_show_accumulated_investment",
                self.timeline_show_accumulated_investment_checkbox.value,
            )
        )
        self.selected_node_path = str(
            state.get("selected_node_path", self.selected_node_path)
        )
        self.selected_visual_month = (
            str(state["visual_month"]) if state.get("visual_month") else None
        )

        for field, picker in (
            (self.date_from_field, self.date_from_picker),
            (self.date_to_field, self.date_to_picker),
        ):
            parsed = parse_date(field.value or "")
            if parsed is not None:
                picker.value = parsed.date()

        self._sync_tab_controls()
        self._sync_vision_section_visibility()
        self._sync_month_slider_from_fields()

    def _save_ui_state(self) -> None:
        payload = {
            "group_by": self.group_dropdown.value or "Concepto",
            "pattern": self.pattern_field.value or "",
            "date_from": self.date_from_field.value or "",
            "date_to": self.date_to_field.value or "",
            "amount_min": self.amount_min_field.value or "",
            "amount_max": self.amount_max_field.value or "",
            "time_display": self.time_display_dropdown.value or "total",
            "visual_month": self.visual_month_dropdown.value,
            "chart_type": self.chart_type_dropdown.value or "sectors",
            "ai_scope": self.ai_scope_dropdown.value or "filtered_view",
            "current_tab": self.current_tab,
            "vision_section": self.current_vision_section,
            "category_mode": bool(self.category_mode_checkbox.value),
            "pending_only": bool(self.pending_only_checkbox.value),
            "review_only": bool(self.review_only_checkbox.value),
            "scope_filter": bool(self.scope_filter_checkbox.value),
            "timeline_show_expense": bool(self.timeline_show_expense_checkbox.value),
            "timeline_show_salary": bool(self.timeline_show_salary_checkbox.value),
            "timeline_show_balance": bool(self.timeline_show_balance_checkbox.value),
            "timeline_show_accumulated_expense": bool(
                self.timeline_show_accumulated_expense_checkbox.value
            ),
            "timeline_show_accumulated_saving": bool(
                self.timeline_show_accumulated_saving_checkbox.value
            ),
            "timeline_show_accumulated_investment": bool(
                self.timeline_show_accumulated_investment_checkbox.value
            ),
            "selected_node_path": self.selected_node_path,
        }
        UI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        UI_STATE_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def open_date_picker(self, boundary: str) -> None:
        picker = self.date_from_picker if boundary == "from" else self.date_to_picker
        current_value = self.date_from_field.value if boundary == "from" else self.date_to_field.value
        parsed = parse_date(current_value)
        if parsed is not None:
            picker.value = parsed.date()
        self.page.show_dialog(picker)

    def _on_date_from_change(self, event: ft.Event[ft.DatePicker]) -> None:
        value = event.control.value
        if value is None:
            return
        selected = pd.to_datetime(value)
        self.date_from_field.value = selected.strftime("%Y-%m-%d")
        self._sync_month_slider_from_fields()
        self.refresh_view("Fecha inicial global actualizada")

    def _on_date_to_change(self, event: ft.Event[ft.DatePicker]) -> None:
        value = event.control.value
        if value is None:
            return
        selected = pd.to_datetime(value)
        self.date_to_field.value = selected.strftime("%Y-%m-%d")
        self._sync_month_slider_from_fields()
        self.refresh_view("Fecha final global actualizada")

    def _month_index(self, value: pd.Timestamp) -> int:
        month = pd.to_datetime(value).to_period("M").to_timestamp()
        return int((month.year - self.month_min.year) * 12 + month.month - self.month_min.month)

    def _month_from_index(self, index: int) -> pd.Timestamp:
        bounded = max(0, min(int(index), self.month_count - 1))
        return self.month_min + pd.DateOffset(months=bounded)

    def _month_range_from_slider(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_index = int(round(float(self.date_range_slider.start_value or 0)))
        end_index = int(round(float(self.date_range_slider.end_value or start_index)))
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        start_month = self._month_from_index(start_index)
        end_month = self._month_from_index(end_index)
        start_date = max(start_month, self.date_min)
        end_date = min(end_month + pd.offsets.MonthEnd(0), self.date_max)
        return pd.Timestamp(start_date), pd.Timestamp(end_date)

    def _sync_month_slider_from_fields(self) -> None:
        date_from = parse_date(self.date_from_field.value or "") or self.default_date_from
        date_to = parse_date(self.date_to_field.value or "") or self.date_max
        start_index = self._month_index(date_from)
        end_index = self._month_index(date_to)
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        start_index = max(0, min(start_index, self.month_count - 1))
        end_index = max(0, min(end_index, self.month_count - 1))
        self.date_range_slider.start_value = start_index
        self.date_range_slider.end_value = end_index
        self._update_date_range_label()

    def _apply_month_slider_to_fields(self) -> None:
        start_date, end_date = self._month_range_from_slider()
        self.date_from_field.value = start_date.strftime("%Y-%m-%d")
        self.date_to_field.value = end_date.strftime("%Y-%m-%d")
        self.date_from_picker.value = start_date.date()
        self.date_to_picker.value = end_date.date()
        self._update_date_range_label()

    def _update_date_range_label(self) -> None:
        start_date, end_date = self._month_range_from_slider()
        start_month = start_date.strftime("%Y-%m")
        end_month = end_date.strftime("%Y-%m")
        self.date_range_text.value = f"{start_month} -> {end_month}"
        self.date_range_hint_text.value = (
            f"Rango activo {start_date.strftime('%Y-%m-%d')} a "
            f"{end_date.strftime('%Y-%m-%d')}. El resto de filtros esta en avanzados."
        )

    def _on_month_range_change(self, event: ft.Event[ft.RangeSlider]) -> None:
        self.date_range_slider.start_value = event.control.start_value
        self.date_range_slider.end_value = event.control.end_value
        self._update_date_range_label()
        self.page.update()

    def _on_month_range_change_end(self, event: ft.Event[ft.RangeSlider]) -> None:
        self.date_range_slider.start_value = event.control.start_value
        self.date_range_slider.end_value = event.control.end_value
        self._apply_month_slider_to_fields()
        self.refresh_view("Rango temporal global actualizado")

    def toggle_advanced_filters(self, _event: Any = None) -> None:
        self.advanced_filters_visible = not self.advanced_filters_visible
        self.advanced_filters_panel.visible = self.advanced_filters_visible
        self.advanced_filters_toggle_button.icon = (
            ft.Icons.EXPAND_LESS if self.advanced_filters_visible else ft.Icons.TUNE
        )
        self.advanced_filters_toggle_button.text = (
            "Ocultar filtros" if self.advanced_filters_visible else "Filtros avanzados"
        )
        self.page.update()

    def _apply_control_tooltips(self) -> None:
        tooltips = {
            self.refresh_button: "Recalcula filtros, graficas y tablas.",
            self.export_button: "Guarda el resumen agrupado actual en CSV.",
            self.apply_rules_button: "Aplica reglas regex locales antes de usar IA.",
            self.clear_cache_button: "Borra cache IA y categorias persistidas.",
            self.ai_full_button: "Categoriza pendientes desde Root en segundo plano.",
            self.ai_node_button: "Categoriza pendientes desde el nodo IA seleccionado.",
            self.ask_ai_button: "Analiza la vista o la seleccion usando resumen y movimientos visibles.",
            self.recategorize_ai_button: "Vuelve a categorizar la vista o seleccion usando tu instruccion como correccion.",
            self.pending_only_checkbox: "Muestra solo movimientos aun sin categoria.",
            self.review_only_checkbox: "Muestra baja confianza y revision manual.",
            self.scope_filter_checkbox: "Limita la tabla al nodo seleccionado en Economia.",
            self.category_mode_checkbox: "Agrupa por taxonomia si hay categorias guardadas.",
            self.time_display_dropdown: "Cambia la visualizacion del rango global activo: total, diario medio, mensual medio, mes concreto, estacional medio o anual medio.",
            self.visual_month_dropdown: "Elige el mes concreto que quieres visualizar dentro del periodo global ya filtrado.",
            self.chart_type_dropdown: "Elige una grafica funcional para entender sectores, meses, entradas, ahorro o picos diarios dentro del filtro global.",
            self.vision_section_dropdown: "Cambia entre vista general, graficas/resumen y movimientos sin scrollear por toda Vision.",
            self.date_range_slider: "Arrastra para elegir de que mes a que mes entra en toda la aplicacion.",
            self.advanced_filters_toggle_button: "Muestra u oculta regex, agrupacion, importes, modo visual y filtros IA.",
            self.save_context_button: "Anade esta memoria como hecho confirmado por el usuario y limpia el formulario.",
            self.apply_ai_clarification_button: "Convierte tu respuesta a dudas o etiquetas concretas en una recategorizacion guiada.",
        }
        for control, tooltip in tooltips.items():
            control.tooltip = tooltip
        self.ai_scope_dropdown.tooltip = "Elige si DeepSeek trabaja sobre toda la vista filtrada o solo sobre las filas seleccionadas."
        self.ai_question_field.tooltip = "Escribe una instruccion operativa: analizar, ubicar movimientos, revisar fechas o recategorizar."
        self.ai_clarification_field.tooltip = "Responde dudas de DeepSeek o indica etiquetas concretas para que las aplique sobre la vista o seleccion."
        self.confirmed_context_field.tooltip = "Anade un hecho confirmado. Al guardar se registra como fuente usuario y el formulario queda limpio."

    def _load_rules_from_disk(self) -> None:
        if not RULES_FILE.exists():
            self._update_rules_count()
            return

        try:
            payload = json.loads(RULES_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("No se pudieron cargar reglas: %s", exc)
            self._update_rules_count()
            return

        self.rules_field.value = str(payload.get("rules_text", RULES_PLACEHOLDER))
        self._update_rules_count()

    def _update_rules_count(self) -> list[CompiledRule]:
        rules = parse_category_rules(self.rules_field.value or "")
        self.rules_count_text.value = (
            f"{format_count(len(rules))} reglas activas"
            if rules
            else "Sin reglas locales"
        )
        return rules

    def _parse_review_threshold(self) -> float:
        raw_value = self.review_threshold_field.value.strip()
        try:
            threshold = float(raw_value.replace(",", "."))
        except ValueError:
            threshold = DEFAULT_REVIEW_CONFIDENCE
        threshold = max(0.0, min(1.0, threshold))
        self.review_threshold_field.value = f"{threshold:.2f}"
        return threshold

    def _show_snackbar(self, message: str, *, error: bool = False) -> None:
        self.page.show_dialog(
            ft.SnackBar(
                content=ft.Text(message, color="#FFFFFF"),
                bgcolor=PALETTE.danger if error else PALETTE.ink,
                duration=2500,
                behavior=ft.SnackBarBehavior.FLOATING,
                show_close_icon=True,
            )
        )

    def _set_status(self, message: str, *, notify: bool = False, error: bool = False) -> None:
        self.status_text.value = message
        logger.info(message)
        if notify:
            self._show_snackbar(message, error=error)

    def _set_busy(self, is_busy: bool, *, message: str | None = None) -> None:
        self.worker_active = is_busy
        self.progress_bar.visible = is_busy
        for control in self.busy_controls:
            control.disabled = is_busy
        if message is not None:
            self._set_status(message)

    def _is_pending(self, frame: pd.DataFrame) -> pd.Series:
        return build_pending_mask(frame)

    def _is_review_candidate(self, frame: pd.DataFrame) -> pd.Series:
        return build_review_mask(
            frame,
            confidence_threshold=self._parse_review_threshold(),
        )

    def _apply_rule_assignments(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        working = ensure_category_columns(frame, self.category_tree)
        rules = self._update_rules_count()
        if not rules:
            return working, 0

        suggested = build_group_column(working, rules)
        missing_mask = self._is_pending(working)
        rule_mask = missing_mask & suggested.notna() & suggested.astype(str).str.strip().ne("")
        if not rule_mask.any():
            return working, 0

        assignments = build_assignments_from_leaf_series(
            suggested.loc[rule_mask],
            self.category_tree,
            source="regla_local",
            confidence=1.0,
            reason="Regla local",
        )
        updated = apply_category_assignments(working, assignments, self.category_tree)
        return updated, int(rule_mask.sum())

    def _apply_rules_to_master(self) -> int:
        updated, applied = self._apply_rule_assignments(self.df)
        self.df = updated
        if applied:
            save_categories_to_disk(self.df)
        return applied

    def _get_ai_scope_path(self) -> str | None:
        if not self.selected_node_path or self.selected_node_path == "Root":
            return None
        return self.selected_node_path

    def _prepare_df_with_rules(self) -> tuple[pd.DataFrame, str]:
        working, _ = self._apply_rule_assignments(self.df.copy())
        requested_group = self.group_dropdown.value or "Concepto"
        if self.category_mode_checkbox.value and requested_group in {"Concepto", "Movimiento"}:
            effective_group = "CategoriaLeaf"
        else:
            effective_group = requested_group
        if effective_group not in working.columns:
            effective_group = "Concepto"
        return working, effective_group

    def _build_current_view(
        self,
        *,
        apply_scope_filter: bool = True,
        apply_pending_filter: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        pattern = self.pattern_field.value.strip()
        value_min = parse_float(self.amount_min_field.value)
        value_max = parse_float(self.amount_max_field.value)
        date_from = parse_date(self.date_from_field.value)
        date_to = parse_date(self.date_to_field.value)

        working, effective_group = self._prepare_df_with_rules()
        mask = build_filter_mask(
            working,
            group_col=effective_group,
            pattern=pattern,
            importe_range=(value_min, value_max),
            date_range=(date_from, date_to),
        )
        subset = working.loc[mask].copy()

        if apply_scope_filter and self.scope_filter_checkbox.value:
            selected_path = self._get_ai_scope_path()
            if selected_path is not None and "CategoriaPath" in subset.columns:
                subset = subset.loc[
                    subset["CategoriaPath"].fillna("").astype(str).str.startswith(selected_path)
                ].copy()

        if apply_pending_filter and self.pending_only_checkbox.value:
            subset = subset.loc[self._is_pending(subset)].copy()

        if self.review_only_checkbox.value:
            subset = subset.loc[self._is_review_candidate(subset)].copy()

        subset = subset.sort_values(["Fecha", "Importe"], ascending=[False, True])
        summary = summarize_subset(subset, effective_group)
        return working, subset, summary, effective_group

    def _build_comparison_source(
        self,
        working: pd.DataFrame,
        effective_group: str,
    ) -> pd.DataFrame:
        pattern = self.pattern_field.value.strip()
        value_min = parse_float(self.amount_min_field.value)
        value_max = parse_float(self.amount_max_field.value)
        mask = build_filter_mask(
            working,
            group_col=effective_group,
            pattern=pattern,
            importe_range=(value_min, value_max),
            date_range=(None, None),
        )
        subset = working.loc[mask].copy()

        if self.scope_filter_checkbox.value:
            selected_path = self._get_ai_scope_path()
            if selected_path is not None and "CategoriaPath" in subset.columns:
                subset = subset.loc[
                    subset["CategoriaPath"].fillna("").astype(str).str.startswith(selected_path)
                ].copy()

        if self.pending_only_checkbox.value:
            subset = subset.loc[self._is_pending(subset)].copy()

        if self.review_only_checkbox.value:
            subset = subset.loc[self._is_review_candidate(subset)].copy()

        return subset

    def _month_from_frame(self, frame: pd.DataFrame) -> str | None:
        if frame.empty or "Fecha" not in frame.columns:
            return None
        dates = pd.to_datetime(frame["Fecha"], errors="coerce").dropna()
        if dates.empty:
            return None
        return str(dates.max().to_period("M"))

    def _baseline_frame_for_month(
        self,
        frame: pd.DataFrame,
        month: str | None,
        *,
        months: int = 3,
    ) -> pd.DataFrame:
        if frame.empty or month is None or "Fecha" not in frame.columns:
            return frame.head(0).copy()
        working = frame.copy()
        working["_month"] = pd.to_datetime(working["Fecha"], errors="coerce").dt.to_period("M")
        working = working.dropna(subset=["_month"])
        if working.empty:
            return frame.head(0).copy()
        target = pd.Period(month, freq="M")
        previous = [value for value in sorted(working["_month"].unique()) if value < target]
        selected = previous[-months:]
        return working.loc[working["_month"].isin(selected)].drop(columns=["_month"]).copy()

    def _month_options_from_frame(self, frame: pd.DataFrame) -> list[str]:
        if frame.empty or "Fecha" not in frame.columns:
            return []
        dates = pd.to_datetime(frame["Fecha"], errors="coerce").dropna()
        if dates.empty:
            return []
        months = sorted(dates.dt.to_period("M").astype(str).unique().tolist())
        return months

    def _sync_visual_month_dropdown(self, frame: pd.DataFrame) -> None:
        months = self._month_options_from_frame(frame)
        self.visual_month_dropdown.options = [
            ft.dropdown.Option(month, month) for month in months
        ]
        if months:
            if self.selected_visual_month not in months:
                self.selected_visual_month = months[-1]
            self.visual_month_dropdown.value = self.selected_visual_month
        else:
            self.selected_visual_month = None
            self.visual_month_dropdown.value = None
        self.visual_month_dropdown.visible = (
            str(self.time_display_dropdown.value or "total") == "monthly_selected"
        )

    def _build_visual_subset(self, subset: pd.DataFrame) -> pd.DataFrame:
        mode = str(self.time_display_dropdown.value or "total")
        if mode != "monthly_selected":
            return subset
        self.selected_visual_month = self.visual_month_dropdown.value
        return filter_dataframe_to_month(subset, self.selected_visual_month)

    def refresh_view(self, status_message: str = "Vista actualizada") -> None:
        try:
            working, subset, _, effective_group = self._build_current_view()
        except Exception as exc:
            self._set_status(f"Error al aplicar filtros: {exc}", notify=True, error=True)
            self.page.update()
            return

        comparison_source = self._build_comparison_source(working, effective_group)
        self._sync_visual_month_dropdown(subset)
        visible_subset = self._build_visual_subset(subset)
        mode_key = str(self.time_display_dropdown.value or "total")
        self.current_time_mode = build_time_display_mode(
            visible_subset,
            mode_key,
            target_month=self.selected_visual_month,
        )
        summary = scale_summary_for_display(
            summarize_subset(visible_subset, effective_group),
            self.current_time_mode.divisor,
        )

        self.current_subset = visible_subset
        self.current_summary = summary
        self.current_group = effective_group
        if self.focused_group is not None:
            summary_values = set(summary[effective_group].astype(str).tolist())
            if self.focused_group not in summary_values:
                self.focused_group = None

        visible_indices = set(visible_subset.index.tolist())
        self.selected_detail_indices &= visible_indices

        self._update_metrics(visible_subset, summary, effective_group, comparison_source)
        self._render_finance_visuals()
        self._render_summary_rows()
        self._render_detail_rows()
        self._render_taxonomy_rows()
        self._render_ai_review_center()
        self._render_selection_preview()
        self._set_status(status_message)
        self._save_ui_state()
        self.page.update()

    def _update_metrics(
        self,
        subset: pd.DataFrame,
        summary: pd.DataFrame,
        effective_group: str,
        comparison_source: pd.DataFrame,
    ) -> None:
        visual_subset = self._build_visual_subset(subset)
        filtered_count = len(subset)
        pending_count = int(self._is_pending(subset).sum()) if filtered_count else 0
        review_count = int(self._is_review_candidate(subset).sum()) if filtered_count else 0
        manual_count = 0
        if filtered_count and "CategoriaLeaf" in subset.columns:
            manual_count = int(
                subset["CategoriaLeaf"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("Revision manual")
                .sum()
            )

        mode_key = str(self.time_display_dropdown.value or "total")
        target_month = self.visual_month_dropdown.value or self._month_from_frame(subset)
        if mode_key == "monthly_average":
            self.current_monthly_health = build_average_period_health(comparison_source)
        else:
            self.current_monthly_health = build_monthly_health(
                comparison_source,
                target_month=target_month,
            )
        baseline_frame = self._baseline_frame_for_month(
            comparison_source,
            self.current_monthly_health.month,
        )
        bizum_intent = build_bizum_intent_summary(
            visual_subset,
            confirmed_context=self.confirmed_context,
        )
        raw_spend_breakdown = build_spend_breakdown(
            visual_subset,
            group_col=effective_group,
            limit=8,
        )
        self.current_spend_breakdown = scale_spend_breakdown(
            raw_spend_breakdown,
            divisor=self.current_time_mode.divisor,
        )
        self.current_sector_spend = scale_sector_spend_for_display(
            build_sector_spend(
                visual_subset,
                baseline_frame=baseline_frame,
                group_col="CategoriaNivel1",
                limit=8,
            ),
            divisor=self.current_time_mode.divisor,
        )
        self.current_monthly_flow = build_monthly_flow(subset, limit=240)
        raw_savings_opportunities = build_savings_opportunities(
            visual_subset,
            group_col=effective_group,
            limit=4,
        )
        self.current_savings_opportunities = [
            SavingsOpportunity(
                label=item.label,
                amount=item.amount / self.current_time_mode.divisor,
                ratio=item.ratio,
                message=(
                    f"Revisar {item.label}: un recorte del 10% liberaria "
                    f"{item.amount * 0.10 / self.current_time_mode.divisor:.2f} EUR"
                    f"{self.current_time_mode.suffix}."
                ),
            )
            for item in raw_savings_opportunities
        ]
        coverage = 0.0
        if filtered_count and "Grupo" in subset.columns:
            coverage = 1 - (pending_count / filtered_count)

        health = self.current_monthly_health
        display_divisor = max(float(self.current_time_mode.divisor), 1.0)
        display_finance = build_personal_finance_summary(visual_subset)
        display_income = display_finance.real_income_total / display_divisor
        display_expense = display_finance.adjusted_expense_total / display_divisor
        display_margin = display_income - display_expense
        display_savings_rate = display_margin / display_income if display_income > 0 else None
        if mode_key in {"daily", "monthly_average", "seasonal", "yearly"}:
            self.health_heading_text.value = f"Vista: {self.current_time_mode.label}"
            self.health_context_text.value = (
                "Todos los importes principales se muestran con el modo de "
                "visualizacion elegido, no como total absoluto del periodo."
            )
        elif mode_key == "monthly_selected":
            month_label = health.month or "sin mes"
            self.health_heading_text.value = f"Salud del mes: {month_label}"
            self.health_context_text.value = (
                f"Mes concreto frente a {format_count(health.comparison_months)} meses previos "
                f"con los mismos filtros de vista."
            )
        else:
            self.health_heading_text.value = "Salud del periodo"
            self.health_context_text.value = (
                "Importes totales del rango filtrado; cambia el modo para ver medias."
            )
        if mode_key in {"daily", "monthly_average", "seasonal", "yearly", "total"}:
            self.health_alert_text.value = (
                f"{self.current_time_mode.label}. "
                "KPIs, resumenes y graficas usan esta escala."
            )
        else:
            self.health_alert_text.value = f"{health.alert_label}. {health.alert_detail}"
        self.metric_visible_value.value = format_currency(display_income)
        self.metric_visible_hint.value = (
            f"{self.current_time_mode.label}; ingreso real nomina/clases"
        )
        self.metric_review_value.value = format_currency(display_expense)
        self.metric_review_hint.value = (
            f"Gasto propio ajustado{self.current_time_mode.suffix}"
        )
        self.metric_spend_value.value = format_currency(display_margin)
        self.metric_spend_hint.value = (
            f"Ahorro {format_percent(display_savings_rate)}"
            if display_savings_rate is not None
            else "Sin ingresos positivos en la vista"
        )
        self.metric_balance_value.value = health.alert_label
        self.metric_balance_hint.value = health.alert_detail
        self.metric_salary_value.value = format_count(filtered_count)
        self.metric_salary_hint.value = (
            f"{format_count(pending_count)} pendientes IA | "
            f"{format_count(review_count)} revision | "
            f"{format_count(manual_count)} manual"
        )
        self.metric_bizum_value.value = format_currency(
            bizum_intent.shared_reimbursements / display_divisor
        )
        self.metric_bizum_hint.value = (
            "Reducen gasto compartido | "
            f"Clases {format_currency(bizum_intent.class_income / display_divisor)} | "
            f"Salientes {format_currency(bizum_intent.outgoing_payments / display_divisor)}"
        )
        audit_metrics = build_audit_metrics(
            subset,
            confidence_threshold=self._parse_review_threshold(),
        )
        avg_confidence = (
            format_percent(audit_metrics.average_confidence)
            if audit_metrics.average_confidence is not None
            else "n/a"
        )
        self.ai_audit_metrics_text.value = (
            f"Cobertura {format_percent(audit_metrics.coverage_ratio)} | "
            f"Reglas {format_count(audit_metrics.local_rules)} | "
            f"IA {format_count(audit_metrics.ai)} | "
            f"Manual {format_count(audit_metrics.manual)} | "
            f"Cache {format_count(audit_metrics.cache_hits)} | "
            f"Confianza media {avg_confidence}"
        )

        group_label = humanize_group_label(effective_group)
        self.summary_heading_text.value = f"Resumen por {group_label}"
        self.detail_heading_text.value = (
            "Movimientos filtrados" if filtered_count else "Sin movimientos para la vista actual"
        )
        self.context_text.value = (
            f"Vista por {group_label} | {format_count(filtered_count)} movimientos visibles | "
            f"{format_count(len(summary))} grupos | {self.current_time_mode.label} | "
            f"Cobertura {format_percent(coverage)}"
        )
        detail_rows = self._get_detail_rows()
        if self.focused_group:
            total_amount = float(detail_rows["Importe"].sum()) if not detail_rows.empty else 0.0
            self.selection_context_text.value = (
                f"Foco en {self.focused_group} | {format_count(len(detail_rows))} movimientos | "
                f"{format_currency(total_amount / display_divisor)}{self.current_time_mode.suffix}"
            )
        else:
            self.selection_context_text.value = "Mostrando todos los movimientos filtrados"

        self.dataset_meta_text.value = (
            f"{format_count(len(self.df))} movimientos | "
            f"{self.date_min.strftime('%Y-%m-%d')} a {self.date_max.strftime('%Y-%m-%d')}"
        )
        self.scope_badge_text.value = f"{self._get_ai_scope_path() or 'Root'}"
        self.selection_badge_text.value = (
            f"{format_count(len(self.selected_detail_indices))} filas seleccionadas"
        )
        self.taxonomy_scope_text.value = f"Mapa economico: {self._get_ai_scope_path() or 'Root'}"

    def _get_detail_rows(self) -> pd.DataFrame:
        if self.focused_group is None or self.current_subset.empty:
            return self.current_subset
        return self.current_subset.loc[
            self.current_subset[self.current_group].astype(str) == self.focused_group
        ].copy()

    def _set_chart_control(self, control: ft.Control) -> None:
        self.chart_container.content = control
        if self.page:
            self.chart_container.update()

    def adjust_chart_zoom(self, *, axis: str, factor: float) -> None:
        if axis in {"x", "global"}:
            self.chart_zoom_x = max(1.0, min(self.chart_zoom_x * factor, 12.0))
        if axis in {"y", "global"}:
            self.chart_zoom_y = max(1.0, min(self.chart_zoom_y * factor, 12.0))
        self._render_finance_visuals()
        self.page.update()

    def reset_chart_zoom(self, _event: Any = None) -> None:
        self.chart_zoom_x = 1.0
        self.chart_zoom_y = 1.0
        self._render_finance_visuals()
        self.page.update()

    def _render_finance_visuals(self) -> None:
        self.chart_zoom_text.value = (
            f"Zoom X {str(round(self.chart_zoom_x, 1)).replace('.', ',')}x | "
            f"Y {str(round(self.chart_zoom_y, 1)).replace('.', ',')}x"
        )
        max_month_amount = max(
            [
                value
                for item in self.current_monthly_flow
                for value in (item.expenses, item.salary or 0.0, item.saving, item.investment)
            ],
            default=0.0,
        )
        monthly_controls = self._build_monthly_flow_controls(max_month_amount)
        self.monthly_flow_list.controls = monthly_controls or [
            self._empty_state("No hay meses para comparar.")
        ]

        opportunity_controls = [
            ft.Container(
                padding=10,
                border_radius=14,
                bgcolor=PALETTE.surface,
                border=ft.border.all(1, PALETTE.line),
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.TRENDING_DOWN_OUTLINED, color=PALETTE.accent, size=18),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    item.label,
                                    size=13,
                                    weight=ft.FontWeight.W_700,
                                    color=PALETTE.ink,
                                ),
                                ft.Text(item.message, size=12, color=PALETTE.muted),
                            ],
                            spacing=3,
                            expand=True,
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            )
            for item in self.current_savings_opportunities
        ] or [self._empty_state("Sin oportunidades claras en esta vista.")]
        self.savings_opportunity_list.controls = opportunity_controls

        chart_type = str(self.chart_type_dropdown.value or "sectors")
        show_timeline_controls = chart_type == "timeline"
        self.timeline_show_expense_checkbox.visible = show_timeline_controls
        self.timeline_show_salary_checkbox.visible = show_timeline_controls
        self.timeline_show_balance_checkbox.visible = show_timeline_controls
        self.timeline_show_accumulated_expense_checkbox.visible = show_timeline_controls
        self.timeline_show_accumulated_saving_checkbox.visible = show_timeline_controls
        self.timeline_show_accumulated_investment_checkbox.visible = show_timeline_controls
        if chart_type == "timeline":
            self.chart_heading_text.value = "Mes a mes"
            self.chart_help_text.value = (
                "Todo el periodo filtrado: barras de gasto y nomina, balance y acumulados. "
                "Usa zoom X, zoom Y, zoom global o reset."
            )
            self._set_chart_control(
                build_monthly_chart(
                    self.current_monthly_flow,
                    zoom_x=self.chart_zoom_x,
                    zoom_y=self.chart_zoom_y,
                    show_expense=bool(self.timeline_show_expense_checkbox.value),
                    show_salary=bool(self.timeline_show_salary_checkbox.value),
                    show_balance=bool(self.timeline_show_balance_checkbox.value),
                    show_accumulated_expense=bool(
                        self.timeline_show_accumulated_expense_checkbox.value
                    ),
                    show_accumulated_saving=bool(
                        self.timeline_show_accumulated_saving_checkbox.value
                    ),
                    show_accumulated_investment=bool(
                        self.timeline_show_accumulated_investment_checkbox.value
                    ),
                )
            )
            self.spend_breakdown_list.controls = self._build_monthly_flow_controls(
                max_month_amount
            ) or [
                self._empty_state("No hay meses para dibujar.")
            ]
        elif chart_type == "income":
            finance_summary = build_personal_finance_summary(self.current_subset)
            display_divisor = max(float(self.current_time_mode.divisor), 1.0)
            values = [
                (
                    "Gasto propio ajustado",
                    finance_summary.adjusted_expense_total / display_divisor,
                    PALETTE.danger,
                ),
                ("Gasto bruto", finance_summary.expense_total / display_divisor, "#B96A64"),
                (
                    "Nomina",
                    (finance_summary.salary_period or 0.0) / display_divisor,
                    PALETTE.accent,
                ),
                ("Clases", finance_summary.class_income / display_divisor, PALETTE.success),
                (
                    "Bizums compartidos",
                    finance_summary.shared_bizum_reimbursements / display_divisor,
                    "#7C8DFF",
                ),
                (
                    "Otros positivos a revisar",
                    finance_summary.other_positive_income / display_divisor,
                    PALETTE.primary,
                ),
            ]
            max_value = max([value for _, value, _ in values], default=0.0)
            self.chart_heading_text.value = "Entradas y gastos"
            self.chart_help_text.value = (
                "Ingreso real es nomina y clases; los Bizums compartidos reducen gasto propio, no son ingreso neto."
            )
            self._set_chart_control(build_value_bar_chart(values))
            self.spend_breakdown_list.controls = [
                self._breakdown_bar(
                    label,
                    value,
                    value / max_value if max_value else 0.0,
                    self.current_time_mode.label.lower(),
                    color,
                )
                for label, value, color in values
                if value > 0
            ] or [self._empty_state("No hay entradas o gastos para comparar.")]
        elif chart_type == "cashflow":
            nature = build_cashflow_nature_summary(self.current_subset)
            display_divisor = max(float(self.current_time_mode.divisor), 1.0)
            values = [
                ("Ingreso real", nature.income / display_divisor, PALETTE.success),
                ("Consumo", nature.consumption / display_divisor, PALETTE.danger),
                ("Ahorro movido", nature.saving / display_divisor, PALETTE.accent),
                ("Inversion", nature.investment / display_divisor, "#7C8DFF"),
                (
                    "Reembolsos",
                    nature.shared_reimbursements / display_divisor,
                    "#5BC0BE",
                ),
            ]
            max_value = max([value for _, value, _ in values], default=0.0)
            self.chart_heading_text.value = "Naturaleza financiera"
            self.chart_help_text.value = (
                "Usa el mismo ingreso real que los KPIs: nomina y Bizums de clases."
            )
            self._set_chart_control(build_value_bar_chart(values))
            self.spend_breakdown_list.controls = [
                self._breakdown_bar(
                    label,
                    value,
                    value / max_value if max_value else 0.0,
                    self.current_time_mode.label.lower(),
                    color,
                )
                for label, value, color in values
                if value > 0
            ] or [self._empty_state("No hay importes para separar por naturaleza.")]
        elif chart_type == "opportunities":
            self.chart_heading_text.value = "Ahorro"
            self.chart_help_text.value = "Areas grandes donde revisar primero posibles recortes."
            opportunity_values = [
                (item.label, item.amount, PALETTE.accent)
                for item in self.current_savings_opportunities
            ]
            self._set_chart_control(
                build_value_bar_chart(opportunity_values)
                if opportunity_values
                else _empty_chart_svg("Sin oportunidades")
            )
            self.spend_breakdown_list.controls = opportunity_controls
        elif chart_type == "heatmap":
            heatmap_items = build_daily_consumption_heatmap(self.current_subset)
            self.chart_heading_text.value = "Mapa de calor de gasto"
            self.chart_help_text.value = (
                "Cada celda es un dia: los tonos rojos senalan picos de consumo "
                "donde revisar compras impulsivas, ocio o gastos concentrados."
            )
            self._set_chart_control(build_daily_heatmap_chart(heatmap_items))
            self.spend_breakdown_list.controls = self._build_heatmap_controls(
                heatmap_items
            )
        elif chart_type == "categories":
            group_label = humanize_group_label(self.current_group)
            self.chart_heading_text.value = f"Gasto por {group_label}"
            self.chart_help_text.value = (
                "Ranking de consumo real del grupo activo, sin mezclar ingresos ni ahorro movido."
            )
            self._set_chart_control(build_spend_breakdown_chart(self.current_spend_breakdown))
            self.spend_breakdown_list.controls = [
                self._breakdown_bar(
                    item.label,
                    item.amount,
                    item.ratio,
                    f"{format_count(item.count)} movimientos{self.current_time_mode.suffix}",
                    PALETTE.danger,
                )
                for item in self.current_spend_breakdown
            ] or [self._empty_state("No hay gasto de consumo con estos filtros.")]
        else:
            self.chart_heading_text.value = "Sectores principales"
            self.chart_help_text.value = (
                "Gasto del mes por CategoriaNivel1 y diferencia frente a la media reciente."
            )
            sector_items = self.current_sector_spend
            if not sector_items and self.current_spend_breakdown:
                self.chart_help_text.value = (
                    "Gasto de la vista actual. Sin baseline suficiente para comparar sectores."
                )
            chart_items = [
                SpendBreakdownItem(
                    label=item.label,
                    amount=item.amount,
                    count=item.count,
                    ratio=item.ratio,
                )
                for item in sector_items
            ]
            if not chart_items:
                chart_items = self.current_spend_breakdown
            self._set_chart_control(
                build_spend_breakdown_chart(chart_items)
            )
            self.spend_breakdown_list.controls = [
                self._sector_bar(item) for item in sector_items
            ] or [
                self._breakdown_bar(
                    item.label,
                    item.amount,
                    item.ratio,
                    f"{format_count(item.count)} movimientos{self.current_time_mode.suffix}",
                    PALETTE.danger,
                )
                for item in self.current_spend_breakdown
            ] or [self._empty_state("No hay sectores para dibujar con estos filtros.")]

    def _build_monthly_flow_controls(self, max_month_amount: float) -> list[ft.Control]:
        controls: list[ft.Control] = []
        for item in self.current_monthly_flow:
            controls.append(
                ft.Container(
                    padding=12,
                    border_radius=16,
                    bgcolor=PALETTE.surface,
                    border=ft.border.all(1, PALETTE.line),
                    content=ft.Column(
                        controls=[
                            ft.Row(
                                controls=[
                                    ft.Text(
                                        item.month,
                                        size=13,
                                        weight=ft.FontWeight.W_700,
                                        color=PALETTE.ink,
                                    ),
                                    ft.Text(
                                        f"Banco {format_currency(item.bank_cashflow)}",
                                        size=12,
                                        color=PALETTE.muted,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            ),
                            self._ratio_bar(
                                "Gasto",
                                item.expenses,
                                max_month_amount,
                                PALETTE.danger,
                            ),
                            self._ratio_bar(
                                "Nomina",
                                item.salary or 0.0,
                                max_month_amount,
                                PALETTE.accent,
                            ),
                            self._ratio_bar(
                                "Ahorro",
                                item.saving,
                                max_month_amount,
                                "#5BC0BE",
                            ),
                            self._ratio_bar(
                                "Inversion",
                                item.investment,
                                max_month_amount,
                                "#7C8DFF",
                            ),
                            ft.Text(
                                (
                                    f"Clases {format_currency(item.class_income)} | "
                                    f"Compartidos {format_currency(item.shared_bizum_reimbursements)}"
                                ),
                                size=11,
                                color=PALETTE.muted,
                            ),
                        ],
                        spacing=7,
                    ),
                )
            )
        return controls

    def _build_heatmap_controls(
        self,
        items: list[DailyConsumptionHeatmapItem],
    ) -> list[ft.Control]:
        positive_items = [item for item in items if item.amount > 0]
        if not positive_items:
            return [self._empty_state("No hay gasto de consumo diario en esta vista.")]

        controls: list[ft.Control] = [
            ft.Text(
                "Dias a revisar primero",
                size=15,
                weight=ft.FontWeight.W_600,
                color=PALETTE.ink,
            )
        ]
        for item in sorted(positive_items, key=lambda row: row.amount, reverse=True)[:5]:
            controls.append(
                self._breakdown_bar(
                    item.date.strftime("%Y-%m-%d"),
                    item.amount,
                    item.ratio,
                    (
                        f"{_weekday_name(item.date)} | "
                        f"{format_count(item.count)} movimientos"
                    ),
                    _heatmap_color(item.ratio),
                )
            )

        weekday_totals: dict[int, list[float]] = {index: [] for index in range(7)}
        for item in items:
            weekday_totals[int(item.date.weekday())].append(item.amount)
        weekday_averages = [
            (
                weekday,
                sum(values) / len(values) if values else 0.0,
            )
            for weekday, values in weekday_totals.items()
        ]
        max_average = max((value for _, value in weekday_averages), default=0.0)
        controls.append(
            ft.Text(
                "Media por dia de la semana",
                size=15,
                weight=ft.FontWeight.W_600,
                color=PALETTE.ink,
            )
        )
        for weekday, value in sorted(
            weekday_averages,
            key=lambda row: row[1],
            reverse=True,
        )[:4]:
            ratio = value / max_average if max_average else 0.0
            controls.append(
                self._breakdown_bar(
                    _weekday_name_from_index(weekday),
                    value,
                    ratio,
                    "Promedio de consumo diario en el rango visible",
                    _heatmap_color(ratio),
                )
            )
        return controls

    def _breakdown_bar(
        self,
        label: str,
        amount: float,
        ratio: float,
        detail: str,
        color: str,
    ) -> ft.Control:
        return ft.Container(
            padding=12,
            border_radius=16,
            bgcolor=PALETTE.surface,
            border=ft.border.all(1, PALETTE.line),
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(
                                label,
                                size=13,
                                weight=ft.FontWeight.W_700,
                                color=PALETTE.ink,
                                expand=True,
                            ),
                            ft.Text(
                                format_currency(amount),
                                size=12,
                                weight=ft.FontWeight.W_700,
                                color=color,
                            ),
                        ],
                    ),
                    ft.ProgressBar(
                        value=max(0.02, min(ratio, 1.0)),
                        color=color,
                        bgcolor=PALETTE.surface_alt,
                        bar_height=8,
                        border_radius=8,
                    ),
                    ft.Text(
                        f"{format_percent(ratio)} | {detail}",
                        size=11,
                        color=PALETTE.muted,
                    ),
                ],
                spacing=7,
            ),
        )

    def _sector_bar(self, item: SectorSpendItem) -> ft.Control:
        if item.delta_amount > 0:
            delta_text = f"{format_currency(item.delta_amount)} sobre media"
            delta_color = PALETTE.danger
        elif item.delta_amount < 0:
            delta_text = f"{format_currency(abs(item.delta_amount))} bajo media"
            delta_color = PALETTE.success
        else:
            delta_text = "Sin desviacion frente a media"
            delta_color = PALETTE.muted
        average_text = (
            f"Media {format_currency(item.average_amount)}"
            if item.average_amount > 0
            else "Sin media previa"
        )
        return ft.Container(
            padding=12,
            border_radius=16,
            bgcolor=PALETTE.surface,
            border=ft.border.all(1, PALETTE.line),
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(
                                item.label,
                                size=13,
                                weight=ft.FontWeight.W_700,
                                color=PALETTE.ink,
                                expand=True,
                            ),
                            ft.Text(
                                format_currency(item.amount),
                                size=12,
                                weight=ft.FontWeight.W_700,
                                color=PALETTE.danger,
                            ),
                        ],
                    ),
                    ft.ProgressBar(
                        value=max(0.02, min(item.ratio, 1.0)),
                        color=PALETTE.danger,
                        bgcolor=PALETTE.surface_alt,
                        bar_height=8,
                        border_radius=8,
                    ),
                    ft.Row(
                        controls=[
                            ft.Text(
                                f"{format_count(item.count)} movimientos | {average_text}",
                                size=11,
                                color=PALETTE.muted,
                                expand=True,
                            ),
                            ft.Text(delta_text, size=11, color=delta_color),
                        ],
                    ),
                ],
                spacing=7,
            ),
        )

    def _ratio_bar(
        self,
        label: str,
        value: float,
        max_value: float,
        color: str,
    ) -> ft.Control:
        ratio = value / max_value if max_value > 0 else 0.0
        return ft.Row(
            controls=[
                ft.Text(label, size=11, color=PALETTE.muted, width=52),
                ft.Container(
                    content=ft.ProgressBar(
                        value=max(0.02, min(ratio, 1.0)) if value > 0 else 0.0,
                        color=color,
                        bgcolor=PALETTE.surface_alt,
                        bar_height=7,
                        border_radius=8,
                    ),
                    expand=True,
                ),
                ft.Text(format_currency(value), size=11, color=PALETTE.ink, width=92),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _render_summary_rows(self) -> None:
        controls: list[ft.Control] = []
        rendered_summary = self.current_summary.head(SUMMARY_ROW_LIMIT)
        for row in rendered_summary.itertuples(index=False):
            group_name = str(getattr(row, self.current_group))
            net_amount = float(row.importe_total)
            consumption = float(getattr(row, "gasto_consumo", 0.0))
            income = float(getattr(row, "ingresos", 0.0))
            saving = float(getattr(row, "ahorro", 0.0))
            investment = float(getattr(row, "inversion", 0.0))
            primary_label = "Gasto real"
            primary_amount = consumption
            amount_color = PALETTE.danger
            if primary_amount <= 0 and saving > 0:
                primary_label = "Ahorro movido"
                primary_amount = saving
                amount_color = PALETTE.accent
            elif primary_amount <= 0 and investment > 0:
                primary_label = "Inversion"
                primary_amount = investment
                amount_color = "#7C8DFF"
            elif primary_amount <= 0 and income > 0:
                primary_label = "Ingreso real"
                primary_amount = income
                amount_color = PALETTE.success
            elif primary_amount <= 0:
                primary_label = "Neto"
                primary_amount = net_amount
                amount_color = PALETTE.success if net_amount >= 0 else PALETTE.danger
            selected = group_name == self.focused_group
            badge_bg = PALETTE.primary_soft if selected else PALETTE.surface_alt
            breakdown_bits = [f"Neto {format_currency(net_amount)}"]
            if income > 0:
                breakdown_bits.append(f"Ingreso real {format_currency(income)}")
            if saving > 0:
                breakdown_bits.append(f"Ahorro {format_currency(saving)}")
            if investment > 0:
                breakdown_bits.append(f"Inversion {format_currency(investment)}")
            row_card = ft.Container(
                    padding=16,
                    border_radius=18,
                    bgcolor=badge_bg,
                    border=ft.border.all(
                        1,
                        PALETTE.primary if selected else PALETTE.line,
                    ),
                    on_click=lambda _event, name=group_name: self.toggle_group_focus(name),
                    content=ft.Row(
                        controls=[
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        group_name,
                                        size=15,
                                        weight=ft.FontWeight.W_600,
                                        color=PALETTE.ink,
                                    ),
                                    ft.Text(
                                        (
                                            f"{format_count(int(row.count))} movimientos | "
                                            + " | ".join(breakdown_bits[:2])
                                        ),
                                        size=12,
                                        color=PALETTE.muted,
                                    ),
                                ],
                                spacing=4,
                                expand=True,
                            ),
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        format_currency(primary_amount),
                                        size=15,
                                        weight=ft.FontWeight.W_700,
                                        color=amount_color,
                                        text_align=ft.TextAlign.RIGHT,
                                    ),
                                    ft.Text(
                                        primary_label,
                                        size=12,
                                        color=PALETTE.muted,
                                    ),
                                ],
                                spacing=4,
                                horizontal_alignment=ft.CrossAxisAlignment.END,
                            ),
                            ft.PopupMenuButton(
                                icon=ft.Icons.MORE_VERT,
                                icon_color=PALETTE.muted,
                                tooltip="Acciones DeepSeek para esta categoria",
                                items=[
                                    ft.PopupMenuItem(
                                        content="Preguntar a DeepSeek",
                                        icon=ft.Icons.PSYCHOLOGY_ALT_OUTLINED,
                                        on_click=(
                                            lambda _event, name=group_name: (
                                                self.ask_deepseek_about_group(name)
                                            )
                                        ),
                                    ),
                                    ft.PopupMenuItem(
                                        content="Rehacer categoria",
                                        icon=ft.Icons.AUTO_FIX_HIGH_OUTLINED,
                                        on_click=(
                                            lambda _event, name=group_name: (
                                                self.recategorize_group_with_deepseek(name)
                                            )
                                        ),
                                    ),
                                ],
                            ),
                        ]
                    ),
                )
            controls.append(
                ft.GestureDetector(
                    content=row_card,
                    on_secondary_tap=(
                        lambda _event, name=group_name: self.ask_deepseek_about_group(name)
                    ),
                )
            )

        if not controls:
            controls.append(self._empty_state("No hay grupos para la vista actual."))
        hidden_groups = len(self.current_summary) - len(rendered_summary)
        if hidden_groups > 0:
            controls.append(
                self._empty_state(
                    f"Mostrando {format_count(len(rendered_summary))} grupos. "
                    f"{format_count(hidden_groups)} mas quedan fuera para mantener la UI rapida."
                )
            )
        self.summary_list.controls = controls

    def _render_detail_rows(self) -> None:
        all_detail_rows = self._get_detail_rows()
        detail_rows = all_detail_rows.head(DETAIL_ROW_LIMIT)
        controls: list[ft.Control] = []
        review_mask = (
            self._is_review_candidate(detail_rows)
            if not detail_rows.empty
            else pd.Series(dtype=bool)
        )

        for row_index, row in detail_rows.iterrows():
            amount_value = float(row["Importe"])
            category_path = row.get("CategoriaPath", "")
            if pd.isna(category_path) or not str(category_path).strip():
                category_path = row.get("Grupo", "")
            source_value = row.get("CategoriaFuente", "")
            if pd.isna(source_value):
                source_value = ""
            confidence_value = format_confidence(row.get("CategoriaConfianza", pd.NA))
            selected = row_index in self.selected_detail_indices
            in_review = bool(review_mask.get(row_index, False))

            controls.append(
                ft.Container(
                    padding=16,
                    border_radius=18,
                    bgcolor=PALETTE.primary_soft if selected else PALETTE.surface,
                    border=ft.border.all(
                        1,
                        PALETTE.primary if selected else PALETTE.line,
                    ),
                    content=ft.Row(
                        controls=[
                            ft.Checkbox(
                                value=selected,
                                active_color=PALETTE.primary,
                                on_change=lambda event, index=row_index: self.toggle_detail_selection(
                                    index,
                                    bool(event.control.value),
                                ),
                            ),
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        pd.to_datetime(row["Fecha"]).strftime("%Y-%m-%d"),
                                        size=12,
                                        color=PALETTE.muted,
                                    ),
                                    ft.Text(
                                        str(row["Concepto"]),
                                        size=15,
                                        weight=ft.FontWeight.W_700,
                                        color=PALETTE.ink,
                                    ),
                                    ft.Text(
                                        str(row["Movimiento"]),
                                        size=13,
                                        color=PALETTE.muted,
                                    ),
                                ],
                                spacing=4,
                                expand=True,
                            ),
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        format_currency(amount_value),
                                        size=15,
                                        weight=ft.FontWeight.W_700,
                                        color=PALETTE.success
                                        if amount_value >= 0
                                        else PALETTE.danger,
                                        text_align=ft.TextAlign.RIGHT,
                                    ),
                                    ft.Text(
                                        str(category_path) if pd.notna(category_path) else "",
                                        size=12,
                                        color=PALETTE.ink,
                                        text_align=ft.TextAlign.RIGHT,
                                    ),
                                    ft.Text(
                                        " | ".join(
                                            part
                                            for part in (
                                                str(source_value),
                                                confidence_value or "sin confianza",
                                                "revision" if in_review else "",
                                            )
                                            if part
                                        ),
                                        size=12,
                                        color=PALETTE.warning if in_review else PALETTE.muted,
                                        text_align=ft.TextAlign.RIGHT,
                                    ),
                                ],
                                spacing=4,
                                horizontal_alignment=ft.CrossAxisAlignment.END,
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                )
            )

        if not controls:
            controls.append(self._empty_state("No hay movimientos visibles con los filtros actuales."))
        hidden_rows = len(all_detail_rows) - len(detail_rows)
        if hidden_rows > 0:
            controls.append(
                self._empty_state(
                    f"Mostrando las primeras {format_count(len(detail_rows))} filas. "
                    f"Afina filtros para revisar las {format_count(hidden_rows)} restantes."
                )
            )
        self.detail_list.controls = controls

    def _render_taxonomy_rows(self) -> None:
        search_query = self.taxonomy_search_field.value or ""
        nodes, pending_total = build_taxonomy_nodes(
            self.category_tree,
            self.current_subset,
            selected_path=self.selected_node_path,
            expanded_paths=self.taxonomy_expanded_paths,
            search_query=search_query,
        )
        controls: list[ft.Control] = []
        for node in nodes:
            icon_name = (
                ft.Icons.ADJUST_ROUNDED
                if node.is_leaf
                else ft.Icons.EXPAND_MORE
                if node.expanded
                else ft.Icons.CHEVRON_RIGHT
            )
            controls.append(
                ft.Container(
                    padding=ft.padding.only(
                        left=14 + (node.depth * 18),
                        right=14,
                        top=12,
                        bottom=12,
                    ),
                    border_radius=18,
                    bgcolor=PALETTE.primary_soft if node.selected else PALETTE.surface,
                    border=ft.border.all(
                        1,
                        PALETTE.primary if node.selected else PALETTE.line,
                    ),
                    on_click=lambda _event, path=node.path: self.select_taxonomy_path(path),
                    content=ft.Row(
                        controls=[
                            ft.IconButton(
                                icon=icon_name,
                                icon_color=PALETTE.primary,
                                icon_size=18,
                                tooltip="Expandir o plegar categoria"
                                if node.has_children
                                else "Categoria hoja",
                                disabled=not node.has_children,
                                on_click=lambda _event, path=node.path: (
                                    self.toggle_taxonomy_expansion(path)
                                ),
                            ),
                            ft.Text(
                                node.label,
                                size=14,
                                weight=ft.FontWeight.W_600
                                if node.depth <= 1
                                else ft.FontWeight.W_500,
                                color=PALETTE.ink,
                                expand=True,
                            ),
                            ft.Container(
                                padding=ft.padding.symmetric(horizontal=10, vertical=6),
                                bgcolor=PALETTE.surface_alt,
                                border_radius=999,
                                content=ft.Text(
                                    format_count(node.count),
                                    size=12,
                                    weight=ft.FontWeight.W_600,
                                    color=PALETTE.ink,
                                ),
                            ),
                            ft.Text(node.share_text, size=12, color=PALETTE.muted),
                        ],
                    ),
                )
            )

        if not controls:
            controls.append(self._empty_state("No hay nodos que mostrar."))
        self.taxonomy_list.controls = controls
        self.taxonomy_meta_text.value = (
            f"{format_count(len(self.current_subset))} movs en vista | "
            f"{format_count(pending_total)} pendientes IA"
        )
        self._render_taxonomy_node_summary()

    def _render_taxonomy_node_summary(self) -> None:
        summary = summarize_taxonomy_path(self.current_subset, self.selected_node_path)
        display_divisor = max(float(self.current_time_mode.divisor), 1.0)
        self.taxonomy_node_summary_text.value = (
            f"{format_count(summary.count)} movimientos | "
            f"Neto {format_currency(summary.net / display_divisor)}"
            f"{self.current_time_mode.suffix} | "
            f"{format_count(summary.pending)} pendientes"
        )
        values = [
            ("Consumo", summary.consumption / display_divisor, PALETTE.danger),
            ("Ingresos", summary.income / display_divisor, PALETTE.success),
            ("Ahorro", summary.saving / display_divisor, "#5BC0BE"),
            ("Inversion", summary.investment / display_divisor, "#7C8DFF"),
        ]
        max_value = max([value for _, value, _ in values], default=0.0)
        self.taxonomy_node_values.controls = [
            self._ratio_bar(label, value, max_value, color)
            for label, value, color in values
            if value > 0
        ] or [self._empty_state("Sin importes en este nodo.")]

    def _render_ai_review_center(self) -> None:
        queue_frame = build_prioritized_review_queue(
            self.current_subset,
            confidence_threshold=self._parse_review_threshold(),
            limit=8,
        )
        controls: list[ft.Control] = []
        for row_index, row in queue_frame.iterrows():
            confidence = format_confidence(row.get("CategoriaConfianza", pd.NA))
            selected = row_index in self.selected_detail_indices
            amount_value = float(row.get("Importe", 0.0))
            controls.append(
                ft.Container(
                    padding=12,
                    border_radius=14,
                    bgcolor=PALETTE.primary_soft if selected else PALETTE.surface,
                    border=ft.border.all(
                        1,
                        PALETTE.primary if selected else PALETTE.line,
                    ),
                    on_click=lambda _event, index=row_index: self.toggle_detail_selection(
                        index,
                        index not in self.selected_detail_indices,
                    ),
                    content=ft.Row(
                        controls=[
                            ft.Icon(
                                ft.Icons.PRIORITY_HIGH_ROUNDED,
                                color=PALETTE.warning,
                                size=18,
                            ),
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        str(row.get("Concepto", "")),
                                        size=13,
                                        weight=ft.FontWeight.W_600,
                                        color=PALETTE.ink,
                                    ),
                                    ft.Text(
                                        str(row.get("CategoriaPath", "Pendiente")),
                                        size=12,
                                        color=PALETTE.muted,
                                    ),
                                ],
                                spacing=3,
                                expand=True,
                            ),
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        format_currency(amount_value),
                                        size=12,
                                        weight=ft.FontWeight.W_700,
                                        color=PALETTE.danger
                                        if amount_value < 0
                                        else PALETTE.success,
                                    ),
                                    ft.Text(
                                        confidence or "sin confianza",
                                        size=12,
                                        color=PALETTE.warning,
                                    ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.END,
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                )
            )

        if not controls:
            controls.append(self._empty_state("No hay movimientos en cola de revision."))
        self.ai_review_queue.controls = controls

        selected_trace = ""
        for index in sorted(self.selected_detail_indices):
            if index in self.df.index:
                selected_trace = summarize_ai_trace(
                    self.df.loc[index].get("CategoriaTrazaIA", pd.NA)
                )
                break
        self.ai_trace_text.value = (
            selected_trace
            if selected_trace
            else "Selecciona una fila categorizada por IA para ver la explicacion por nodo."
        )

    def _render_selection_preview(self) -> None:
        selected_count = len(self.selected_detail_indices)
        self.selection_preview_text.value = (
            f"{format_count(selected_count)} filas seleccionadas para revision manual."
            if selected_count
            else "Sin filas seleccionadas para revision manual."
        )

        controls: list[ft.Control] = []
        for index in sorted(self.selected_detail_indices)[:5]:
            if index not in self.df.index:
                continue
            row = self.df.loc[index]
            controls.append(
                ft.Container(
                    padding=14,
                    border_radius=16,
                    bgcolor=PALETTE.surface,
                    border=ft.border.all(1, PALETTE.line),
                    content=ft.Row(
                        controls=[
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        str(row.get("Concepto", "")),
                                        size=14,
                                        weight=ft.FontWeight.W_600,
                                        color=PALETTE.ink,
                                    ),
                                    ft.Text(
                                        str(row.get("Movimiento", "")),
                                        size=12,
                                        color=PALETTE.muted,
                                    ),
                                ],
                                spacing=4,
                                expand=True,
                            ),
                            ft.Text(
                                format_currency(float(row.get("Importe", 0.0))),
                                size=13,
                                weight=ft.FontWeight.W_700,
                                color=PALETTE.danger
                                if float(row.get("Importe", 0.0)) < 0
                                else PALETTE.success,
                            ),
                        ]
                    ),
                )
            )

        if selected_count > 5:
            controls.append(
                ft.Text(
                    f"+ {format_count(selected_count - 5)} filas adicionales en la seleccion.",
                    size=12,
                    color=PALETTE.muted,
                )
            )
        self.selection_preview_column.controls = controls

    def _empty_state(self, message: str) -> ft.Control:
        return ft.Container(
            padding=24,
            border_radius=18,
            bgcolor=PALETTE.surface,
            border=ft.border.all(1, PALETTE.line),
            content=ft.Text(message, size=13, color=PALETTE.muted),
        )

    def _refresh_current_metrics(self) -> None:
        working, effective_group = self._prepare_df_with_rules()
        comparison_source = self._build_comparison_source(working, effective_group)
        self._update_metrics(
            self.current_subset,
            self.current_summary,
            self.current_group,
            comparison_source,
        )

    def _refresh_from_event(self, _event: Any = None) -> None:
        self.refresh_view("Vista actualizada")

    def _group_indices(self, group_name: str) -> list[int]:
        if self.current_subset.empty or self.current_group not in self.current_subset.columns:
            return []
        group_values = self.current_subset[self.current_group].fillna("").astype(str)
        indices: list[int] = []
        for index in self.current_subset.loc[group_values.eq(group_name)].index.tolist():
            try:
                indices.append(int(index))
            except (TypeError, ValueError):
                continue
        return indices

    def _prepare_group_for_deepseek(self, group_name: str, instruction: str) -> bool:
        selected_indices = self._group_indices(group_name)
        if not selected_indices:
            self._set_status(
                f"No hay movimientos visibles para {group_name}.",
                notify=True,
                error=True,
            )
            self.page.update()
            return False
        self.focused_group = group_name
        self.selected_detail_indices = set(selected_indices)
        self.ai_scope_dropdown.value = "selection"
        self.ai_question_field.value = instruction
        self.select_tab("vision")
        self._refresh_current_metrics()
        self._render_summary_rows()
        self._render_detail_rows()
        self._render_ai_review_center()
        self._render_selection_preview()
        self._save_ui_state()
        return True

    def ask_deepseek_about_group(self, group_name: str) -> None:
        instruction = (
            f"Analiza la categoria '{group_name}' con sus movimientos visibles. "
            "Identifica si hay recurrentes, suscripciones, Bizums que compensan "
            "gastos compartidos y categorias mal asignadas. Cita movimientos concretos."
        )
        if self._prepare_group_for_deepseek(group_name, instruction):
            self.page.run_task(self.ask_ai_about_current_view)
            self.page.update()

    def recategorize_group_with_deepseek(self, group_name: str) -> None:
        instruction = (
            f"Revisa y corrige la categoria '{group_name}'. Si algun movimiento esta "
            "mal categorizado, reasignalo a la hoja correcta del arbol. Trata Bizums "
            "compartidos como reembolsos que reducen gasto, no como ingreso."
        )
        if self._prepare_group_for_deepseek(group_name, instruction):
            self.page.run_task(self.recategorize_with_ai_guidance)
            self.page.update()

    def _render_taxonomy_from_event(self, _event: Any = None) -> None:
        self._render_taxonomy_rows()
        self.page.update()

    def toggle_taxonomy_expansion(self, path: str) -> None:
        if path in self.taxonomy_expanded_paths:
            self.taxonomy_expanded_paths.remove(path)
        else:
            self.taxonomy_expanded_paths.add(path)
        self._render_taxonomy_rows()
        self.page.update()

    def expand_taxonomy_tree(self, _event: Any = None) -> None:
        nodes, _ = build_taxonomy_nodes(
            self.category_tree,
            self.current_subset,
            selected_path=self.selected_node_path,
        )
        self.taxonomy_expanded_paths = {
            node.path for node in nodes if node.has_children
        } | {"Root"}
        self._render_taxonomy_rows()
        self.page.update()

    def collapse_taxonomy_tree(self, _event: Any = None) -> None:
        self.taxonomy_expanded_paths = {"Root"}
        self._render_taxonomy_rows()
        self.page.update()

    def toggle_group_focus(self, group_name: str) -> None:
        self.focused_group = None if self.focused_group == group_name else group_name
        self._refresh_current_metrics()
        self._render_summary_rows()
        self._render_detail_rows()
        self._render_ai_review_center()
        self._render_selection_preview()
        self.page.update()

    def select_tab(self, tab_name: str) -> None:
        self.current_tab = tab_name
        self._sync_tab_controls()
        self._save_ui_state()
        self.page.update()

    def clear_group_focus(self, _event: Any = None) -> None:
        self.focused_group = None
        self._refresh_current_metrics()
        self._render_summary_rows()
        self._render_detail_rows()
        self._render_ai_review_center()
        self.page.update()

    def toggle_detail_selection(self, index: int, selected: bool) -> None:
        if selected:
            self.selected_detail_indices.add(index)
        else:
            self.selected_detail_indices.discard(index)
        self._refresh_current_metrics()
        self._render_detail_rows()
        self._render_ai_review_center()
        self._render_selection_preview()
        self.page.update()

    def select_visible_rows(self, _event: Any = None) -> None:
        visible_indices = set(self._get_detail_rows().head(DETAIL_ROW_LIMIT).index.tolist())
        self.selected_detail_indices |= visible_indices
        self.refresh_view("Filas visibles seleccionadas")

    def clear_selected_rows(self, _event: Any = None) -> None:
        self.selected_detail_indices.clear()
        self.refresh_view("Seleccion limpiada")

    def select_taxonomy_path(self, path: str) -> None:
        self.selected_node_path = path
        current_parts: list[str] = []
        for part in [part.strip() for part in path.split(" > ") if part.strip()]:
            current_parts.append(part)
            self.taxonomy_expanded_paths.add(" > ".join(current_parts))
        self.refresh_view("Mapa economico actualizado")

    def set_taxonomy_scope_root(self, _event: Any = None) -> None:
        self.selected_node_path = "Root"
        self.refresh_view("Mapa economico restablecido a Root")

    def select_date_preset(self, days: int | None) -> None:
        if days is None:
            self.date_from_field.value = self.date_min.strftime("%Y-%m-%d")
            self.date_to_field.value = self.date_max.strftime("%Y-%m-%d")
            self.date_from_picker.value = self.date_min.date()
            self.date_to_picker.value = self.date_max.date()
            status_message = "Rango global completo aplicado"
        else:
            start = (self.date_max - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            self.date_from_field.value = start
            self.date_to_field.value = self.date_max.strftime("%Y-%m-%d")
            self.date_from_picker.value = pd.to_datetime(start).date()
            self.date_to_picker.value = self.date_max.date()
            status_message = f"Rango global de {days} dias aplicado"
        self._sync_month_slider_from_fields()
        self.refresh_view(status_message)

    def reset_filters(self, _event: Any = None) -> None:
        self.group_dropdown.value = "CategoriaNivel1" if self.category_mode_checkbox.value else "Concepto"
        self.pattern_field.value = ""
        self.amount_min_field.value = f"{self.amount_min:.2f}"
        self.amount_max_field.value = f"{self.amount_max:.2f}"
        self.date_from_field.value = self.default_date_from.strftime("%Y-%m-%d")
        self.date_to_field.value = self.date_max.strftime("%Y-%m-%d")
        self.date_from_picker.value = self.default_date_from.date()
        self.date_to_picker.value = self.date_max.date()
        self._sync_month_slider_from_fields()
        self.pending_only_checkbox.value = False
        self.review_only_checkbox.value = False
        self.scope_filter_checkbox.value = False
        self.time_display_dropdown.value = "total"
        self.chart_type_dropdown.value = "sectors"
        self.chart_zoom_x = 1.0
        self.chart_zoom_y = 1.0
        self.refresh_view("Filtros globales restablecidos")

    async def export_summary(self, _event: Any = None) -> None:
        if self.current_summary.empty:
            self._set_status("No hay datos para exportar.", notify=True)
            self.page.update()
            return

        csv_bytes = self.current_summary.to_csv(index=False).encode("utf-8")
        kwargs: dict[str, Any] = {
            "dialog_title": "Exportar resumen",
            "file_name": EXPORT_FILE.name,
            "initial_directory": str(DIR_EXPORTS),
            "file_type": ft.FilePickerFileType.CUSTOM,
            "allowed_extensions": ["csv"],
        }
        if self.page.web:
            kwargs["src_bytes"] = csv_bytes

        path = await ft.FilePicker().save_file(**kwargs)
        if self.page.web:
            self._set_status("Resumen descargado en el navegador.", notify=True)
        elif path:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_bytes(csv_bytes)
            self._set_status(f"Resumen exportado en {export_path}", notify=True)
        self.page.update()

    def save_rules_to_disk(self, _event: Any = None) -> None:
        payload = {"rules_text": self.rules_field.value}
        RULES_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._update_rules_count()
        self._set_status("Reglas guardadas correctamente", notify=True)
        self.page.update()

    def _resolve_ai_agent_target_frame(self) -> tuple[pd.DataFrame, str]:
        scope = str(self.ai_scope_dropdown.value or "filtered_view")
        if scope == "selection":
            selected_indices = [
                index
                for index in sorted(self.selected_detail_indices)
                if index in self.current_subset.index
            ]
            if not selected_indices:
                raise ValueError("Selecciona una o varias filas visibles para usar la seleccion.")
            return self.current_subset.loc[selected_indices].copy(), "seleccion actual"

        if self.current_subset.empty:
            raise ValueError("No hay movimientos en la vista filtrada actual.")
        return self.current_subset.copy(), "vista filtrada"

    def _build_ai_agent_context(
        self,
        target_frame: pd.DataFrame,
        *,
        scope_label: str,
        instruction: str,
    ) -> dict[str, Any]:
        target_summary = summarize_subset(target_frame, self.current_group).head(12)
        context = build_deepseek_agent_context(
            target_frame,
            group_col=self.current_group,
            confirmed_context=self.confirmed_context,
        )
        context["resumen_visible_por_grupo"] = [
            {
                "grupo": str(row.get(self.current_group, "")),
                "movimientos": int(row.get("count", 0)),
                "gasto_real": round(float(row.get("gasto_consumo", 0.0)), 2),
                "ingresos": round(float(row.get("ingresos", 0.0)), 2),
                "ahorro_movido": round(float(row.get("ahorro", 0.0)), 2),
                "inversion": round(float(row.get("inversion", 0.0)), 2),
                "neto_banco": round(float(row.get("importe_total", 0.0)), 2),
            }
            for row in target_summary.to_dict(orient="records")
        ]
        context["alcance_operativo"] = {
            "scope": scope_label,
            "nodo_ia_seleccionado": self._get_ai_scope_path() or "Root",
            "scope_filter_activo": bool(self.scope_filter_checkbox.value),
            "solo_pendientes": bool(self.pending_only_checkbox.value),
            "solo_revision": bool(self.review_only_checkbox.value),
            "modo_temporal": self.current_time_mode.label,
            "mes_visualizado": self.selected_visual_month,
            "instruccion_usuario": instruction,
        }
        return context

    async def ask_ai_about_current_view(self, _event: Any = None) -> None:
        if self.worker_active:
            self._set_status("Ya hay una tarea IA en curso", notify=True)
            self.page.update()
            return
        question = (self.ai_question_field.value or "").strip()
        if not question:
            self._set_status("Escribe una pregunta para DeepSeek.", notify=True)
            self.page.update()
            return

        try:
            target_frame, scope_label = self._resolve_ai_agent_target_frame()
        except ValueError as exc:
            self._set_status(str(exc), notify=True, error=True)
            self.page.update()
            return

        context = self._build_ai_agent_context(
            target_frame,
            scope_label=scope_label,
            instruction=question,
        )
        self._set_busy(True, message=f"DeepSeek analizando {scope_label}")
        self.ai_answer_text.value = "Analizando movimientos y contexto de la vista..."
        self.page.update()
        monitor_task = asyncio.create_task(self._drain_status_queue())
        try:
            answer = await asyncio.to_thread(
                ask_deepseek_about_expenses,
                question,
                context,
                status_callback=self._status_queue.put,
            )
            self.ai_answer_text.value = answer
            self._set_status("Respuesta DeepSeek lista", notify=True)
        except Exception as exc:
            logger.error("Consulta DeepSeek fallo: %s", exc)
            self.ai_answer_text.value = f"Error al consultar DeepSeek: {exc}"
            self._set_status(f"Consulta DeepSeek error: {exc}", notify=True, error=True)
        finally:
            self._set_busy(False)
            await monitor_task
            self.page.update()

    async def apply_ai_clarification(self, _event: Any = None) -> None:
        clarification = (self.ai_clarification_field.value or "").strip()
        if not clarification:
            self._set_status(
                "Escribe una respuesta o etiquetas concretas para aplicar.",
                notify=True,
            )
            self.page.update()
            return

        base_instruction = (self.ai_question_field.value or "").strip()
        original_instruction = self.ai_question_field.value
        self.ai_question_field.value = (
            f"{base_instruction}\n\n" if base_instruction else ""
        ) + (
            "Respuesta del usuario a dudas y etiquetas concretas:\n"
            f"{clarification}\n"
            "Aplica estas correcciones sobre los movimientos indicados. Si un Bizum "
            "recibido compensa un pago de tarjeta cercano, tratalo como gasto negativo "
            "o reembolso compartido, no como ingreso."
        )
        try:
            await self.recategorize_with_ai_guidance()
        finally:
            self.ai_question_field.value = original_instruction
            self.page.update()

    async def recategorize_with_ai_guidance(self, _event: Any = None) -> None:
        if self.worker_active:
            self._set_status("Ya hay una tarea IA en curso", notify=True)
            self.page.update()
            return

        instruction = (self.ai_question_field.value or "").strip()
        if not instruction:
            self._set_status("Escribe una instruccion para recategorizar.", notify=True)
            self.page.update()
            return

        try:
            target_frame, scope_label = self._resolve_ai_agent_target_frame()
        except ValueError as exc:
            self._set_status(str(exc), notify=True, error=True)
            self.page.update()
            return

        while True:
            try:
                self._status_queue.get_nowait()
            except queue.Empty:
                break

        scope_path = self._get_ai_scope_path() if self.scope_filter_checkbox.value else None
        scope_name = scope_path or "Root"
        rules_applied = self._apply_rules_to_master()
        self._set_busy(
            True,
            message=(
                f"DeepSeek recategorizando {format_count(len(target_frame))} movimientos "
                f"sobre {scope_label}"
            ),
        )
        self.ai_answer_text.value = "Recategorizando con tu correccion..."
        self.page.update()
        monitor_task = asyncio.create_task(self._drain_status_queue())
        try:
            assignments = await asyncio.to_thread(
                categorize_transactions_by_tree,
                target_frame,
                tree=self.category_tree,
                root_path=scope_path,
                cache_path=TREE_LLM_CACHE_FILE,
                status_callback=self._status_queue.put,
                user_instruction=instruction,
            )
            self.df = apply_category_assignments(self.df, assignments, self.category_tree)
            save_categories_to_disk(self.df)
            self.category_mode_checkbox.value = True
            if self.group_dropdown.value in {"Concepto", "Movimiento"}:
                self.group_dropdown.value = "CategoriaLeaf"
            self.ai_answer_text.value = (
                f"Recategorizados {format_count(len(assignments))} movimientos sobre "
                f"{scope_label}. Nodo IA aplicado: {scope_name}. "
                f"Reglas previas: {format_count(rules_applied)}."
            )
            self._set_busy(False)
            self.refresh_view("Recategorizacion IA guiada completada")
            self._show_snackbar(
                (
                    f"DeepSeek recategorizo {format_count(len(assignments))} movimientos "
                    f"en {scope_label} desde {scope_name}."
                )
            )
        except Exception as exc:
            logger.error("Recategorizacion IA guiada fallo: %s", exc)
            self.ai_answer_text.value = f"Error al recategorizar: {exc}"
            self._set_busy(False)
            self._set_status(
                f"Recategorizacion IA error: {exc}",
                notify=True,
                error=True,
            )
        finally:
            await monitor_task
            self.page.update()

    def apply_rules_to_master_dataframe(self, _event: Any = None) -> None:
        applied = self._apply_rules_to_master()
        self.category_mode_checkbox.value = True
        if self.group_dropdown.value in {"Concepto", "Movimiento"}:
            self.group_dropdown.value = "CategoriaLeaf"
        self.refresh_view("Reglas locales aplicadas")
        self._show_snackbar(
            (
                f"Se actualizaron {format_count(applied)} movimientos por reglas."
                if applied
                else "No hubo movimientos pendientes que coincidieran con reglas."
            )
        )

    async def _drain_status_queue(self) -> None:
        while self.worker_active or not self._status_queue.empty():
            changed = False
            while True:
                try:
                    message = self._status_queue.get_nowait()
                except queue.Empty:
                    break
                self._set_status(message)
                changed = True
            if changed:
                self.page.update()
            await asyncio.sleep(0.12)

    async def run_tree_categorization(self, selected_scope: bool) -> None:
        if self.worker_active:
            self._set_status("Ya hay una categorizacion IA en curso", notify=True)
            self.page.update()
            return

        try:
            rules_applied = self._apply_rules_to_master()
            _, subset, _, _ = self._build_current_view(
                apply_scope_filter=False,
                apply_pending_filter=False,
            )
            to_categorize = subset.loc[self._is_pending(subset)].copy()
            scope_path = self._get_ai_scope_path() if selected_scope else None
            if to_categorize.empty:
                self._set_status(
                    "No hay filas pendientes de categorizar en la vista actual.",
                    notify=True,
                )
                self.page.update()
                return

            while True:
                try:
                    self._status_queue.get_nowait()
                except queue.Empty:
                    break

            scope_label = scope_path or "Root"
            self._set_busy(
                True,
                message=(
                    f"IA arbol recorriendo {scope_label} sobre "
                    f"{format_count(len(to_categorize))} movimientos"
                ),
            )
            self.page.update()
            monitor_task = asyncio.create_task(self._drain_status_queue())
            try:
                assignments = await asyncio.to_thread(
                    categorize_transactions_by_tree,
                    to_categorize,
                    tree=self.category_tree,
                    root_path=scope_path,
                    cache_path=TREE_LLM_CACHE_FILE,
                    status_callback=self._status_queue.put,
                )
            finally:
                self.worker_active = False
                await monitor_task

            self.df = apply_category_assignments(self.df, assignments, self.category_tree)
            save_categories_to_disk(self.df)
            self.category_mode_checkbox.value = True
            if self.group_dropdown.value in {"Concepto", "Movimiento"}:
                self.group_dropdown.value = "CategoriaLeaf"
            self._set_busy(False)
            self.refresh_view("Categorizacion IA completada")
            self._show_snackbar(
                (
                    f"IA arbol categorizo {format_count(len(to_categorize))} movimientos. "
                    f"Reglas previas: {format_count(rules_applied)}."
                )
            )
        except Exception as exc:
            logger.error("IA arbol fallo: %s", exc)
            self._set_busy(False)
            self._set_status(f"IA arbol error: {exc}", notify=True, error=True)
            self.page.update()

    def assign_selected_rows_to_node(self, _event: Any = None) -> None:
        if not self.selected_detail_indices:
            self._set_status(
                "Selecciona una o varias filas en la tabla operativa.",
                notify=True,
            )
            self.page.update()
            return

        selected_path = self.selected_node_path
        selected_node = find_node_by_path(self.category_tree, selected_path)
        if selected_node is None:
            self._set_status("Nodo de taxonomia no encontrado.", notify=True, error=True)
            self.page.update()
            return
        if selected_node.get("children"):
            self._set_status(
                "Selecciona una hoja del arbol para asignar manualmente.",
                notify=True,
            )
            self.page.update()
            return

        leaf_name = str(selected_node["name"])
        selected_indices = sorted(self.selected_detail_indices)
        assignments = build_assignments_from_leaf_series(
            pd.Series(leaf_name, index=selected_indices, dtype="object"),
            self.category_tree,
            source="manual_ui",
            confidence=1.0,
            reason=f"Asignacion manual desde {selected_path}",
        )
        self.df = apply_category_assignments(self.df, assignments, self.category_tree)
        save_categories_to_disk(self.df)
        self.category_mode_checkbox.value = True
        if self.group_dropdown.value in {"Concepto", "Movimiento"}:
            self.group_dropdown.value = "CategoriaLeaf"
        self.selected_detail_indices.clear()
        self.refresh_view("Asignacion manual aplicada")
        self._show_snackbar(
            f"Se reasignaron {format_count(len(selected_indices))} movimientos a {selected_path}."
        )

    def accept_selected_ai_rows(self, _event: Any = None) -> None:
        if not self.selected_detail_indices:
            self._set_status("Selecciona filas categorizadas por IA para aceptarlas.", notify=True)
            self.page.update()
            return

        selected_indices = [
            index for index in self.selected_detail_indices if index in self.df.index
        ]
        if not selected_indices:
            self._set_status("La seleccion actual no existe en el dataframe.", notify=True)
            self.page.update()
            return

        self.df = ensure_category_columns(self.df, self.category_tree)
        self.df.loc[selected_indices, "CategoriaFuente"] = "manual_accept"
        self.df.loc[selected_indices, "CategoriaConfianza"] = 1.0
        self.df.loc[
            selected_indices,
            "CategoriaMotivoIA",
        ] = "Categoria IA aceptada manualmente"
        save_categories_to_disk(self.df)
        self.selected_detail_indices.clear()
        self.refresh_view("Categorias IA aceptadas")
        self._show_snackbar(f"Se aceptaron {format_count(len(selected_indices))} movimientos.")

    def send_selected_rows_to_manual_review(self, _event: Any = None) -> None:
        if not self.selected_detail_indices:
            self._set_status("Selecciona filas para enviar a revision manual.", notify=True)
            self.page.update()
            return

        manual_node = find_node_by_path(self.category_tree, "Otros > Revision manual")
        manual_path = "Otros > Revision manual" if manual_node is not None else "Revision manual"
        selected_indices = [
            index for index in self.selected_detail_indices if index in self.df.index
        ]
        assignments = build_assignments_from_leaf_series(
            pd.Series("Revision manual", index=selected_indices, dtype="object"),
            self.category_tree,
            source="manual_review_ui",
            confidence=0.0,
            reason=f"Enviado manualmente a {manual_path}",
        )
        self.df = apply_category_assignments(self.df, assignments, self.category_tree)
        save_categories_to_disk(self.df)
        self.selected_detail_indices.clear()
        self.refresh_view("Movimientos enviados a revision manual")
        self._show_snackbar(
            f"Se enviaron {format_count(len(selected_indices))} movimientos a revision manual."
        )

    def clear_deepseek_cache(self, _event: Any = None) -> None:
        if LLM_CACHE_FILE.exists():
            LLM_CACHE_FILE.unlink()
        if TREE_LLM_CACHE_FILE.exists():
            TREE_LLM_CACHE_FILE.unlink()
        if CATEGORIES_FILE.exists():
            CATEGORIES_FILE.unlink()
        for column in CATEGORY_COLUMNS:
            if column in self.df.columns:
                self.df[column] = pd.NA
        self.category_mode_checkbox.value = False
        self.pending_only_checkbox.value = False
        self.review_only_checkbox.value = False
        self.scope_filter_checkbox.value = False
        self.focused_group = None
        self.selected_detail_indices.clear()
        self.refresh_view("Cache de DeepSeek y categorias limpiada")
        self._show_snackbar("Se limpiaron cache y categorias persistidas.")
