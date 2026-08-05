"""Interactive Flet charts replacing static SVGs."""
from __future__ import annotations

import flet as ft
from typing import Any
import pandas as pd

from typing import TYPE_CHECKING
from src.core.insights import SpendBreakdownItem, MonthlyFlowItem
if TYPE_CHECKING:
    from src.ui.dashboard import DailyConsumptionHeatmapItem
from src.ui.theme import PALETTE

def format_currency(value: float) -> str:
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{formatted} EUR"

def build_empty_chart(message: str) -> ft.Control:
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Icon(ft.Icons.AREA_CHART_OUTLINED, size=48, color=ft.colors.with_opacity(0.3, PALETTE.muted)),
                ft.Text(message, color=PALETTE.muted, size=15, weight=ft.FontWeight.W_500)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10
        ),
        alignment=ft.alignment.center,
        height=220,
        bgcolor=ft.colors.with_opacity(0.4, PALETTE.surface),
        border_radius=18,
        border=ft.border.all(1, ft.colors.with_opacity(0.1, PALETTE.line))
    )

def build_spend_breakdown_chart(items: list[SpendBreakdownItem]) -> ft.Control:
    if not items:
        return build_empty_chart("Sin gastos para graficar")
    rows = items[:6]
    max_amount = max((item.amount for item in rows), default=0.0)
    if max_amount <= 0:
        return build_empty_chart("Sin gastos para graficar")

    chart_groups = []
    colors = [PALETTE.danger, PALETTE.accent, PALETTE.primary, PALETTE.primary_deep, PALETTE.primary_soft, PALETTE.muted]
    
    for i, item in enumerate(rows):
        color = colors[i % len(colors)]
        tooltip = f"{item.label}\\n{format_currency(item.amount)}"
        width_ratio = item.amount / max_amount
        
        bar_row = ft.Row(
            controls=[
                ft.Text(item.label[:22] + ("..." if len(item.label) > 22 else ""), color=PALETTE.ink, width=150, size=13, weight=ft.FontWeight.BOLD),
                ft.Stack(
                    controls=[
                        ft.Container(bgcolor=PALETTE.surface_alt, height=20, border_radius=10, expand=True),
                        ft.Container(
                            gradient=ft.LinearGradient(
                                colors=[color, ft.colors.with_opacity(0.7, color)],
                                begin=ft.Alignment(-1, 0), end=ft.Alignment(1, 0)
                            ),
                            height=20, 
                            border_radius=10, 
                            width=max(8, width_ratio * 380),
                            tooltip=tooltip,
                            animate_size=ft.animation.Animation(800, ft.AnimationCurve.DECELERATE),
                            shadow=ft.BoxShadow(spread_radius=1, blur_radius=8, color=ft.colors.with_opacity(0.3, color))
                        )
                    ],
                    width=380
                ),
                ft.Text(format_currency(item.amount), color=PALETTE.ink, size=12, text_align=ft.TextAlign.RIGHT, expand=True)
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )
        chart_groups.append(bar_row)
        
    return ft.Container(
        content=ft.Column(controls=chart_groups, spacing=11, alignment=ft.MainAxisAlignment.CENTER),
        padding=ft.padding.only(left=18, right=18, top=20, bottom=20),
        height=220,
        bgcolor=PALETTE.surface,
        border_radius=18
    )

def build_monthly_chart(
    items: list[MonthlyFlowItem],
    zoom_x: float = 1.0,
    zoom_y: float = 1.0,
    show_expense: bool = True,
    show_salary: bool = True,
    show_balance: bool = True,
    show_accumulated_expense: bool = True,
    show_accumulated_saving: bool = False,
    show_accumulated_investment: bool = False,
) -> ft.Control:
    if not items:
        return build_empty_chart("Sin meses para graficar")
    
    data_series = []
    visible_count = max(3, int(len(items) / max(float(zoom_x), 1.0)))
    rows = items[-visible_count:]
    
    if show_expense:
        data_series.append(ft.LineChartData(
            data_points=[ft.LineChartDataPoint(i, item.expenses, tooltip=f"Gasto: {format_currency(item.expenses)}") for i, item in enumerate(rows)],
            stroke_width=4, color=PALETTE.danger, curved=True,
            below_line_bgcolor=ft.colors.with_opacity(0.1, PALETTE.danger)
        ))
        
    if show_salary:
        data_series.append(ft.LineChartData(
            data_points=[ft.LineChartDataPoint(i, item.salary or 0.0, tooltip=f"Nomina: {format_currency(item.salary or 0.0)}") for i, item in enumerate(rows)],
            stroke_width=4, color=PALETTE.accent, curved=True,
            below_line_bgcolor=ft.colors.with_opacity(0.1, PALETTE.accent)
        ))

    if show_balance:
        data_series.append(ft.LineChartData(
            data_points=[ft.LineChartDataPoint(i, item.bank_cashflow, tooltip=f"Balance: {format_currency(item.bank_cashflow)}") for i, item in enumerate(rows)],
            stroke_width=3, color=PALETTE.success, curved=False
        ))

    chart = ft.LineChart(
        data_series=data_series,
        border=ft.border.all(1, ft.colors.with_opacity(0.2, PALETTE.line)),
        horizontal_grid_lines=ft.ChartGridLines(interval=1000, color=ft.colors.with_opacity(0.1, PALETTE.line), width=1),
        vertical_grid_lines=ft.ChartGridLines(interval=1, color=ft.colors.with_opacity(0.05, PALETTE.line), width=1),
        left_axis=ft.ChartAxis(labels_size=40, labels_interval=1000),
        bottom_axis=ft.ChartAxis(
            labels=[ft.ChartAxisLabel(value=i, label=ft.Text(item.month[2:], size=10, color=PALETTE.muted)) for i, item in enumerate(rows)],
            labels_size=32
        ),
        tooltip_bgcolor=PALETTE.surface_alt,
        expand=True,
        interactive=True
    )

    return ft.Container(
        content=chart, padding=20, height=220, bgcolor=PALETTE.surface, border_radius=18
    )

def build_value_bar_chart(values: list[tuple[str, float, str]]) -> ft.Control:
    positive_values = [(label, value, color) for label, value, color in values if value > 0]
    if not positive_values:
        return build_empty_chart("Sin valores para graficar")
    
    max_value = max(value for _, value, _ in positive_values)
    chart_groups = []
    
    for label, value, color in positive_values[:5]:
        width_ratio = value / max_value
        tooltip = f"{label}\\n{format_currency(value)}"
        
        bar_row = ft.Row(
            controls=[
                ft.Text(label[:22] + ("..." if len(label) > 22 else ""), color=PALETTE.ink, width=160, size=13, weight=ft.FontWeight.BOLD),
                ft.Stack(
                    controls=[
                        ft.Container(bgcolor=PALETTE.surface_alt, height=22, border_radius=10, expand=True),
                        ft.Container(
                            gradient=ft.LinearGradient(
                                colors=[color, ft.colors.with_opacity(0.7, color)],
                                begin=ft.Alignment(-1, 0), end=ft.Alignment(1, 0)
                            ),
                            height=22, border_radius=10, 
                            width=max(8, width_ratio * 380),
                            tooltip=tooltip, 
                            animate_size=ft.animation.Animation(800, ft.AnimationCurve.DECELERATE),
                            shadow=ft.BoxShadow(spread_radius=1, blur_radius=8, color=ft.colors.with_opacity(0.3, color))
                        )
                    ],
                    width=380
                ),
                ft.Text(format_currency(value), color=PALETTE.ink, size=12, text_align=ft.TextAlign.RIGHT, expand=True)
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )
        chart_groups.append(bar_row)
        
    return ft.Container(
        content=ft.Column(controls=chart_groups, spacing=14, alignment=ft.MainAxisAlignment.CENTER),
        padding=ft.padding.only(left=18, right=18, top=20, bottom=20),
        height=220, bgcolor=PALETTE.surface, border_radius=18
    )

def build_daily_heatmap_chart(items: list[Any]) -> ft.Control:
    # A simple heatmap visualization in Flet for now, since it requires custom grids.
    if not items or not any(item.amount > 0 for item in items):
        return build_empty_chart("Sin gasto diario para mapa de calor")
        
    def _heatmap_color(ratio: float) -> str:
        if ratio <= 0: return "#15242D"
        if ratio < 0.18: return "#1D3A37"
        if ratio < 0.38: return "#31574B"
        if ratio < 0.62: return "#806337"
        if ratio < 0.82: return "#B76548"
        return "#F06A63"
        
    # Since Heatmap is hard to represent purely with BarCharts, 
    # we use a wrapped Row of small containers mimicking GitHub's commit graph.
    cells = []
    for item in items:
        color = _heatmap_color(item.ratio)
        cells.append(ft.Container(
            width=15, height=15, bgcolor=color, border_radius=4,
            border=ft.border.all(1, PALETTE.line),
            tooltip=f"{item.date.strftime('%Y-%m-%d')}\\n{format_currency(item.amount)}"
        ))
        
    return ft.Container(
        content=ft.Row(controls=cells, wrap=True, alignment=ft.MainAxisAlignment.CENTER),
        padding=20, height=220, bgcolor=PALETTE.surface, border_radius=18,
        alignment=ft.alignment.center
    )
