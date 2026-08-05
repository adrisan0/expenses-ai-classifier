"""Tests for audit metrics and review queue helpers."""
from __future__ import annotations

import json

import pandas as pd

from src.core.insights import (
    build_average_period_health,
    build_bizum_offset_candidates,
    build_cashflow_nature_summary,
    build_bizum_intent_summary,
    build_class_bizum_income_mask,
    build_deepseek_agent_context,
    classify_cashflow_nature,
    build_deepseek_expense_context,
    build_audit_metrics,
    filter_dataframe_to_month,
    build_monthly_health,
    build_monthly_flow,
    build_personal_finance_summary,
    build_prioritized_review_queue,
    build_recurring_expense_patterns,
    build_salary_income_mask,
    build_sector_spend,
    build_shared_bizum_reimbursement_mask,
    build_spend_breakdown,
    build_time_display_mode,
    scale_spend_breakdown,
    summarize_ai_trace,
)


def test_build_audit_metrics_counts_sources_and_cache_hits() -> None:
    dataframe = pd.DataFrame(
        {
            "Grupo": ["Supermercado", "Restaurante", "", "Revision manual"],
            "CategoriaLeaf": [
                "Supermercado",
                "Restaurante",
                pd.NA,
                "Revision manual",
            ],
            "CategoriaFuente": [
                "regla_local",
                "ia_arbol:Root",
                pd.NA,
                "ia_arbol:fallback",
            ],
            "CategoriaConfianza": [1.0, 0.72, pd.NA, 0.0],
            "CategoriaTrazaIA": [
                pd.NA,
                json.dumps([{"nodo": "Root", "eleccion": "Ocio", "cache": True}]),
                pd.NA,
                json.dumps([{"nodo": "Root", "eleccion": "__revision_manual__"}]),
            ],
        }
    )

    metrics = build_audit_metrics(dataframe)

    assert metrics.total == 4
    assert metrics.categorized == 3
    assert metrics.pending == 1
    assert metrics.review == 2
    assert metrics.local_rules == 1
    assert metrics.ai == 2
    assert metrics.manual == 1
    assert metrics.cache_hits == 1


def test_build_prioritized_review_queue_orders_risk_before_amount() -> None:
    dataframe = pd.DataFrame(
        {
            "Importe": [-10.0, -500.0, -20.0],
            "Grupo": ["Supermercado", "Restaurante", "Revision manual"],
            "CategoriaLeaf": ["Supermercado", "Restaurante", "Revision manual"],
            "CategoriaFuente": ["ia_arbol:Root", "ia_arbol:Root", "ia_arbol:fallback"],
            "CategoriaConfianza": [0.20, 0.40, 0.0],
        },
        index=[10, 20, 30],
    )

    queue_frame = build_prioritized_review_queue(dataframe, confidence_threshold=0.65)

    assert queue_frame.index.tolist() == [30, 10, 20]


def test_empty_metrics_and_trace_summary_are_stable() -> None:
    metrics = build_audit_metrics(pd.DataFrame())

    assert metrics.total == 0
    assert metrics.coverage_ratio == 0.0
    assert summarize_ai_trace(pd.NA) == ""
    assert "Root -> Alimentacion" in summarize_ai_trace(
        json.dumps(
            [
                {
                    "nodo": "Root",
                    "eleccion": "Alimentacion",
                    "confianza": 0.91,
                    "motivo": "compra",
                    "cache": True,
                }
            ]
        )
    )


def test_personal_finance_summary_detects_payroll_and_class_bizums() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-02-01",
                ]
            ),
            "Concepto": [
                "Transferencia nomina empresa",
                "Bizum clase Marta",
                "Bizum cena amigos",
                "Bizum recibido",
                "Mercadona",
            ],
            "Movimiento": ["TRANSFERENCIA", "BIZUM", "BIZUM", "BIZUM", "COMPRA"],
            "Importe": [1000.0, 18.0, 32.5, 27.0, -100.0],
            "CategoriaLeaf": [
                "Nomina",
                "Bizum recibido",
                "Bizum recibido",
                "Bizum recibido",
                "Super",
            ],
        }
    )

    salary_mask = build_salary_income_mask(dataframe)
    class_mask = build_class_bizum_income_mask(dataframe)
    shared_mask = build_shared_bizum_reimbursement_mask(dataframe)
    summary = build_personal_finance_summary(dataframe)

    assert salary_mask.tolist() == [True, False, False, False, False]
    assert class_mask.tolist() == [False, True, False, True, False]
    assert shared_mask.tolist() == [False, False, True, False, False]
    assert summary.months == 2
    assert summary.expense_total == 100.0
    assert summary.class_income == 45.0
    assert summary.shared_bizum_reimbursements == 32.5
    assert summary.adjusted_expense_total == 67.5
    assert summary.salary_period == 1000.0
    assert summary.spend_vs_income == 977.5


def test_spend_breakdown_and_deepseek_context_are_compact() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "Concepto": ["Restaurante A", "Super B", "Bizum clase"],
            "Movimiento": ["TARJETA", "TARJETA", "BIZUM"],
            "Importe": [-40.0, -60.0, 18.0],
            "CategoriaLeaf": ["Restaurante", "Supermercado", "Bizum recibido"],
            "CategoriaPath": [
                "Alimentacion > Restaurante",
                "Alimentacion > Supermercado",
                "Transferencias > Bizum recibido",
            ],
        }
    )

    breakdown = build_spend_breakdown(dataframe, group_col="CategoriaLeaf")
    context = build_deepseek_expense_context(
        dataframe,
        group_col="CategoriaLeaf",
        confirmed_context="Viaje confirmado a Lisboa el 2026-01-06.",
    )

    assert [item.label for item in breakdown] == ["Supermercado", "Restaurante"]
    assert context["finanzas"]["gasto_total"] == 100.0
    assert context["finanzas"]["ingresos_clases_bizum"] == 18.0
    assert context["finanzas"]["ingreso_real_nomina_y_clases"] == 18.0
    assert len(context["top_movimientos_gasto"]) == 2
    assert context["memoria_confirmada_usuario"].startswith("Viaje confirmado")
    assert context["patrones_fechas"]
    assert "top_categorias_gasto" in context
    assert "patrones_recurrentes" in context
    assert "bizums_que_atenuan_gastos_compartidos" in context


def test_recurring_patterns_and_bizum_offsets_detect_shared_expense() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-03-01",
                    "2026-03-10",
                    "2026-03-10",
                    "2026-03-11",
                ]
            ),
            "Concepto": [
                "Netflix",
                "Netflix",
                "Netflix",
                "Cena grupo",
                "Compra generica",
                "Bizum cena Ana",
            ],
            "Movimiento": [
                "TARJETA",
                "TARJETA",
                "TARJETA",
                "TARJETA",
                "TARJETA",
                "BIZUM",
            ],
            "Importe": [-12.99, -12.99, -12.99, -120.0, -200.0, 40.0],
            "CategoriaLeaf": [
                "Streaming",
                "Streaming",
                "Streaming",
                "Restaurante",
                "Compras",
                "Bizum recibido",
            ],
        }
    )

    patterns = build_recurring_expense_patterns(dataframe, group_col="CategoriaLeaf")
    offsets = build_bizum_offset_candidates(dataframe, group_col="CategoriaLeaf")
    context = build_deepseek_expense_context(dataframe, group_col="CategoriaLeaf")

    assert patterns[0].label == "Streaming"
    assert patterns[0].kind == "subscripcion"
    assert offsets[0].bizum_amount == 40.0
    assert offsets[0].expense_amount == 120.0
    assert offsets[0].expense_concept == "Cena grupo"
    assert offsets[0].net_expense_estimate == 80.0
    assert context["finanzas"]["gasto_ajustado_por_bizums_compartidos"] == 318.97


def test_deepseek_agent_context_includes_visible_movements() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-07", "2026-01-06", "2026-01-05"]),
            "Concepto": ["Restaurante A", "Cuenta remunerada pibank", "Nomina empresa"],
            "Movimiento": ["TARJETA", "Transferencia realizada", "TRANSFERENCIA"],
            "Importe": [-40.0, -300.0, 1200.0],
            "CategoriaLeaf": ["Restaurante", "Ahorro", "Nomina"],
            "CategoriaPath": [
                "Ocio > Restaurante",
                "Ahorro > Cuenta remunerada",
                "Ingreso > Nomina",
            ],
        }
    )

    context = build_deepseek_agent_context(
        dataframe,
        group_col="CategoriaLeaf",
        confirmed_context="Viaje confirmado.",
        movement_limit=2,
    )

    assert context["movimientos_visibles_total"] == 3
    assert context["movimientos_visibles_truncados"] == 1
    assert len(context["movimientos_visibles"]) == 2
    assert context["movimientos_visibles"][0]["fecha"] == "2026-01-07"
    assert context["movimientos_visibles"][1]["naturaleza_financiera"] == "ahorro"
    assert context["movimientos_visibles"][1]["categoria_actual"] == "Ahorro > Cuenta remunerada"


def test_time_display_mode_scales_spend_without_changing_ratios() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-31"]),
            "Concepto": ["Super", "Restaurante"],
            "Movimiento": ["TARJETA", "TARJETA"],
            "Importe": [-62.0, -31.0],
            "CategoriaLeaf": ["Supermercado", "Restaurante"],
        }
    )

    mode = build_time_display_mode(dataframe, "daily")
    breakdown = build_spend_breakdown(dataframe, group_col="CategoriaLeaf")
    scaled = scale_spend_breakdown(breakdown, divisor=mode.divisor)

    assert mode.label == "Gasto diario medio"
    assert mode.divisor == 31.0
    assert mode.suffix == " / dia"
    assert scaled[0].amount == 2.0
    assert scaled[0].ratio == breakdown[0].ratio


def test_time_display_mode_supports_month_average_and_selected_month() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-15"]),
            "Importe": [-10.0, -20.0, -30.0],
        }
    )

    average_mode = build_time_display_mode(dataframe, "monthly_average")
    selected_mode = build_time_display_mode(
        dataframe,
        "monthly_selected",
        target_month="2026-02",
    )

    assert average_mode.label == "Gasto mensual medio"
    assert average_mode.divisor == 2.0
    assert selected_mode.label == "Mes concreto: 2026-02"
    assert selected_mode.divisor == 1.0
    assert selected_mode.month == "2026-02"


def test_time_display_mode_supports_seasonal_average() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-07-01"]),
            "Importe": [-10.0, -20.0, -30.0],
        }
    )

    mode = build_time_display_mode(dataframe, "seasonal")

    assert mode.label == "Gasto estacional medio"
    assert mode.divisor == 3.0
    assert mode.suffix == " / estacion"


def test_filter_dataframe_to_month_keeps_only_selected_month() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-15"]),
            "Importe": [-10.0, -20.0, -30.0],
        }
    )

    filtered = filter_dataframe_to_month(dataframe, "2026-02")

    assert len(filtered) == 2
    assert filtered["Importe"].tolist() == [-20.0, -30.0]


def test_monthly_health_compares_latest_month_against_recent_average() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-03",
                    "2026-02-01",
                    "2026-02-03",
                    "2026-03-01",
                    "2026-03-03",
                ]
            ),
            "Concepto": ["Nomina", "Super", "Nomina", "Super", "Nomina", "Restaurante"],
            "Movimiento": ["TRANSFER", "CARD", "TRANSFER", "CARD", "TRANSFER", "CARD"],
            "Importe": [1000.0, -500.0, 1000.0, -700.0, 1000.0, -900.0],
        }
    )

    health = build_monthly_health(dataframe, baseline_months=2)

    assert health.month == "2026-03"
    assert health.current_income == 1000.0
    assert health.current_expense == 900.0
    assert health.average_expense == 600.0
    assert health.expense_delta == 300.0
    assert health.alert_label == "Gasto por encima"


def test_average_period_health_uses_whole_selected_period() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(
                ["2026-01-01", "2026-01-03", "2026-02-01", "2026-02-03"]
            ),
            "Importe": [1000.0, -500.0, 1000.0, -700.0],
        }
    )

    health = build_average_period_health(dataframe)

    assert health.month is None
    assert health.comparison_months == 2
    assert health.current_income == 1000.0
    assert health.current_expense == 600.0
    assert health.current_margin == 400.0
    assert health.alert_label in {"Ahorro medio sano", "Media del periodo"}


def test_sector_spend_uses_level_one_categories_and_recent_average() -> None:
    current = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
            "Concepto": ["Super", "Restaurante", "Taxi"],
            "Movimiento": ["CARD", "CARD", "CARD"],
            "Importe": [-120.0, -80.0, -50.0],
            "CategoriaNivel1": ["Alimentacion", "Alimentacion", "Transporte"],
        }
    )
    baseline = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-02"]),
            "Concepto": ["Super", "Super", "Taxi"],
            "Movimiento": ["CARD", "CARD", "CARD"],
            "Importe": [-100.0, -140.0, -40.0],
            "CategoriaNivel1": ["Alimentacion", "Alimentacion", "Transporte"],
        }
    )

    sectors = build_sector_spend(current, baseline_frame=baseline)

    assert [sector.label for sector in sectors] == ["Alimentacion", "Transporte"]
    assert sectors[0].amount == 200.0
    assert sectors[0].average_amount == 120.0
    assert sectors[0].delta_amount == 80.0
    assert sectors[0].count == 2


def test_bizum_intent_summary_uses_memory_for_shared_events() -> None:
    dataframe = pd.DataFrame(
        {
            "Concepto": [
                "Bizum clase Marta",
                "Bizum cena grupo",
                "Bizum enviado cena",
            ],
            "Movimiento": ["BIZUM", "BIZUM", "BIZUM"],
            "Importe": [18.0, 42.0, -21.0],
        }
    )

    summary = build_bizum_intent_summary(
        dataframe,
        confirmed_context="Cena de grupo confirmada el 2026-03-02.",
    )

    assert summary.class_income == 18.0
    assert summary.shared_reimbursements == 42.0
    assert summary.contextual_shared_reimbursements == 42.0
    assert summary.outgoing_payments == 21.0
    assert summary.memory_used is True


def test_cashflow_nature_distinguishes_income_consumption_saving_and_investment() -> None:
    dataframe = pd.DataFrame(
        {
            "Concepto": [
                "Nomina empresa",
                "Transferencia dudosa",
                "Mercadona",
                "Traspaso ahorro",
                "Indexa acciones",
            ],
            "Movimiento": [
                "TRANSFERENCIA",
                "TRANSFERENCIA",
                "TARJETA",
                "TRANSFERENCIA",
                "ADEUDO",
            ],
            "Importe": [1200.0, 80.0, -55.0, -300.0, -250.0],
        }
    )

    nature = classify_cashflow_nature(dataframe)
    summary = build_cashflow_nature_summary(dataframe)
    context = build_deepseek_expense_context(dataframe)

    assert nature.tolist() == ["ingreso", "ingreso", "consumo", "ahorro", "inversion"]
    assert summary.income == 1200.0
    assert summary.expense == 605.0
    assert summary.consumption == 55.0
    assert summary.saving == 300.0
    assert summary.investment == 250.0
    assert context["naturaleza_financiera"]["consumo"] == 55.0
    assert context["naturaleza_financiera"]["ahorro"] == 300.0
    assert context["naturaleza_financiera"]["inversion"] == 250.0


def test_cashflow_nature_treats_pibank_transfers_as_saving() -> None:
    dataframe = pd.DataFrame(
        {
            "Concepto": ["Transferencia realizada", "Transferencia recibida"],
            "Movimiento": ["Cuenta remunerada pibank", "Transferencia pibank"],
            "Importe": [-1200.0, 400.0],
        }
    )

    nature = classify_cashflow_nature(dataframe)
    summary = build_cashflow_nature_summary(dataframe)

    assert nature.tolist() == ["ahorro", "ingreso"]
    assert summary.saving == 1200.0


def test_personal_finance_summary_excludes_saving_from_expense_total() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "Concepto": ["Nomina", "Mercadona", "Transferencia realizada"],
            "Movimiento": ["TRANSFERENCIA", "TARJETA", "Cuenta remunerada pibank"],
            "Importe": [2000.0, -400.0, -900.0],
        }
    )

    summary = build_personal_finance_summary(dataframe)
    health = build_monthly_health(dataframe, baseline_months=1)
    monthly_flow = build_monthly_flow(dataframe, limit=1)
    breakdown = build_spend_breakdown(dataframe, group_col="Concepto")

    assert summary.expense_total == 400.0
    assert summary.spend_vs_income == 1600.0
    assert health.current_expense == 400.0
    assert health.current_margin == 1600.0
    assert monthly_flow[0].expenses == 400.0
    assert monthly_flow[0].saving == 900.0
    assert monthly_flow[0].investment == 0.0
    assert [item.label for item in breakdown] == ["Mercadona"]


def test_deepseek_context_reports_deficit_without_negative_savings() -> None:
    dataframe = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Concepto": ["Nomina", "Restaurante"],
            "Movimiento": ["TRANSFERENCIA", "TARJETA"],
            "Importe": [1000.0, -1200.0],
        }
    )

    context = build_deepseek_expense_context(dataframe)

    assert context["finanzas"]["margen_disponible_estimado"] is None
    assert context["finanzas"]["ratio_ahorro_estimado"] is None
    assert context["finanzas"]["deficit_estimado"] == 200.0
    assert context["reglas_calculo"]["ahorro"] == (
        "Ahorro es dinero movido a ahorro. Nunca debe ser negativo."
    )
