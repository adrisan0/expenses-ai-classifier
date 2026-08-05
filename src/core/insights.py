"""Audit metrics, personal finance summaries and review prioritization."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.review import DEFAULT_REVIEW_CONFIDENCE, build_pending_mask, build_review_mask


@dataclass(frozen=True, slots=True)
class AuditMetrics:
    """High-level audit counters for the current categorization state."""

    total: int
    categorized: int
    pending: int
    review: int
    manual: int
    local_rules: int
    ai: int
    cache_hits: int
    average_confidence: float | None

    @property
    def coverage_ratio(self) -> float:
        """Return categorized / total."""
        if self.total == 0:
            return 0.0
        return self.categorized / self.total

    @property
    def review_ratio(self) -> float:
        """Return review / total."""
        if self.total == 0:
            return 0.0
        return self.review / self.total


@dataclass(frozen=True, slots=True)
class PersonalFinanceSummary:
    """Compact money summary for the current expense view."""

    total_movements: int
    months: int
    expense_total: float
    adjusted_expense_total: float
    bank_income_total: float
    bank_cashflow: float
    real_income_total: float
    salary_period: float | None
    class_income: float
    shared_bizum_reimbursements: float
    other_positive_income: float
    spend_vs_income: float | None
    savings_rate: float | None


@dataclass(frozen=True, slots=True)
class MonthlyHealth:
    """Health snapshot for the active month against recent history."""

    month: str | None
    current_income: float
    current_expense: float
    current_margin: float
    savings_rate: float | None
    comparison_months: int
    average_income: float
    average_expense: float
    average_margin: float
    expense_delta: float
    margin_delta: float
    alert_label: str
    alert_detail: str


@dataclass(frozen=True, slots=True)
class SpendBreakdownItem:
    """One ranked expense bucket."""

    label: str
    amount: float
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class SectorSpendItem:
    """One top-level category sector with recent-month comparison."""

    label: str
    amount: float
    average_amount: float
    delta_amount: float
    delta_ratio: float | None
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class MonthlyFlowItem:
    """Monthly aggregate used by the visual dashboard."""

    month: str
    expenses: float
    class_income: float
    shared_bizum_reimbursements: float
    bank_cashflow: float
    salary: float | None
    saving: float = 0.0
    investment: float = 0.0


@dataclass(frozen=True, slots=True)
class SavingsOpportunity:
    """Simple deterministic hint for high-impact expense buckets."""

    label: str
    amount: float
    ratio: float
    message: str


@dataclass(frozen=True, slots=True)
class TimeDisplayMode:
    """How totals should be displayed over the selected source range."""

    key: str
    label: str
    divisor: float
    suffix: str
    month: str | None = None


@dataclass(frozen=True, slots=True)
class BizumIntentSummary:
    """Deterministic Bizum intent split used before asking the LLM."""

    class_income: float
    shared_reimbursements: float
    contextual_shared_reimbursements: float
    ambiguous_income: float
    outgoing_payments: float
    memory_used: bool


@dataclass(frozen=True, slots=True)
class CashflowNatureSummary:
    """High-level financial nature split independent from taxonomy."""

    # Real income uses the same definition as Vision KPIs: payroll plus class Bizums.
    income: float
    expense: float
    consumption: float
    saving: float
    investment: float
    shared_reimbursements: float = 0.0


@dataclass(frozen=True, slots=True)
class RecurringExpensePattern:
    """Repeated expense pattern useful for DeepSeek analysis."""

    label: str
    kind: str
    total_amount: float
    average_amount: float
    months: int
    movements: int
    examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BizumOffsetCandidate:
    """Incoming Bizum that likely attenuates a nearby shared expense."""

    bizum_date: str
    bizum_amount: float
    expense_date: str
    expense_amount: float
    net_expense_estimate: float
    expense_concept: str
    expense_category: str
    days_delta: int


def build_monthly_health(
    dataframe: pd.DataFrame,
    *,
    target_month: str | None = None,
    baseline_months: int = 3,
) -> MonthlyHealth:
    """Compare the active month with the previous available months."""
    empty = MonthlyHealth(
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
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return empty

    working = dataframe.copy()
    working["_month"] = pd.to_datetime(working["Fecha"], errors="coerce").dt.to_period("M")
    working = working.dropna(subset=["_month"])
    if working.empty:
        return empty

    months = sorted(working["_month"].unique())
    selected_month = pd.Period(target_month, freq="M") if target_month else months[-1]
    current = working.loc[working["_month"] == selected_month].copy()
    if current.empty:
        return empty

    previous_months = [month for month in months if month < selected_month][-baseline_months:]
    baseline = working.loc[working["_month"].isin(previous_months)].copy()

    current_totals = _income_expense_margin(current)
    monthly_baseline = [
        _income_expense_margin(baseline.loc[baseline["_month"] == month])
        for month in previous_months
    ]
    average_income = _average_dict_value(monthly_baseline, "income")
    average_expense = _average_dict_value(monthly_baseline, "expense")
    average_margin = _average_dict_value(monthly_baseline, "margin")
    expense_delta = current_totals["expense"] - average_expense
    margin_delta = current_totals["margin"] - average_margin
    savings_rate = (
        current_totals["margin"] / current_totals["income"]
        if current_totals["income"] > 0
        else None
    )
    alert_label, alert_detail = _build_health_alert(
        current_expense=current_totals["expense"],
        average_expense=average_expense,
        current_margin=current_totals["margin"],
        average_margin=average_margin,
        savings_rate=savings_rate,
        comparison_months=len(previous_months),
    )

    return MonthlyHealth(
        month=str(selected_month),
        current_income=current_totals["income"],
        current_expense=current_totals["expense"],
        current_margin=current_totals["margin"],
        savings_rate=savings_rate,
        comparison_months=len(previous_months),
        average_income=average_income,
        average_expense=average_expense,
        average_margin=average_margin,
        expense_delta=expense_delta,
        margin_delta=margin_delta,
        alert_label=alert_label,
        alert_detail=alert_detail,
    )


def build_average_period_health(dataframe: pd.DataFrame) -> MonthlyHealth:
    """Build an average monthly health snapshot for the whole selected period."""
    empty = MonthlyHealth(
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
        alert_detail="No hay movimientos fechados para calcular la media mensual.",
    )
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return empty

    months = max(_count_months(dataframe), 0)
    if months == 0:
        return empty

    totals = _income_expense_margin(dataframe)
    average_income = totals["income"] / months
    average_expense = totals["expense"] / months
    average_margin = totals["margin"] / months
    savings_rate = average_margin / average_income if average_income > 0 else None

    if savings_rate is not None and savings_rate >= 0.20:
        alert_label = "Ahorro medio sano"
        alert_detail = "La media mensual del periodo supera el 20% de los ingresos."
    else:
        alert_label = "Media del periodo"
        alert_detail = "Mostrando el promedio mensual del rango global filtrado."

    return MonthlyHealth(
        month=None,
        current_income=average_income,
        current_expense=average_expense,
        current_margin=average_margin,
        savings_rate=savings_rate,
        comparison_months=months,
        average_income=average_income,
        average_expense=average_expense,
        average_margin=average_margin,
        expense_delta=0.0,
        margin_delta=0.0,
        alert_label=alert_label,
        alert_detail=alert_detail,
    )


def build_audit_metrics(
    dataframe: pd.DataFrame,
    *,
    confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE,
) -> AuditMetrics:
    """Build audit metrics from a dataframe with optional category columns."""
    total = len(dataframe)
    if total == 0:
        return AuditMetrics(0, 0, 0, 0, 0, 0, 0, 0, None)

    pending_mask = build_pending_mask(dataframe)
    review_mask = build_review_mask(
        dataframe,
        confidence_threshold=confidence_threshold,
    )
    source = _string_column(dataframe, "CategoriaFuente")
    leaf = _string_column(dataframe, "CategoriaLeaf")
    confidence = pd.to_numeric(
        dataframe.get("CategoriaConfianza", pd.Series(index=dataframe.index)),
        errors="coerce",
    )

    categorized = int((~pending_mask).sum())
    confidence_values = confidence.dropna()
    average_confidence = (
        float(confidence_values.mean()) if not confidence_values.empty else None
    )

    return AuditMetrics(
        total=total,
        categorized=categorized,
        pending=int(pending_mask.sum()),
        review=int(review_mask.sum()),
        manual=int(leaf.eq("Revision manual").sum()),
        local_rules=int(source.eq("regla_local").sum()),
        ai=int(source.str.startswith("ia_arbol", na=False).sum()),
        cache_hits=count_trace_cache_hits(dataframe),
        average_confidence=average_confidence,
    )


def build_time_display_mode(
    dataframe: pd.DataFrame,
    mode: str,
    *,
    target_month: str | None = None,
) -> TimeDisplayMode:
    """Build the divisor used to display totals or time averages."""
    normalized = (
        mode
        if mode
        in {"total", "daily", "monthly_average", "monthly_selected", "seasonal", "yearly"}
        else "total"
    )
    if normalized == "total":
        return TimeDisplayMode("total", "Total del rango", 1.0, "")
    if normalized == "monthly_selected":
        return TimeDisplayMode(
            "monthly_selected",
            f"Mes concreto: {target_month}" if target_month else "Mes concreto",
            1.0,
            "",
            month=target_month,
        )

    units = _count_time_units(dataframe, normalized)
    divisor = float(max(units, 1))
    labels = {
        "daily": ("Gasto diario medio", " / dia"),
        "monthly_average": ("Gasto mensual medio", " / mes"),
        "seasonal": ("Gasto estacional medio", " / estacion"),
        "yearly": ("Gasto anual medio", " / ano"),
    }
    label, suffix = labels[normalized]
    return TimeDisplayMode(normalized, label, divisor, suffix)


def filter_dataframe_to_month(
    dataframe: pd.DataFrame,
    target_month: str | None,
) -> pd.DataFrame:
    """Return only rows for the selected YYYY-MM month."""
    if dataframe.empty or target_month is None or "Fecha" not in dataframe.columns:
        return dataframe.copy()
    dates = pd.to_datetime(dataframe["Fecha"], errors="coerce")
    mask = dates.dt.to_period("M").astype(str) == target_month
    return dataframe.loc[mask].copy()


def build_personal_finance_summary(
    dataframe: pd.DataFrame,
) -> PersonalFinanceSummary:
    """Summarize expenses, detected payroll and special Bizum inflows."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return PersonalFinanceSummary(
            total_movements=0,
            months=0,
            expense_total=0.0,
            adjusted_expense_total=0.0,
            bank_income_total=0.0,
            bank_cashflow=0.0,
            real_income_total=0.0,
            salary_period=None,
            class_income=0.0,
            shared_bizum_reimbursements=0.0,
            other_positive_income=0.0,
            spend_vs_income=None,
            savings_rate=None,
        )

    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(dataframe)
    positive_mask = amounts > 0
    salary_mask = build_salary_income_mask(dataframe)
    class_mask = build_class_bizum_income_mask(dataframe)
    shared_bizum_mask = build_shared_bizum_reimbursement_mask(dataframe)

    months = _count_months(dataframe)
    expense_total = float(amounts.loc[(amounts < 0) & nature.eq("consumo")].abs().sum())
    bank_income_total = float(amounts.loc[positive_mask].sum())
    salary_total = float(amounts.loc[salary_mask].sum())
    class_income = float(amounts.loc[class_mask].sum())
    shared_bizum = float(amounts.loc[shared_bizum_mask].sum())
    other_positive = float(
        amounts.loc[positive_mask & ~salary_mask & ~class_mask & ~shared_bizum_mask].sum()
    )
    salary_period = salary_total if salary_total > 0 else None
    income_basis = (salary_period if salary_period is not None else 0.0) + class_income
    adjusted_expense_total = max(expense_total - shared_bizum, 0.0)
    spend_vs_income = (
        income_basis - adjusted_expense_total if salary_period is not None else None
    )
    savings_rate = None
    if spend_vs_income is not None and income_basis > 0:
        savings_rate = spend_vs_income / income_basis

    return PersonalFinanceSummary(
        total_movements=len(dataframe),
        months=months,
        expense_total=expense_total,
        adjusted_expense_total=adjusted_expense_total,
        bank_income_total=bank_income_total,
        bank_cashflow=float(amounts.sum()),
        real_income_total=income_basis,
        salary_period=salary_period,
        class_income=class_income,
        shared_bizum_reimbursements=shared_bizum,
        other_positive_income=other_positive,
        spend_vs_income=spend_vs_income,
        savings_rate=savings_rate,
    )


def build_spend_breakdown(
    dataframe: pd.DataFrame,
    *,
    group_col: str = "CategoriaLeaf",
    limit: int = 8,
) -> list[SpendBreakdownItem]:
    """Return largest consumption buckets as positive amounts."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return []
    effective_group = group_col if group_col in dataframe.columns else "Concepto"
    if effective_group not in dataframe.columns:
        return []

    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(dataframe)
    expenses = dataframe.loc[(amounts < 0) & nature.eq("consumo")].copy()
    if expenses.empty:
        return []
    expenses["_expense_amount"] = amounts.loc[expenses.index].abs()
    labels = expenses[effective_group].fillna("Sin categoria").astype(str).str.strip()
    expenses["_expense_label"] = labels.replace("", "Sin categoria")

    grouped = (
        expenses.groupby("_expense_label")
        .agg(amount=("_expense_amount", "sum"), count=("_expense_amount", "size"))
        .sort_values(["amount", "count"], ascending=[False, False])
        .head(limit)
    )
    total = float(expenses["_expense_amount"].sum())
    return [
        SpendBreakdownItem(
            label=str(index),
            amount=float(row["amount"]),
            count=int(row["count"]),
            ratio=float(row["amount"]) / total if total else 0.0,
        )
        for index, row in grouped.iterrows()
    ]


def build_sector_spend(
    dataframe: pd.DataFrame,
    *,
    baseline_frame: pd.DataFrame | None = None,
    group_col: str = "CategoriaNivel1",
    limit: int = 8,
) -> list[SectorSpendItem]:
    """Return top-level expense sectors with optional monthly baseline."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return []
    effective_group = _effective_group_column(dataframe, group_col)
    if effective_group is None:
        return []

    current_items = build_spend_breakdown(
        dataframe,
        group_col=effective_group,
        limit=limit,
    )
    if not current_items:
        return []

    baseline_by_label: dict[str, float] = {}
    if baseline_frame is not None and not baseline_frame.empty:
        baseline_group = _effective_group_column(baseline_frame, group_col)
        if baseline_group is not None:
            baseline_by_label = _average_monthly_expense_by_group(
                baseline_frame,
                group_col=baseline_group,
            )

    return [
        SectorSpendItem(
            label=item.label,
            amount=item.amount,
            average_amount=baseline_by_label.get(item.label, 0.0),
            delta_amount=item.amount - baseline_by_label.get(item.label, 0.0),
            delta_ratio=(
                (item.amount - baseline_by_label[item.label]) / baseline_by_label[item.label]
                if baseline_by_label.get(item.label, 0.0) > 0
                else None
            ),
            count=item.count,
            ratio=item.ratio,
        )
        for item in current_items
    ]


def build_monthly_flow(
    dataframe: pd.DataFrame,
    *,
    limit: int = 12,
) -> list[MonthlyFlowItem]:
    """Build month-level aggregates for compact visual diagrams."""
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return []

    working = dataframe.copy()
    working["_month"] = pd.to_datetime(working["Fecha"], errors="coerce").dt.to_period("M")
    working = working.dropna(subset=["_month"])
    if working.empty:
        return []

    amounts = pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(working)
    salary_mask = build_salary_income_mask(working)
    class_mask = build_class_bizum_income_mask(working)
    shared_bizum_mask = build_shared_bizum_reimbursement_mask(working)
    items: list[MonthlyFlowItem] = []
    for month, group in working.groupby("_month", sort=True):
        group_amounts = amounts.loc[group.index]
        items.append(
            MonthlyFlowItem(
                month=str(month),
                expenses=float(
                    group_amounts.loc[
                        (group_amounts < 0) & nature.loc[group.index].eq("consumo")
                    ].abs().sum()
                ),
                class_income=float(group_amounts.loc[class_mask.loc[group.index]].sum()),
                shared_bizum_reimbursements=float(
                    group_amounts.loc[shared_bizum_mask.loc[group.index]].sum()
                ),
                bank_cashflow=float(group_amounts.sum()),
                salary=float(group_amounts.loc[salary_mask.loc[group.index]].sum())
                or None,
                saving=float(
                    group_amounts.loc[
                        (group_amounts < 0) & nature.loc[group.index].eq("ahorro")
                    ].abs().sum()
                ),
                investment=float(
                    group_amounts.loc[
                        (group_amounts < 0) & nature.loc[group.index].eq("inversion")
                    ].abs().sum()
                ),
            )
        )
    return items[-limit:]


def scale_spend_breakdown(
    items: list[SpendBreakdownItem],
    *,
    divisor: float,
) -> list[SpendBreakdownItem]:
    """Scale spend amounts while preserving counts and relative ratios."""
    safe_divisor = max(float(divisor), 1.0)
    return [
        SpendBreakdownItem(
            label=item.label,
            amount=item.amount / safe_divisor,
            count=item.count,
            ratio=item.ratio,
        )
        for item in items
    ]


def build_savings_opportunities(
    dataframe: pd.DataFrame,
    *,
    group_col: str = "CategoriaLeaf",
    limit: int = 4,
) -> list[SavingsOpportunity]:
    """Suggest deterministic high-impact areas to inspect first."""
    breakdown = build_spend_breakdown(dataframe, group_col=group_col, limit=limit)
    opportunities: list[SavingsOpportunity] = []
    for item in breakdown:
        if item.ratio < 0.05:
            continue
        target = item.amount * 0.10
        opportunities.append(
            SavingsOpportunity(
                label=item.label,
                amount=item.amount,
                ratio=item.ratio,
                message=(
                    f"Revisar {item.label}: un recorte del 10% liberaria "
                    f"{target:.2f} EUR en esta vista."
                ),
            )
        )
    return opportunities


def build_deepseek_expense_context(
    dataframe: pd.DataFrame,
    *,
    group_col: str = "CategoriaLeaf",
    confirmed_context: str = "",
) -> dict[str, Any]:
    """Build a compact JSON-safe context for expense Q&A."""
    finance = build_personal_finance_summary(dataframe)
    cashflow_nature = build_cashflow_nature_summary(dataframe)
    top_categories = build_spend_breakdown(dataframe, group_col=group_col, limit=8)
    monthly_flow = build_monthly_flow(dataframe, limit=6)
    top_movements = _build_top_expense_movements(dataframe, limit=12)
    date_patterns = _build_date_pattern_context(dataframe, group_col=group_col, limit=8)
    recurring_patterns = build_recurring_expense_patterns(
        dataframe,
        group_col=group_col,
        limit=10,
    )
    bizum_offsets = build_bizum_offset_candidates(dataframe, group_col=group_col, limit=10)
    bizum_intent = build_bizum_intent_summary(
        dataframe,
        confirmed_context=confirmed_context,
    )
    detected_deficit = (
        abs(finance.spend_vs_income)
        if finance.spend_vs_income is not None and finance.spend_vs_income < 0
        else 0.0
    )
    positive_margin = (
        finance.spend_vs_income
        if finance.spend_vs_income is not None and finance.spend_vs_income >= 0
        else None
    )
    positive_savings_rate = (
        finance.savings_rate
        if finance.savings_rate is not None and finance.savings_rate >= 0
        else None
    )
    return {
        "periodo": _date_range_context(dataframe),
        "memoria_confirmada_usuario": confirmed_context.strip(),
        "reglas_calculo": {
            "signo_importe": "Importe positivo es entrada; importe negativo es salida.",
            "gasto_total": "Solo consumo real; excluye transferencias a ahorro e inversion.",
            "gasto_ajustado": (
                "Resta Bizums compartidos al consumo para estimar gasto propio. "
                "No los metas como ingreso neto."
            ),
            "ingreso_real": (
                "Ingreso real de input es nomina detectada y Bizums de clases "
                "particulares; otros positivos son entradas a revisar."
            ),
            "ahorro": "Ahorro es dinero movido a ahorro. Nunca debe ser negativo.",
            "deficit": "Si el margen es negativo, llamalo deficit, no ahorro negativo.",
        },
        "finanzas": {
            "movimientos": finance.total_movements,
            "meses": finance.months,
            "gasto_total": round(finance.expense_total, 2),
            "gasto_ajustado_por_bizums_compartidos": round(
                finance.adjusted_expense_total,
                2,
            ),
            "ingresos_banco": round(finance.bank_income_total, 2),
            "ingreso_real_nomina_y_clases": round(finance.real_income_total, 2),
            "balance_banco": round(finance.bank_cashflow, 2),
            "salario_periodo": _round_optional(finance.salary_period),
            "ingresos_clases_bizum": round(finance.class_income, 2),
            "bizums_compartidos_no_ingreso_neto": round(
                finance.shared_bizum_reimbursements,
                2,
            ),
            "entradas_positivas_no_ingreso_neto": round(
                finance.shared_bizum_reimbursements + finance.other_positive_income,
                2,
            ),
            "otros_ingresos_positivos": round(finance.other_positive_income, 2),
            "margen_disponible_estimado": _round_optional(positive_margin),
            "deficit_estimado": round(detected_deficit, 2),
            "ratio_ahorro_estimado": _round_optional(positive_savings_rate),
        },
        "naturaleza_financiera": {
            "ingreso": round(cashflow_nature.income, 2),
            "gasto": round(cashflow_nature.expense, 2),
            "consumo": round(cashflow_nature.consumption, 2),
            "ahorro": round(cashflow_nature.saving, 2),
            "inversion": round(cashflow_nature.investment, 2),
            "reembolsos_compartidos": round(cashflow_nature.shared_reimbursements, 2),
        },
        "top_categorias_gasto": [
            {
                "categoria": item.label,
                "importe": round(item.amount, 2),
                "movimientos": item.count,
                "peso": round(item.ratio, 4),
            }
            for item in top_categories
        ],
        "flujo_mensual": [
            {
                "mes": item.month,
                "gastos": round(item.expenses, 2),
                "salario": _round_optional(item.salary),
                "ingresos_clases_bizum": round(item.class_income, 2),
                "bizums_compartidos": round(item.shared_bizum_reimbursements, 2),
                "balance_banco": round(item.bank_cashflow, 2),
            }
            for item in monthly_flow
        ],
        "patrones_fechas": date_patterns,
        "patrones_recurrentes": [
            {
                "concepto_o_categoria": item.label,
                "tipo": item.kind,
                "importe_total": round(item.total_amount, 2),
                "importe_medio": round(item.average_amount, 2),
                "meses": item.months,
                "movimientos": item.movements,
                "ejemplos": list(item.examples),
            }
            for item in recurring_patterns
        ],
        "bizums_que_atenuan_gastos_compartidos": [
            {
                "fecha_bizum": item.bizum_date,
                "bizum_recibido": round(item.bizum_amount, 2),
                "fecha_gasto": item.expense_date,
                "gasto_bruto": round(item.expense_amount, 2),
                "gasto_neto_estimado": round(item.net_expense_estimate, 2),
                "concepto_gasto": item.expense_concept,
                "categoria_gasto": item.expense_category,
                "dias_diferencia": item.days_delta,
            }
            for item in bizum_offsets
        ],
        "top_movimientos_gasto": top_movements,
        "bizum_intencion": {
            "clases": round(bizum_intent.class_income, 2),
            "reembolsos_compartidos": round(bizum_intent.shared_reimbursements, 2),
            "reembolsos_por_memoria": round(
                bizum_intent.contextual_shared_reimbursements,
                2,
            ),
            "entradas_ambiguas": round(bizum_intent.ambiguous_income, 2),
            "pagos_salientes": round(bizum_intent.outgoing_payments, 2),
            "memoria_usada": bizum_intent.memory_used,
        },
    }


def build_deepseek_agent_context(
    dataframe: pd.DataFrame,
    *,
    group_col: str = "CategoriaLeaf",
    confirmed_context: str = "",
    movement_limit: int = 40,
) -> dict[str, Any]:
    """Build DeepSeek context with summary plus movement-level evidence."""
    context = build_deepseek_expense_context(
        dataframe,
        group_col=group_col,
        confirmed_context=confirmed_context,
    )
    context["movimientos_visibles"] = _build_visible_movements_context(
        dataframe,
        limit=movement_limit,
    )
    context["movimientos_visibles_total"] = int(len(dataframe))
    context["movimientos_visibles_truncados"] = max(len(dataframe) - movement_limit, 0)
    return context


def build_bizum_intent_summary(
    dataframe: pd.DataFrame,
    *,
    confirmed_context: str = "",
) -> BizumIntentSummary:
    """Split Bizums by likely intent before LLM review."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return BizumIntentSummary(0.0, 0.0, 0.0, 0.0, 0.0, False)

    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    text = _combined_text(dataframe)
    bizum_mask = text.str.contains("bizum")
    class_mask = build_class_bizum_income_mask(dataframe)
    memory_used = _memory_suggests_shared_event(confirmed_context)
    contextual_shared_mask = (
        bizum_mask
        & (amounts > 0)
        & ~class_mask
        & _bizum_context_mask(text)
        & memory_used
    )
    shared_mask = build_shared_bizum_reimbursement_mask(dataframe)
    ambiguous_income_mask = bizum_mask & (amounts > 0) & ~class_mask & ~shared_mask

    return BizumIntentSummary(
        class_income=float(amounts.loc[class_mask].sum()),
        shared_reimbursements=float(amounts.loc[shared_mask].sum()),
        contextual_shared_reimbursements=float(amounts.loc[contextual_shared_mask].sum()),
        ambiguous_income=float(amounts.loc[ambiguous_income_mask].sum()),
        outgoing_payments=float(amounts.loc[bizum_mask & (amounts < 0)].abs().sum()),
        memory_used=memory_used,
    )


def build_recurring_expense_patterns(
    dataframe: pd.DataFrame,
    *,
    group_col: str = "CategoriaLeaf",
    limit: int = 10,
) -> list[RecurringExpensePattern]:
    """Detect monthly recurring, subscription and frequent consumption patterns."""
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return []
    effective_group = _effective_group_column(dataframe, group_col)
    if effective_group is None:
        return []

    working = dataframe.copy()
    working["_month"] = pd.to_datetime(working["Fecha"], errors="coerce").dt.to_period("M")
    working = working.dropna(subset=["_month"])
    if working.empty:
        return []

    amounts = pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(working)
    expenses = working.loc[(amounts < 0) & nature.eq("consumo")].copy()
    if expenses.empty:
        return []

    expenses["_expense_amount"] = amounts.loc[expenses.index].abs()
    label_source = expenses[effective_group].fillna("").astype(str).str.strip()
    fallback = expenses["Concepto"].fillna("").astype(str).str.strip()
    expenses["_pattern_label"] = label_source.where(label_source.ne(""), fallback)
    expenses["_pattern_label"] = expenses["_pattern_label"].replace("", "Sin categoria")
    expenses["_pattern_text"] = _combined_text(expenses)

    patterns: list[RecurringExpensePattern] = []
    for label, group in expenses.groupby("_pattern_label"):
        monthly = group.groupby("_month")["_expense_amount"].sum()
        months = int(monthly.shape[0])
        movements = int(group.shape[0])
        total = float(group["_expense_amount"].sum())
        average = float(monthly.mean()) if months else 0.0
        variation = (
            float((monthly.max() - monthly.min()) / average)
            if average > 0 and months > 1
            else 0.0
        )
        text_blob = " ".join(group["_pattern_text"].astype(str).tolist())
        is_subscription = _subscription_text_mask(pd.Series([text_blob])).iloc[0]
        if months >= 2 and (variation <= 0.35 or is_subscription):
            kind = "subscripcion" if is_subscription or average <= 80 else "recurrente_mensual"
        elif movements >= 3:
            kind = "frecuente"
        else:
            continue
        examples = tuple(
            group["Concepto"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Sin concepto")
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        patterns.append(
            RecurringExpensePattern(
                label=str(label),
                kind=kind,
                total_amount=total,
                average_amount=average if months >= 2 else total / max(movements, 1),
                months=months,
                movements=movements,
                examples=examples,
            )
        )

    return sorted(
        patterns,
        key=lambda item: (
            item.kind != "subscripcion",
            item.kind != "recurrente_mensual",
            -item.total_amount,
        ),
    )[:limit]


def build_bizum_offset_candidates(
    dataframe: pd.DataFrame,
    *,
    group_col: str = "CategoriaLeaf",
    limit: int = 10,
    max_days_delta: int = 4,
) -> list[BizumOffsetCandidate]:
    """Match incoming shared Bizums with nearby larger shared expenses."""
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return []

    working = dataframe.copy()
    working["_date"] = pd.to_datetime(working["Fecha"], errors="coerce")
    working = working.dropna(subset=["_date"])
    if working.empty:
        return []

    amounts = pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0)
    shared_mask = build_shared_bizum_reimbursement_mask(working)
    nature = classify_cashflow_nature(working)
    expenses = working.loc[(amounts < 0) & nature.eq("consumo")].copy()
    bizums = working.loc[shared_mask].copy()
    if expenses.empty or bizums.empty:
        return []

    expenses["_expense_amount"] = amounts.loc[expenses.index].abs()
    bizums["_bizum_amount"] = amounts.loc[bizums.index]
    effective_group = _effective_group_column(expenses, group_col) or "Concepto"
    candidates: list[BizumOffsetCandidate] = []
    for _, bizum in bizums.sort_values("_date").iterrows():
        bizum_amount = float(bizum["_bizum_amount"])
        if bizum_amount <= 0:
            continue
        nearby = expenses.copy()
        nearby["_days_delta"] = (
            nearby["_date"].sub(bizum["_date"]).abs().dt.days.astype(int)
        )
        nearby["_shared_context"] = _bizum_context_mask(_combined_text(nearby)).astype(int)
        nearby = nearby.loc[
            (nearby["_days_delta"] <= max_days_delta)
            & (nearby["_expense_amount"] >= bizum_amount)
        ].sort_values(
            ["_shared_context", "_days_delta", "_expense_amount"],
            ascending=[False, True, False],
        )
        if nearby.empty:
            continue
        expense = nearby.iloc[0]
        candidates.append(
            BizumOffsetCandidate(
                bizum_date=_format_date_value(bizum.get("Fecha")),
                bizum_amount=bizum_amount,
                expense_date=_format_date_value(expense.get("Fecha")),
                expense_amount=float(expense["_expense_amount"]),
                net_expense_estimate=max(float(expense["_expense_amount"]) - bizum_amount, 0.0),
                expense_concept=str(expense.get("Concepto", "")).strip(),
                expense_category=str(expense.get(effective_group, "")).strip(),
                days_delta=int(expense["_days_delta"]),
            )
        )

    return sorted(candidates, key=lambda item: item.bizum_amount, reverse=True)[:limit]


def classify_cashflow_nature(dataframe: pd.DataFrame) -> pd.Series:
    """Classify each row as ingreso, gasto, consumo, ahorro or inversion."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return pd.Series("gasto", index=dataframe.index, dtype="object")

    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    text = _combined_text(dataframe)
    salary_mask = build_salary_income_mask(dataframe)
    class_mask = build_class_bizum_income_mask(dataframe)
    shared_bizum_mask = build_shared_bizum_reimbursement_mask(dataframe)
    investment_mask = _investment_mask(text)
    saving_mask = _saving_mask(text)

    nature = pd.Series("gasto", index=dataframe.index, dtype="object")
    nature.loc[amounts > 0] = "ingreso"
    nature.loc[(amounts < 0) & investment_mask] = "inversion"
    nature.loc[(amounts < 0) & saving_mask & ~investment_mask] = "ahorro"
    nature.loc[(amounts < 0) & ~investment_mask & ~saving_mask] = "consumo"
    nature.loc[shared_bizum_mask] = "reembolso_compartido"
    nature.loc[salary_mask | class_mask] = "ingreso"
    return nature


def build_cashflow_nature_summary(dataframe: pd.DataFrame) -> CashflowNatureSummary:
    """Aggregate amounts by financial nature."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return CashflowNatureSummary(0.0, 0.0, 0.0, 0.0, 0.0)

    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(dataframe)
    salary_mask = build_salary_income_mask(dataframe)
    class_mask = build_class_bizum_income_mask(dataframe)
    income = float(amounts.loc[(salary_mask | class_mask) & (amounts > 0)].sum())
    expense = float(amounts.loc[amounts < 0].abs().sum())
    consumption = float(amounts.loc[nature.eq("consumo") & (amounts < 0)].abs().sum())
    saving = float(amounts.loc[nature.eq("ahorro") & (amounts < 0)].abs().sum())
    investment = float(amounts.loc[nature.eq("inversion") & (amounts < 0)].abs().sum())
    shared_reimbursements = float(
        amounts.loc[nature.eq("reembolso_compartido") & (amounts > 0)].sum()
    )
    return CashflowNatureSummary(
        income=income,
        expense=expense,
        consumption=consumption,
        saving=saving,
        investment=investment,
        shared_reimbursements=shared_reimbursements,
    )


def build_class_bizum_income_mask(dataframe: pd.DataFrame) -> pd.Series:
    """Return incoming Bizums that look like class income."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype=bool)
    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    text = _combined_text(dataframe)
    amount_matches_class_price = amounts.apply(_matches_class_amount)
    bizum_text = text.str.contains("bizum")
    return (amounts > 0) & (
        text.str.contains("clase") | (bizum_text & amount_matches_class_price)
    )


def build_salary_income_mask(dataframe: pd.DataFrame) -> pd.Series:
    """Return incoming transfers that look like payroll salary."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype=bool)
    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    text = _combined_text(dataframe)
    return (amounts > 0) & text.str.contains("nomina|nómina")


def build_shared_bizum_reimbursement_mask(dataframe: pd.DataFrame) -> pd.Series:
    """Return incoming Bizums that are treated as shared-payment reimbursements."""
    if dataframe.empty or "Importe" not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype=bool)
    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    text = _combined_text(dataframe)
    class_mask = build_class_bizum_income_mask(dataframe)
    return (amounts > 0) & text.str.contains("bizum") & ~class_mask


def build_prioritized_review_queue(
    dataframe: pd.DataFrame,
    *,
    confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return review candidates ordered by highest value and lowest confidence."""
    if dataframe.empty:
        return dataframe.copy()

    review_mask = build_review_mask(
        dataframe,
        confidence_threshold=confidence_threshold,
    )
    queue_frame = dataframe.loc[review_mask].copy()
    if queue_frame.empty:
        return queue_frame

    confidence = pd.to_numeric(
        queue_frame.get("CategoriaConfianza", pd.Series(index=queue_frame.index)),
        errors="coerce",
    ).fillna(-1.0)
    leaf = _string_column(queue_frame, "CategoriaLeaf")
    source = _string_column(queue_frame, "CategoriaFuente")

    queue_frame["_review_confidence"] = confidence
    queue_frame["_review_manual"] = leaf.eq("Revision manual").astype(int)
    queue_frame["_review_fallback"] = source.eq("ia_arbol:fallback").astype(int)
    queue_frame["_review_abs_importe"] = (
        pd.to_numeric(queue_frame["Importe"], errors="coerce").fillna(0.0).abs()
        if "Importe" in queue_frame.columns
        else 0.0
    )
    ordered = queue_frame.sort_values(
        [
            "_review_fallback",
            "_review_manual",
            "_review_confidence",
            "_review_abs_importe",
        ],
        ascending=[False, False, True, False],
    )
    ordered = ordered.drop(
        columns=[
            "_review_confidence",
            "_review_manual",
            "_review_fallback",
            "_review_abs_importe",
        ]
    )
    if limit is not None:
        return ordered.head(limit)
    return ordered


def summarize_ai_trace(value: Any) -> str:
    """Convert a compact JSON trace into a readable one-line explanation."""
    steps = parse_ai_trace(value)
    if not steps:
        return ""

    rendered: list[str] = []
    for step in steps:
        node = str(step.get("nodo", "")).strip()
        choice = str(step.get("eleccion", "")).strip()
        confidence = _format_confidence(step.get("confianza"))
        cached = " cache" if step.get("cache") is True else ""
        reason = str(step.get("motivo", "")).strip()
        label = f"{node} -> {choice} ({confidence}{cached})"
        rendered.append(f"{label}: {reason}" if reason else label)
    return " | ".join(rendered)


def parse_ai_trace(value: Any) -> list[dict[str, Any]]:
    """Parse a stored AI trace, tolerating missing or legacy values."""
    if value is None:
        return []
    if isinstance(value, list):
        return [step for step in value if isinstance(step, dict)]
    if pd.isna(value):
        return []
    raw_text = str(value).strip()
    if not raw_text:
        return []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [step for step in parsed if isinstance(step, dict)]


def count_trace_cache_hits(dataframe: pd.DataFrame) -> int:
    """Count rows with at least one AI trace step served from cache."""
    if "CategoriaTrazaIA" not in dataframe.columns:
        return 0
    hits = 0
    for value in dataframe["CategoriaTrazaIA"]:
        if any(step.get("cache") is True for step in parse_ai_trace(value)):
            hits += 1
    return hits


def _build_top_expense_movements(
    dataframe: pd.DataFrame,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if dataframe.empty or "Importe" not in dataframe.columns:
        return []
    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(dataframe)
    expenses = dataframe.loc[(amounts < 0) & nature.eq("consumo")].copy()
    if expenses.empty:
        return []
    expenses["_abs_importe"] = amounts.loc[expenses.index].abs()
    expenses = expenses.sort_values("_abs_importe", ascending=False).head(limit)
    output: list[dict[str, Any]] = []
    for _, row in expenses.iterrows():
        output.append(
            {
                "fecha": _format_date_value(row.get("Fecha")),
                "concepto": str(row.get("Concepto", "")).strip(),
                "movimiento": str(row.get("Movimiento", "")).strip(),
                "importe": round(float(row.get("_abs_importe", 0.0)), 2),
                "categoria": str(row.get("CategoriaPath", row.get("Grupo", ""))).strip(),
            }
        )
    return output


def _build_visible_movements_context(
    dataframe: pd.DataFrame,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if dataframe.empty:
        return []

    working = dataframe.copy()
    if "Fecha" in working.columns:
        working["_fecha_sort"] = pd.to_datetime(working["Fecha"], errors="coerce")
    else:
        working["_fecha_sort"] = pd.NaT
    if "Importe" in working.columns:
        working["_abs_importe"] = (
            pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0).abs()
        )
    else:
        working["_abs_importe"] = 0.0

    working = working.sort_values(
        ["_fecha_sort", "_abs_importe"],
        ascending=[False, False],
    ).head(limit)
    nature = classify_cashflow_nature(working)
    output: list[dict[str, Any]] = []
    for index, row in working.iterrows():
        current_category = (
            str(row.get("CategoriaPath", row.get("CategoriaLeaf", row.get("Grupo", ""))))
            .strip()
        )
        output.append(
            {
                "id": int(index) if isinstance(index, int) else str(index),
                "fecha": _format_date_value(row.get("Fecha")),
                "concepto": str(row.get("Concepto", "")).strip(),
                "movimiento": str(row.get("Movimiento", "")).strip(),
                "importe": round(_coerce_float(row.get("Importe")), 2),
                "categoria_actual": current_category,
                "naturaleza_financiera": str(nature.loc[index]),
                "tratamiento_contable": _accounting_treatment(str(nature.loc[index])),
                "traza_ia": summarize_ai_trace(row.get("CategoriaTrazaIA", pd.NA)),
            }
        )
    return output


def _build_date_pattern_context(
    dataframe: pd.DataFrame,
    *,
    group_col: str,
    limit: int,
) -> list[dict[str, Any]]:
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return []
    effective_group = group_col if group_col in dataframe.columns else "Concepto"
    if effective_group not in dataframe.columns:
        return []

    working = dataframe.copy()
    working["_date"] = pd.to_datetime(working["Fecha"], errors="coerce").dt.date
    amounts = pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(working)
    expenses = working.loc[(amounts < 0) & nature.eq("consumo")].copy()
    if expenses.empty:
        return []
    expenses["_expense_amount"] = amounts.loc[expenses.index].abs()
    grouped = (
        expenses.groupby("_date")
        .agg(
            gasto=("_expense_amount", "sum"),
            movimientos=("_expense_amount", "size"),
        )
        .sort_values(["gasto", "movimientos"], ascending=[False, False])
        .head(limit)
    )

    output: list[dict[str, Any]] = []
    for date_value, row in grouped.iterrows():
        date_rows = expenses.loc[expenses["_date"] == date_value]
        category_values = (
            date_rows[effective_group]
            .fillna("Sin categoria")
            .astype(str)
            .str.strip()
            .replace("", "Sin categoria")
        )
        top_categories = (
            date_rows.assign(_category=category_values)
            .groupby("_category")["_expense_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        output.append(
            {
                "fecha": str(date_value),
                "gasto": round(float(row["gasto"]), 2),
                "movimientos": int(row["movimientos"]),
                "categorias": [
                    {"categoria": str(index), "importe": round(float(value), 2)}
                    for index, value in top_categories.items()
                ],
            }
        )
    return output


def _income_expense_margin(dataframe: pd.DataFrame) -> dict[str, float]:
    if dataframe.empty or "Importe" not in dataframe.columns:
        return {"income": 0.0, "expense": 0.0, "margin": 0.0}
    amounts = pd.to_numeric(dataframe["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(dataframe)
    income = float(amounts.loc[nature.eq("ingreso") & (amounts > 0)].sum())
    gross_expense = float(amounts.loc[(amounts < 0) & nature.eq("consumo")].abs().sum())
    shared_reimbursements = float(
        amounts.loc[nature.eq("reembolso_compartido") & (amounts > 0)].sum()
    )
    expense = max(gross_expense - shared_reimbursements, 0.0)
    return {"income": income, "expense": expense, "margin": income - expense}


def _average_dict_value(items: list[dict[str, float]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(item[key] for item in items) / len(items))


def _build_health_alert(
    *,
    current_expense: float,
    average_expense: float,
    current_margin: float,
    average_margin: float,
    savings_rate: float | None,
    comparison_months: int,
) -> tuple[str, str]:
    if comparison_months == 0:
        return "Primer mes", "No hay meses previos suficientes para comparar."

    expense_delta = current_expense - average_expense
    margin_delta = current_margin - average_margin
    expense_threshold = max(50.0, average_expense * 0.10)
    margin_threshold = max(50.0, abs(average_margin) * 0.10)
    if expense_delta > expense_threshold:
        return (
            "Gasto por encima",
            f"{expense_delta:.2f} EUR mas que la media reciente.",
        )
    if margin_delta < -margin_threshold:
        return (
            "Margen menor",
            f"{abs(margin_delta):.2f} EUR menos que la media reciente.",
        )
    if savings_rate is not None and savings_rate >= 0.20:
        return "Ahorro sano", "El margen del mes supera el 20% de los ingresos."
    return "Mes estable", "No hay desviaciones fuertes frente a la media reciente."


def _effective_group_column(dataframe: pd.DataFrame, requested: str) -> str | None:
    for candidate in (requested, "CategoriaNivel1", "CategoriaLeaf", "Grupo", "Concepto"):
        if candidate in dataframe.columns:
            return candidate
    return None


def _average_monthly_expense_by_group(
    dataframe: pd.DataFrame,
    *,
    group_col: str,
) -> dict[str, float]:
    if dataframe.empty or "Fecha" not in dataframe.columns or "Importe" not in dataframe.columns:
        return {}

    working = dataframe.copy()
    working["_month"] = pd.to_datetime(working["Fecha"], errors="coerce").dt.to_period("M")
    working = working.dropna(subset=["_month"])
    if working.empty:
        return {}

    amounts = pd.to_numeric(working["Importe"], errors="coerce").fillna(0.0)
    nature = classify_cashflow_nature(working)
    expenses = working.loc[(amounts < 0) & nature.eq("consumo")].copy()
    if expenses.empty:
        return {}

    expenses["_expense_amount"] = amounts.loc[expenses.index].abs()
    labels = expenses[group_col].fillna("Sin categoria").astype(str).str.strip()
    expenses["_expense_label"] = labels.replace("", "Sin categoria")
    grouped = (
        expenses.groupby(["_month", "_expense_label"])["_expense_amount"]
        .sum()
        .reset_index()
    )
    return {
        str(label): float(group["_expense_amount"].mean())
        for label, group in grouped.groupby("_expense_label")
    }


def _memory_suggests_shared_event(confirmed_context: str) -> bool:
    normalized = confirmed_context.lower()
    return any(
        token in normalized
        for token in (
            "quedada",
            "fiesta",
            "viaje",
            "cena",
            "comida",
            "grupo",
            "compartid",
            "reembolso",
        )
    )


def _bizum_context_mask(text: pd.Series) -> pd.Series:
    return text.str.contains(
        "bar|restaurante|tapas|cena|comida|copas|pub|fiesta|viaje|hotel|"
        "airbnb|grupo|compart|reembolso|amigos"
    )


def _subscription_text_mask(text: pd.Series) -> pd.Series:
    return text.str.contains(
        "netflix|spotify|hbo|max|disney|prime|amazon prime|apple|icloud|"
        "youtube|google|microsoft|office|adobe|canva|notion|patreon|"
        "suscrip|subscription|cuota|mensualidad|gimnasio|gym|telefon|fibra|"
        "movistar|vodafone|orange|dazn|playstation|xbox"
    )


def _accounting_treatment(nature: str) -> str:
    if nature == "reembolso_compartido":
        return "restar_del_gasto_compartido; no contar como ingreso"
    if nature == "ingreso":
        return "ingreso_real_si_es_nomina_o_clase"
    if nature == "consumo":
        return "gasto_de_consumo"
    if nature == "ahorro":
        return "movimiento_a_ahorro; no consumo"
    if nature == "inversion":
        return "movimiento_a_inversion; no consumo"
    return "revisar"


def _investment_mask(text: pd.Series) -> pd.Series:
    return text.str.contains(
        r"indexa|fondo|acciones|bonos|\betf\b|broker|cartera|inversion|investment"
    )


def _saving_mask(text: pd.Series) -> pd.Series:
    return text.str.contains(
        "ahorro|saving|hucha|deposito|dep[oó]sito|cuenta ahorro|traspaso ahorro|"
        "cuenta remunerada|pibank|remunerada|traspaso cuenta|transferencia pibank"
    )


def _date_range_context(dataframe: pd.DataFrame) -> dict[str, str | None]:
    if dataframe.empty or "Fecha" not in dataframe.columns:
        return {"desde": None, "hasta": None}
    dates = pd.to_datetime(dataframe["Fecha"], errors="coerce").dropna()
    if dates.empty:
        return {"desde": None, "hasta": None}
    return {
        "desde": dates.min().strftime("%Y-%m-%d"),
        "hasta": dates.max().strftime("%Y-%m-%d"),
    }


def _count_months(dataframe: pd.DataFrame) -> int:
    if dataframe.empty or "Fecha" not in dataframe.columns:
        return 0
    dates = pd.to_datetime(dataframe["Fecha"], errors="coerce").dropna()
    if dates.empty:
        return 0
    return int(dates.dt.to_period("M").nunique())


def _count_time_units(dataframe: pd.DataFrame, mode: str) -> int:
    if dataframe.empty or "Fecha" not in dataframe.columns:
        return 1
    dates = pd.to_datetime(dataframe["Fecha"], errors="coerce").dropna()
    if dates.empty:
        return 1
    if mode == "daily":
        return int((dates.max().normalize() - dates.min().normalize()).days) + 1
    if mode == "monthly_average":
        return int(dates.dt.to_period("M").nunique())
    if mode == "seasonal":
        return _count_seasons(dates)
    if mode == "yearly":
        return int(dates.dt.to_period("Y").nunique())
    return 1


def _count_seasons(dates: pd.Series) -> int:
    season_keys: set[tuple[int, int]] = set()
    for value in dates:
        parsed = pd.Timestamp(value)
        month = int(parsed.month)
        if month in {12, 1, 2}:
            season = 0
            year = int(parsed.year) + 1 if month == 12 else int(parsed.year)
        elif month in {3, 4, 5}:
            season = 1
            year = int(parsed.year)
        elif month in {6, 7, 8}:
            season = 2
            year = int(parsed.year)
        else:
            season = 3
            year = int(parsed.year)
        season_keys.add((year, season))
    return len(season_keys)


def _combined_text(dataframe: pd.DataFrame) -> pd.Series:
    concept = _string_column(dataframe, "Concepto")
    movement = _string_column(dataframe, "Movimiento")
    return (concept + " " + movement).str.lower()


def _matches_class_amount(value: float) -> bool:
    if not math.isfinite(float(value)):
        return False
    return any(abs(float(value) - expected) <= 0.01 for expected in (18.0, 27.0))


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _coerce_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_date_value(value: Any) -> str:
    try:
        parsed = pd.to_datetime(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _string_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series("", index=dataframe.index, dtype="object")
    return dataframe[column].fillna("").astype(str).str.strip()


def _format_confidence(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "0%"
