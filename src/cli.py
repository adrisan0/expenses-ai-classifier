"""Command-line workflows for headless usage."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.core.insights import (
    build_audit_metrics,
    build_prioritized_review_queue,
    summarize_ai_trace,
)
from src.core.processing import filter_and_group
from src.core.review import DEFAULT_REVIEW_CONFIDENCE, build_pending_mask, build_review_mask
from src.core.rules import CompiledRule, build_group_column, parse_category_rules
from src.core.taxonomy import (
    apply_category_assignments,
    build_assignments_from_leaf_series,
    ensure_category_columns,
    iter_tree_paths,
)
from src.infra.deepseek import categorize_transactions_by_tree
from src.infra.storage import save_categories_to_disk
from src.support.paths import CATEGORIES_FILE, EXPORT_FILE, RULES_FILE, TREE_LLM_CACHE_FILE

logger = logging.getLogger(__name__)

SUMMARY_GROUP_CHOICES = (
    "Concepto",
    "Movimiento",
    "Grupo",
    "CategoriaNivel1",
    "CategoriaLeaf",
    "CategoriaPath",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for headless workflows."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Dashboard local con modo GUI y flujos CLI explicitos.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Ruta alternativa al CSV de entrada.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="No abrir la GUI Flet y resolver la operacion en consola.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Imprimir un resumen agrupado en consola.",
    )
    parser.add_argument(
        "--group-by",
        default="Concepto",
        choices=SUMMARY_GROUP_CHOICES,
        help="Columna de agrupacion para el resumen CLI.",
    )
    parser.add_argument(
        "--export-summary",
        nargs="?",
        type=Path,
        const=EXPORT_FILE,
        help="Exportar el resumen agrupado a CSV. Sin ruta usa la exportacion por defecto.",
    )
    parser.add_argument(
        "--ai-audit",
        action="store_true",
        help="Imprimir metricas de cobertura, confianza y cola de revision IA.",
    )
    parser.add_argument(
        "--export-ai-audit",
        type=Path,
        help="Exportar la cola priorizada de auditoria IA a CSV.",
    )
    parser.add_argument(
        "--list-tree",
        action="store_true",
        help="Listar los nodos de la taxonomia activa.",
    )
    parser.add_argument(
        "--categorize-tree",
        action="store_true",
        help="Ejecutar la categorizacion IA por arbol en modo CLI.",
    )
    parser.add_argument(
        "--scope",
        help="Nodo de inicio para la categorizacion CLI, por ejemplo 'Alimentacion'.",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Trabajar solo sobre la cola de revision en vez de los pendientes puros.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_REVIEW_CONFIDENCE,
        help="Umbral de confianza para la cola de revision.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else [])
    if args.export_summary is not None:
        args.summary = True
    if args.export_ai_audit is not None:
        args.ai_audit = True
    return args


def has_cli_actions(args: argparse.Namespace) -> bool:
    """Return whether explicit CLI actions were requested."""
    return bool(
        args.no_gui
        or args.summary
        or args.list_tree
        or args.categorize_tree
        or args.ai_audit
        or args.export_summary is not None
        or args.export_ai_audit is not None
    )


def cli_needs_dataframe(args: argparse.Namespace) -> bool:
    """Return whether the requested CLI actions need the loaded dataframe."""
    return bool(
        args.no_gui
        or args.summary
        or args.categorize_tree
        or args.ai_audit
        or args.export_summary is not None
        or args.export_ai_audit is not None
    )


def print_tree_paths(category_tree: dict[str, object]) -> None:
    """Print the active taxonomy tree paths."""
    for path in iter_tree_paths(category_tree, include_root=True):
        print(path)


def execute_cli(
    args: argparse.Namespace,
    *,
    dataframe: pd.DataFrame | None,
    category_tree: dict[str, object],
    categories_file: Path = CATEGORIES_FILE,
    rules_file: Path = RULES_FILE,
) -> bool:
    """Execute the explicit CLI actions requested by the user."""
    handled = False

    if args.list_tree:
        print_tree_paths(category_tree)
        handled = True

    if dataframe is None:
        return handled

    working = ensure_category_columns(dataframe, category_tree)

    if args.categorize_tree:
        working, rules_applied = _apply_saved_rules(
            working,
            category_tree,
            rules_file=rules_file,
        )
        target_mask = _build_cli_target_mask(
            working,
            review_only=args.review_only,
            confidence_threshold=args.confidence_threshold,
        )
        to_categorize = working.loc[target_mask].copy()
        if to_categorize.empty:
            print("No hay filas para categorizar con el modo CLI solicitado.")
        else:
            assignments = categorize_transactions_by_tree(
                to_categorize,
                tree=category_tree,
                root_path=_normalize_scope(args.scope),
                cache_path=TREE_LLM_CACHE_FILE,
                status_callback=print,
            )
            working = apply_category_assignments(working, assignments, category_tree)
            save_categories_to_disk(working, categories_file=categories_file)
            remaining_pending = int(build_pending_mask(working).sum())
            print(
                "IA arbol categorizo "
                f"{len(to_categorize)} movimientos. "
                f"Reglas previas: {rules_applied}. "
                f"Pendientes restantes: {remaining_pending}."
            )
        handled = True

    if args.summary or args.no_gui:
        summary_source = working
        if args.review_only:
            summary_source = summary_source.loc[
                build_review_mask(
                    summary_source,
                    confidence_threshold=args.confidence_threshold,
                )
            ].copy()

        summary = filter_and_group(
            summary_source,
            group_col=args.group_by,
            pattern="",
            importe_range=(None, None),
            date_range=(None, None),
        )
        if summary.empty:
            print("No hay datos para la vista CLI solicitada.")
        else:
            print(summary.to_string(index=False))

        if args.export_summary is not None:
            args.export_summary.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(args.export_summary, index=False)
            print(f"Resumen exportado en {args.export_summary}")
        handled = True

    if args.ai_audit:
        audit_source = working
        if args.review_only:
            audit_source = build_prioritized_review_queue(
                audit_source,
                confidence_threshold=args.confidence_threshold,
            )
        _print_ai_audit(
            audit_source,
            confidence_threshold=args.confidence_threshold,
        )
        if args.export_ai_audit is not None:
            export_frame = build_prioritized_review_queue(
                working,
                confidence_threshold=args.confidence_threshold,
            )
            export_frame = export_frame.copy()
            if "CategoriaTrazaIA" in export_frame.columns:
                export_frame["CategoriaTrazaTexto"] = export_frame[
                    "CategoriaTrazaIA"
                ].apply(summarize_ai_trace)
            args.export_ai_audit.parent.mkdir(parents=True, exist_ok=True)
            export_frame.to_csv(args.export_ai_audit, index=False)
            print(f"Auditoria IA exportada en {args.export_ai_audit}")
        handled = True

    return handled


def _print_ai_audit(
    dataframe: pd.DataFrame,
    *,
    confidence_threshold: float,
) -> None:
    metrics = build_audit_metrics(
        dataframe,
        confidence_threshold=confidence_threshold,
    )
    average_confidence = (
        f"{metrics.average_confidence:.2f}"
        if metrics.average_confidence is not None
        else "n/a"
    )
    print("Auditoria IA")
    print(f"- movimientos: {metrics.total}")
    print(f"- categorizados: {metrics.categorized} ({metrics.coverage_ratio:.1%})")
    print(f"- pendientes: {metrics.pending}")
    print(f"- revision: {metrics.review} ({metrics.review_ratio:.1%})")
    print(f"- reglas locales: {metrics.local_rules}")
    print(f"- IA arbol: {metrics.ai}")
    print(f"- manual: {metrics.manual}")
    print(f"- cache hits aprox: {metrics.cache_hits}")
    print(f"- confianza media: {average_confidence}")

    queue_frame = build_prioritized_review_queue(
        dataframe,
        confidence_threshold=confidence_threshold,
        limit=10,
    )
    if queue_frame.empty:
        print("No hay movimientos en cola de revision.")
        return

    columns = [
        column
        for column in (
            "Fecha",
            "Concepto",
            "Movimiento",
            "Importe",
            "CategoriaPath",
            "CategoriaConfianza",
            "CategoriaFuente",
            "CategoriaMotivoIA",
        )
        if column in queue_frame.columns
    ]
    print("Top cola de revision:")
    print(queue_frame.loc[:, columns].to_string(index=False))


def _load_saved_rules(
    *,
    rules_file: Path = RULES_FILE,
) -> list[CompiledRule]:
    if not rules_file.exists():
        return []

    try:
        payload = json.loads(rules_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("No se pudieron cargar reglas CLI: %s", exc)
        return []

    return parse_category_rules(str(payload.get("rules_text", "")))


def _apply_saved_rules(
    dataframe: pd.DataFrame,
    category_tree: dict[str, object],
    *,
    rules_file: Path = RULES_FILE,
) -> tuple[pd.DataFrame, int]:
    working = ensure_category_columns(dataframe, category_tree)
    rules = _load_saved_rules(rules_file=rules_file)
    if not rules:
        return working, 0

    suggested = build_group_column(working, rules)
    pending_mask = build_pending_mask(working)
    rule_mask = pending_mask & suggested.notna() & suggested.astype(str).str.strip().ne("")
    if not rule_mask.any():
        return working, 0

    assignments = build_assignments_from_leaf_series(
        suggested.loc[rule_mask],
        category_tree,
        source="regla_local",
        confidence=1.0,
        reason="Regla local",
    )
    updated = apply_category_assignments(working, assignments, category_tree)
    return updated, int(rule_mask.sum())


def _build_cli_target_mask(
    dataframe: pd.DataFrame,
    *,
    review_only: bool,
    confidence_threshold: float,
) -> pd.Series:
    if review_only:
        return build_review_mask(
            dataframe,
            confidence_threshold=confidence_threshold,
        )
    return build_pending_mask(dataframe)


def _normalize_scope(scope: str | None) -> str | None:
    if scope is None:
        return None
    normalized = scope.strip()
    if not normalized or normalized == "Root":
        return None
    return normalized
