"""Disk persistence helpers for movements, categories and legacy paths."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Final

import pandas as pd

from src.core.taxonomy import CATEGORY_COLUMNS
from src.support.paths import (
    CATEGORIES_FILE,
    CLEAN_FILE,
    DIR_DOCS,
    EXPORT_FILE,
    INSIGHTS_FILE,
    LLM_CACHE_FILE,
    LOG_FILE,
    PROJECT_ROOT,
    RAW_EXPENSES_FILE,
    RULES_FILE,
)

logger = logging.getLogger(__name__)

EXPORT_BASE_COLUMNS: Final[list[str]] = [
    "Fecha",
    "Concepto",
    "Movimiento",
    "Importe",
]


def migrate_legacy_files(legacy_map: dict[Path, Path] | None = None) -> None:
    """Move legacy root files into the current directory layout."""
    mapping = legacy_map or {
        PROJECT_ROOT / "expenses.log": LOG_FILE,
        PROJECT_ROOT / "expenses.csv": RAW_EXPENSES_FILE,
        PROJECT_ROOT / "clean.csv": CLEAN_FILE,
        PROJECT_ROOT / "categorias_export.csv": EXPORT_FILE,
        PROJECT_ROOT / "movimientos_categorias.csv": CATEGORIES_FILE,
        PROJECT_ROOT / "category_rules.json": RULES_FILE,
        PROJECT_ROOT / "llm_cache.json": LLM_CACHE_FILE,
        PROJECT_ROOT / "deepseek_category_insights.json": INSIGHTS_FILE,
        PROJECT_ROOT / "roadmap.md": DIR_DOCS / "roadmap.md",
    }

    for old_path, new_path in mapping.items():
        if not old_path.exists() or new_path.exists():
            continue
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            logger.info("Movido %s -> %s", old_path.name, new_path)
        except OSError as exc:
            logger.warning("No se pudo mover %s: %s", old_path, exc)


def save_categories_to_disk(
    df: pd.DataFrame,
    *,
    categories_file: Path = CATEGORIES_FILE,
) -> None:
    """Persist assigned categories to disk."""
    category_columns = [column for column in CATEGORY_COLUMNS if column in df.columns]
    if not category_columns:
        return

    mask = _build_category_mask(df)
    if not mask.any():
        return

    export_columns = EXPORT_BASE_COLUMNS + category_columns
    exported = df.loc[mask, export_columns].copy()
    exported.to_csv(categories_file, index=False)
    logger.info("Guardadas %s categorias en %s", len(exported), categories_file)


def load_categories_from_disk(
    df: pd.DataFrame,
    *,
    categories_file: Path = CATEGORIES_FILE,
) -> pd.DataFrame:
    """Load persisted categories and apply them to the working dataframe."""
    if not categories_file.exists():
        return df

    try:
        stored = pd.read_csv(categories_file)
        category_columns = [
            column for column in CATEGORY_COLUMNS if column in stored.columns
        ]
        if not category_columns:
            return df

        working = df.copy()
        stored["_key"] = stored.apply(_make_transaction_key, axis=1)
        working["_key"] = working.apply(_make_transaction_key, axis=1)

        indexed = stored.set_index("_key")
        for column in category_columns:
            mapped = working["_key"].map(indexed[column].to_dict())
            if column in working.columns:
                working[column] = mapped.combine_first(working[column])
            else:
                working[column] = mapped
        working = working.drop(columns=["_key"])

        logger.info(
            "Cargadas %s categorias desde %s",
            int(_build_category_mask(working).sum()),
            categories_file,
        )
        return working
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Error cargando categorias: %s", exc)
        return df


def _make_transaction_key(row: pd.Series) -> str:
    return (
        f"{row['Concepto']}||{row['Movimiento']}||"
        f"{float(row['Importe']):.2f}"
    )


def _build_category_mask(df: pd.DataFrame) -> pd.Series:
    candidate_columns = [
        column
        for column in ("CategoriaPath", "CategoriaLeaf", "Grupo")
        if column in df.columns
    ]
    if not candidate_columns:
        return pd.Series(False, index=df.index, dtype=bool)

    mask = pd.Series(False, index=df.index, dtype=bool)
    for column in candidate_columns:
        values = df[column]
        mask |= values.notna() & values.astype(str).str.strip().ne("")
    return mask
