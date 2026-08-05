"""Project paths and filesystem helpers."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIR_DATA = PROJECT_ROOT / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_EXPORTS = DIR_DATA / "exports"
DIR_MAPPINGS = DIR_DATA / "mappings"
DIR_INSIGHTS = DIR_DATA / "insights"
DIR_CONFIG = PROJECT_ROOT / "config"
DIR_CACHE = PROJECT_ROOT / "cache"
DIR_LOGS = PROJECT_ROOT / "logs"
DIR_DOCS = PROJECT_ROOT / "docs"

RAW_EXPENSES_FILE = DIR_RAW / "expenses.csv"
CLEAN_FILE = DIR_PROCESSED / "clean.csv"
EXPORT_FILE = DIR_EXPORTS / "categorias_export.csv"
RULES_FILE = DIR_CONFIG / "category_rules.json"
ANALYSIS_CONTEXT_FILE = DIR_CONFIG / "analysis_context.json"
UI_STATE_FILE = DIR_CONFIG / "ui_state.json"
CATEGORY_TREE_FILE = DIR_CONFIG / "category_tree.json"
LLM_CACHE_FILE = DIR_CACHE / "llm_cache.json"
TREE_LLM_CACHE_FILE = DIR_CACHE / "llm_tree_cache.json"
CATEGORIES_FILE = DIR_MAPPINGS / "movimientos_categorias.csv"
INSIGHTS_FILE = DIR_INSIGHTS / "deepseek_category_insights.json"
LOG_FILE = DIR_LOGS / "expenses.log"
ENV_FILE = PROJECT_ROOT / ".env"


def ensure_directories() -> None:
    """Create the runtime directories used by the application."""
    for folder in (
        DIR_DATA,
        DIR_RAW,
        DIR_PROCESSED,
        DIR_EXPORTS,
        DIR_MAPPINGS,
        DIR_INSIGHTS,
        DIR_CONFIG,
        DIR_CACHE,
        DIR_LOGS,
        DIR_DOCS,
    ):
        folder.mkdir(parents=True, exist_ok=True)
