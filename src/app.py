"""Application entrypoints and startup orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.cli import cli_needs_dataframe, execute_cli, has_cli_actions, parse_args
from src.core.processing import filter_and_group, load_and_clean_csv
from src.core.taxonomy import (
    ensure_category_columns,
    load_category_tree,
    merge_missing_leaves,
    save_category_tree,
)
from src.infra.storage import load_categories_from_disk, migrate_legacy_files
from src.support.env import load_dotenv
from src.support.logging_utils import configure_logging
from src.support.paths import (
    CATEGORIES_FILE,
    CATEGORY_TREE_FILE,
    RAW_EXPENSES_FILE,
    ensure_directories,
)
from src.ui.dashboard import gui_unavailable_reason, is_gui_available, run_gui


def initialize_runtime() -> None:
    """Prepare folders, environment and logging for the application."""
    ensure_directories()
    load_dotenv()
    configure_logging()
    migrate_legacy_files()


def bootstrap_dataframe(
    csv_path: Path | str = RAW_EXPENSES_FILE,
    *,
    categories_file: Path = CATEGORIES_FILE,
    tree_file: Path = CATEGORY_TREE_FILE,
) -> pd.DataFrame:
    """Load the working dataframe with any persisted categories applied."""
    initialize_runtime()
    dataframe = load_and_clean_csv(csv_path)
    dataframe = load_categories_from_disk(dataframe, categories_file=categories_file)

    category_tree = load_category_tree(tree_file)
    category_tree, tree_changed = merge_missing_leaves(
        category_tree,
        _collect_legacy_leaves(dataframe),
    )
    if tree_changed:
        save_category_tree(category_tree, tree_file=tree_file)
    return ensure_category_columns(dataframe, category_tree)


def headless_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build the console summary used when GUI support is unavailable."""
    return filter_and_group(
        df,
        group_col="Concepto",
        pattern="",
        importe_range=(None, None),
        date_range=(None, None),
    )


def print_headless_summary(df: pd.DataFrame) -> None:
    """Print a compact summary to stdout."""
    print(headless_summary(df).head(20).to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the GUI if available, otherwise fall back to console mode."""
    args = parse_args(argv)

    if has_cli_actions(args) and not cli_needs_dataframe(args):
        initialize_runtime()
        execute_cli(
            args,
            dataframe=None,
            category_tree=load_category_tree(),
        )
        return

    if args.csv is None:
        dataframe = bootstrap_dataframe()
    else:
        dataframe = bootstrap_dataframe(args.csv)
    category_tree = load_category_tree()

    if has_cli_actions(args):
        execute_cli(
            args,
            dataframe=dataframe,
            category_tree=category_tree,
        )
        return

    if is_gui_available():
        run_gui(dataframe)
    else:
        print(f"GUI no disponible: {gui_unavailable_reason()}")
        print("Instala/ejecuta Flet en este mismo interprete para abrir la interfaz.")
        print_headless_summary(dataframe)


def _collect_legacy_leaves(dataframe: pd.DataFrame) -> list[str]:
    columns = [column for column in ("CategoriaLeaf", "Grupo") if column in dataframe.columns]
    values: list[str] = []
    for column in columns:
        series = dataframe[column].dropna()
        values.extend(
            str(value).strip()
            for value in series.tolist()
            if str(value).strip()
        )
    return values
