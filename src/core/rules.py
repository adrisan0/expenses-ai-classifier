"""Local regex rules for category assignment."""
from __future__ import annotations

import re
from typing import Pattern

import pandas as pd

CompiledRule = tuple[str, Pattern[str]]


def parse_category_rules(text: str) -> list[CompiledRule]:
    """Parse `Categoria = regex` lines into compiled rules."""
    rules: list[CompiledRule] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            name, pattern = line.split("=", 1)
        elif ":" in line:
            name, pattern = line.split(":", 1)
        else:
            continue

        category = name.strip()
        resolved_pattern = pattern.strip()
        if not category or not resolved_pattern:
            continue

        try:
            rules.append(
                (category, re.compile(resolved_pattern, flags=re.IGNORECASE))
            )
        except re.error:
            continue

    return rules


def build_group_column(
    df: pd.DataFrame,
    rules: list[CompiledRule],
    *,
    source_columns: tuple[str, ...] = ("Concepto", "Movimiento"),
    fallback_label: str | None = None,
) -> pd.Series:
    """Build the effective `Grupo` column from saved labels and local rules."""
    if "Grupo" in df.columns:
        existing = df["Grupo"].copy()
    else:
        existing = pd.Series(pd.NA, index=df.index, dtype="object")

    if not rules:
        return existing

    proposed = pd.Series(pd.NA, index=df.index, dtype="object")
    for category, compiled in rules:
        matches = pd.Series(False, index=df.index, dtype=bool)
        for column in source_columns:
            if column in df.columns:
                matches |= df[column].astype(str).str.contains(compiled, na=False)
        proposed = proposed.mask(proposed.isna() & matches, other=category)

    combined = existing.where(existing.notna() & (existing != ""), other=proposed)
    if fallback_label is None:
        return combined
    return combined.fillna(fallback_label)
