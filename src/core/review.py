"""Helpers for review and quality-control queues."""
from __future__ import annotations

import pandas as pd

DEFAULT_REVIEW_CONFIDENCE = 0.65


def build_pending_mask(df: pd.DataFrame) -> pd.Series:
    """Return the rows that still do not have a resolved category."""
    if "Grupo" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    values = df["Grupo"]
    return values.isna() | values.astype(str).str.strip().eq("")


def build_review_mask(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE,
) -> pd.Series:
    """Return rows that should appear in the review queue."""
    pending_mask = build_pending_mask(df)

    if "CategoriaLeaf" in df.columns:
        manual_mask = (
            df["CategoriaLeaf"].fillna("").astype(str).str.strip().eq("Revision manual")
        )
    else:
        manual_mask = pd.Series(False, index=df.index, dtype=bool)

    if "CategoriaConfianza" in df.columns:
        confidence_values = pd.to_numeric(
            df["CategoriaConfianza"],
            errors="coerce",
        )
        leaf_values = df.get("CategoriaLeaf", pd.Series("", index=df.index))
        low_confidence_mask = (
            confidence_values.notna()
            & leaf_values.fillna("").astype(str).str.strip().ne("")
            & confidence_values.lt(confidence_threshold)
        )
    else:
        low_confidence_mask = pd.Series(False, index=df.index, dtype=bool)

    return pending_mask | manual_mask | low_confidence_mask
