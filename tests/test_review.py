"""Tests for review queue helpers."""
from __future__ import annotations

import pandas as pd

from src.core.review import build_pending_mask, build_review_mask


def test_build_pending_mask_flags_empty_groups() -> None:
    dataframe = pd.DataFrame({"Grupo": ["Supermercado", pd.NA, ""]})

    pending = build_pending_mask(dataframe)

    assert pending.tolist() == [False, True, True]


def test_build_review_mask_includes_manual_and_low_confidence_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "Grupo": ["Supermercado", "Revision manual", "Restaurante", "Nomina"],
            "CategoriaLeaf": [
                "Supermercado",
                "Revision manual",
                "Restaurante",
                "Nomina",
            ],
            "CategoriaConfianza": [0.92, 0.40, 0.55, 0.91],
        }
    )

    review_mask = build_review_mask(dataframe, confidence_threshold=0.60)

    assert review_mask.tolist() == [False, True, True, False]


def test_build_review_mask_marks_pending_rows_even_without_confidence() -> None:
    dataframe = pd.DataFrame(
        {
            "Grupo": [pd.NA, "Bizum enviado"],
            "CategoriaLeaf": [pd.NA, "Bizum enviado"],
        }
    )

    review_mask = build_review_mask(dataframe)

    assert review_mask.tolist() == [True, False]
