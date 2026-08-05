"""Category taxonomy utilities for hierarchical classification."""
from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.support.paths import CATEGORY_TREE_FILE

CATEGORY_COLUMNS = [
    "Grupo",
    "CategoriaLeaf",
    "CategoriaPath",
    "CategoriaNivel1",
    "CategoriaNivel2",
    "CategoriaNivel3",
    "CategoriaFuente",
    "CategoriaConfianza",
    "CategoriaMotivoIA",
    "CategoriaTrazaIA",
]

CANONICAL_LEAF_ALIASES = {
    "compras online": "Compras online",
    "servicios web": "Servicios web",
    "transferencia personal": "Transferencia personal",
    "bizum enviado": "Bizum enviado",
    "bizum recibido": "Bizum recibido",
    "panaderia": "Panadería",
    "optica": "Óptica",
}

DEFAULT_CATEGORY_TREE: dict[str, Any] = {
    "name": "Root",
    "children": [
        {
            "name": "Ingresos",
            "children": [
                {"name": "Ingresos"},
                {"name": "Ingresos personales"},
                {"name": "Transferencia recibida"},
                {"name": "Bonificación"},
                {"name": "Inversiones"},
            ],
        },
        {
            "name": "Alimentacion",
            "children": [
                {"name": "Alimentación"},
                {"name": "Supermercado"},
                {"name": "Frutería"},
                {"name": "Panadería"},
                {"name": "Cafetería"},
                {"name": "Restaurante"},
                {"name": "Bar"},
                {"name": "Comida rápida"},
                {"name": "Bebidas"},
                {"name": "Comida y Bebida"},
            ],
        },
        {
            "name": "Movilidad",
            "children": [
                {"name": "Transporte"},
                {"name": "Aeropuerto"},
                {"name": "Combustible"},
                {"name": "Gasolina"},
                {"name": "Recarga"},
            ],
        },
        {
            "name": "Vivienda y Servicios",
            "children": [
                {"name": "Servicios"},
                {"name": "Administración Pública"},
                {"name": "Asociaciones"},
                {"name": "Comisiones bancarias"},
                {"name": "Tarjetas Prepago"},
            ],
        },
        {
            "name": "Salud y Cuidado",
            "children": [
                {"name": "Salud"},
                {"name": "Farmacia"},
                {"name": "Peluquería"},
                {"name": "Perfumería"},
                {"name": "Óptica"},
            ],
        },
        {
            "name": "Ocio y Cultura",
            "children": [
                {"name": "Ocio"},
                {"name": "Cultura"},
                {"name": "Espectáculos"},
                {"name": "Entradas"},
                {"name": "Entretenimiento"},
                {"name": "Actividades"},
                {"name": "Juegos"},
                {"name": "Deporte"},
                {"name": "Deportes"},
                {"name": "Libros"},
                {"name": "Educación"},
                {"name": "Servicios Digitales"},
                {"name": "Servicios web"},
                {"name": "Software"},
                {"name": "Tecnología"},
                {"name": "Electrónica"},
            ],
        },
        {
            "name": "Viajes y Estancias",
            "children": [
                {"name": "Alojamiento"},
                {"name": "Camping"},
            ],
        },
        {
            "name": "Compras y Comercio",
            "children": [
                {"name": "Compras"},
                {"name": "Compras online"},
                {"name": "Compras varias"},
                {"name": "Comercio"},
                {"name": "Ropa"},
                {"name": "Tabaco"},
            ],
        },
        {
            "name": "Transferencias y Personal",
            "children": [
                {"name": "Transferencia"},
                {"name": "Transferencia P2P"},
                {"name": "Transferencia personal"},
                {"name": "Bizum enviado"},
                {"name": "Bizum recibido"},
                {"name": "Pagos personales"},
                {"name": "Donaciones"},
            ],
        },
        {
            "name": "Otros",
            "children": [
                {"name": "Varios"},
                {"name": "Revision manual"},
            ],
        },
    ],
}


def load_category_tree(
    tree_file: Path = CATEGORY_TREE_FILE,
) -> dict[str, Any]:
    """Load the taxonomy tree, creating a default file if needed."""
    if not tree_file.exists():
        save_category_tree(DEFAULT_CATEGORY_TREE, tree_file=tree_file)
        return deepcopy(DEFAULT_CATEGORY_TREE)

    try:
        raw_tree = json.loads(tree_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        save_category_tree(DEFAULT_CATEGORY_TREE, tree_file=tree_file)
        return deepcopy(DEFAULT_CATEGORY_TREE)

    return normalize_category_tree(raw_tree)


def save_category_tree(
    tree: dict[str, Any],
    *,
    tree_file: Path = CATEGORY_TREE_FILE,
) -> None:
    """Persist the taxonomy tree to disk."""
    normalized_tree = normalize_category_tree(tree)
    tree_file.write_text(
        json.dumps(normalized_tree, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_category_tree(tree: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw taxonomy structure into a predictable nested tree."""
    name = str(tree.get("name", "Root")).strip() or "Root"
    raw_children = tree.get("children", [])

    children: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(raw_children, list):
        for child in raw_children:
            if not isinstance(child, dict):
                continue
            normalized_child = normalize_category_tree(child)
            key = normalize_key(normalized_child["name"])
            if key in seen:
                continue
            seen.add(key)
            children.append(normalized_child)

    return {"name": name, "children": children}


def merge_missing_leaves(
    tree: dict[str, Any],
    leaves: list[str],
) -> tuple[dict[str, Any], bool]:
    """Merge missing legacy leaves under a dedicated `Legado` branch."""
    working_tree = deepcopy(normalize_category_tree(tree))
    leaf_index = build_leaf_path_map(working_tree)
    missing = sorted(
        {
            canonicalize_leaf(leaf)
            for leaf in leaves
            if leaf and normalize_key(canonicalize_leaf(leaf)) not in leaf_index
        }
    )
    if not missing:
        return working_tree, False

    legacy_branch = find_node_by_path(working_tree, "Legado")
    if legacy_branch is None:
        legacy_branch = {"name": "Legado", "children": []}
        working_tree.setdefault("children", []).append(legacy_branch)

    existing_children = {
        normalize_key(child["name"]) for child in legacy_branch.get("children", [])
    }
    for leaf in missing:
        key = normalize_key(leaf)
        if key not in existing_children:
            legacy_branch.setdefault("children", []).append({"name": leaf})
            existing_children.add(key)

    return working_tree, True


def iter_tree_paths(
    tree: dict[str, Any],
    *,
    include_root: bool = False,
) -> list[str]:
    """Return all node paths in depth-first order."""
    paths: list[str] = []

    def walk(node: dict[str, Any], prefix: list[str]) -> None:
        current_name = node["name"]
        current_path = prefix + ([current_name] if current_name != "Root" else [])
        if current_path or include_root:
            paths.append(" > ".join(current_path) if current_path else "Root")
        for child in node.get("children", []):
            walk(child, current_path)

    walk(normalize_category_tree(tree), [])
    return paths


def iter_leaf_paths(tree: dict[str, Any]) -> list[str]:
    """Return the list of leaf paths in the taxonomy."""
    leaf_paths: list[str] = []

    def walk(node: dict[str, Any], prefix: list[str]) -> None:
        current_name = node["name"]
        current_path = prefix + ([current_name] if current_name != "Root" else [])
        children = node.get("children", [])
        if not children and current_path:
            leaf_paths.append(" > ".join(current_path))
            return
        for child in children:
            walk(child, current_path)

    walk(normalize_category_tree(tree), [])
    return leaf_paths


def build_leaf_path_map(tree: dict[str, Any]) -> dict[str, str]:
    """Build a normalized leaf-to-path map for the taxonomy."""
    mapping: dict[str, str] = {}
    for path in iter_leaf_paths(tree):
        mapping[normalize_key(path.split(" > ")[-1])] = path
    return mapping


def ensure_category_columns(
    df: pd.DataFrame,
    tree: dict[str, Any],
) -> pd.DataFrame:
    """Ensure a dataframe contains normalized hierarchical category columns."""
    working = df.copy()
    leaf_path_map = build_leaf_path_map(tree)

    if "CategoriaLeaf" not in working.columns:
        if "Grupo" in working.columns:
            working["CategoriaLeaf"] = working["Grupo"]
        else:
            working["CategoriaLeaf"] = pd.NA

    if "CategoriaPath" not in working.columns:
        working["CategoriaPath"] = pd.NA

    working["CategoriaLeaf"] = working["CategoriaLeaf"].apply(
        lambda value: canonicalize_leaf(value) if pd.notna(value) else pd.NA
    )

    def resolve_path(row: pd.Series) -> str | pd.NA:
        current_path = row.get("CategoriaPath")
        if pd.notna(current_path) and str(current_path).strip():
            return str(current_path).strip()

        leaf_value = row.get("CategoriaLeaf")
        if pd.isna(leaf_value) or not str(leaf_value).strip():
            return pd.NA

        normalized_key = normalize_key(str(leaf_value))
        return leaf_path_map.get(normalized_key, f"Legado > {leaf_value}")

    working["CategoriaPath"] = working.apply(resolve_path, axis=1)
    working["CategoriaLeaf"] = working["CategoriaPath"].apply(
        lambda value: path_to_leaf(str(value)) if pd.notna(value) else pd.NA
    )
    working["Grupo"] = working["CategoriaLeaf"]

    levels = working["CategoriaPath"].apply(path_to_levels)
    working["CategoriaNivel1"] = levels.apply(lambda value: value[0])
    working["CategoriaNivel2"] = levels.apply(lambda value: value[1])
    working["CategoriaNivel3"] = levels.apply(lambda value: value[2])

    if "CategoriaFuente" not in working.columns:
        working["CategoriaFuente"] = pd.NA
    if "CategoriaConfianza" not in working.columns:
        working["CategoriaConfianza"] = pd.NA
    if "CategoriaMotivoIA" not in working.columns:
        working["CategoriaMotivoIA"] = pd.NA
    if "CategoriaTrazaIA" not in working.columns:
        working["CategoriaTrazaIA"] = pd.NA

    return working


def apply_category_assignments(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    tree: dict[str, Any],
) -> pd.DataFrame:
    """Apply a dataframe of category assignments to the working dataframe."""
    working = ensure_category_columns(df, tree)
    for column in assignments.columns:
        working.loc[assignments.index, column] = assignments[column]
    return ensure_category_columns(working, tree)


def build_assignments_from_leaf_series(
    leaf_series: pd.Series,
    tree: dict[str, Any],
    *,
    source: str,
    confidence: float | pd.Series | None = None,
    reason: str | pd.Series | None = None,
    trace: str | pd.Series | None = None,
) -> pd.DataFrame:
    """Build assignment columns from a series of leaf labels."""
    leaf_path_map = build_leaf_path_map(tree)

    def to_path(value: Any) -> str | pd.NA:
        if pd.isna(value) or not str(value).strip():
            return pd.NA
        canonical_leaf = canonicalize_leaf(str(value))
        return leaf_path_map.get(
            normalize_key(canonical_leaf),
            f"Legado > {canonical_leaf}",
        )

    canonical_leafs = leaf_series.apply(
        lambda value: canonicalize_leaf(value) if pd.notna(value) else pd.NA
    )
    paths = canonical_leafs.apply(to_path)
    levels = paths.apply(path_to_levels)

    payload = {
        "Grupo": canonical_leafs,
        "CategoriaLeaf": canonical_leafs,
        "CategoriaPath": paths,
        "CategoriaNivel1": levels.apply(lambda value: value[0]),
        "CategoriaNivel2": levels.apply(lambda value: value[1]),
        "CategoriaNivel3": levels.apply(lambda value: value[2]),
        "CategoriaFuente": source,
        "CategoriaConfianza": confidence if confidence is not None else pd.NA,
        "CategoriaMotivoIA": reason if reason is not None else pd.NA,
        "CategoriaTrazaIA": trace if trace is not None else pd.NA,
    }
    return pd.DataFrame(payload, index=leaf_series.index)


def canonicalize_leaf(value: Any) -> str:
    """Canonicalize a leaf label using alias resolution."""
    text = str(value).strip()
    if not text:
        return text

    alias_key = normalize_key(text)
    if alias_key in CANONICAL_LEAF_ALIASES:
        return CANONICAL_LEAF_ALIASES[alias_key]
    return text


def normalize_key(value: str) -> str:
    """Normalize a label for tree matching and deduplication."""
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(without_accents.lower().split())


def path_to_leaf(path: str) -> str:
    """Extract the leaf name from a category path."""
    return str(path).split(" > ")[-1].strip()


def path_to_levels(path: Any) -> tuple[str | pd.NA, str | pd.NA, str | pd.NA]:
    """Expand a path into up to three category levels."""
    if pd.isna(path) or not str(path).strip():
        return (pd.NA, pd.NA, pd.NA)

    parts = [part.strip() for part in str(path).split(" > ") if part.strip()]
    padded = parts + [pd.NA] * (3 - len(parts))
    return padded[0], padded[1], padded[2]


def find_node_by_path(
    tree: dict[str, Any],
    path: str | None,
) -> dict[str, Any] | None:
    """Return the subtree at a given path."""
    normalized_tree = normalize_category_tree(tree)
    if path is None or not str(path).strip():
        return normalized_tree

    parts = [part.strip() for part in str(path).split(" > ") if part.strip()]
    current = normalized_tree
    for part in parts:
        next_node = None
        for child in current.get("children", []):
            if normalize_key(child["name"]) == normalize_key(part):
                next_node = child
                break
        if next_node is None:
            return None
        current = next_node
    return current
