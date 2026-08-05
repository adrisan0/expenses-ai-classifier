"""DeepSeek client and cache-aware categorization helpers."""
from __future__ import annotations

import json
import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.core.insights import classify_cashflow_nature
from src.core.taxonomy import (
    build_assignments_from_leaf_series,
    find_node_by_path,
    normalize_category_tree,
    normalize_key,
    path_to_leaf,
)
from src.support.paths import LLM_CACHE_FILE, TREE_LLM_CACHE_FILE

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
TREE_FALLBACK_LABEL = "__revision_manual__"
_SUPERMARKET_PATTERNS = (
    "alcampo",
    "aldi",
    "carrefour",
    "consum",
    "dia",
    "eroski",
    "hipercor",
    "lidl",
    "mercadona",
    "supercor",
    "supermercado",
)
_FRUIT_SHOP_PATTERNS = ("fruteria", "verduleria")
_BAKERY_PATTERNS = ("panaderia", "pasteleria", "obrador")
_CAFE_PATTERNS = ("cafe", "cafeteria", "starbucks")
_FAST_FOOD_PATTERNS = (
    "burger king",
    "domino",
    "goiko",
    "kfc",
    "mcdonald",
    "taco bell",
    "telepizza",
)
_BAR_PATTERNS = (
    " bar ",
    "cerveceria",
    "pub",
    "taberna",
    "taperia",
    "vinoteca",
)
_RESTAURANT_PATTERNS = (
    "asador",
    "bistro",
    "braseria",
    "parrilla",
    "pizzeria",
    "restaurante",
    "ristorante",
)
_DRINKS_PATTERNS = ("bebidas", "licoreria")
_FOOD_ROOT_PATTERNS = (
    _SUPERMARKET_PATTERNS
    + _FRUIT_SHOP_PATTERNS
    + _BAKERY_PATTERNS
    + _CAFE_PATTERNS
    + _FAST_FOOD_PATTERNS
    + _BAR_PATTERNS
    + _RESTAURANT_PATTERNS
    + _DRINKS_PATTERNS
)

logger = logging.getLogger(__name__)
StatusCallback = Callable[[str], None]


def read_deepseek_config() -> tuple[str | None, str, str]:
    """Read the DeepSeek configuration from environment variables."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    return api_key, base_url, model


def llm_request_deepseek(payload: dict[str, Any]) -> dict[str, Any]:
    """Send a raw chat-completions request to DeepSeek."""
    api_key, base_url, _ = read_deepseek_config()
    if not api_key:
        raise RuntimeError("Configura DEEPSEEK_API_KEY en el entorno o en .env")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(url, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bearer {api_key}")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    logger.info("Llamando a DeepSeek: %s", url)
    try:
        with urllib.request.urlopen(request, data=body, timeout=60) as response:
            logger.info("DeepSeek respondio con codigo %s", response.status)
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.error("DeepSeek HTTPError %s", exc.code)
        details = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"DeepSeek HTTPError {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        logger.error("DeepSeek URLError %s", exc)
        raise RuntimeError(f"DeepSeek URLError: {exc}") from exc


def load_llm_cache(cache_path: Path = LLM_CACHE_FILE) -> dict[str, str]:
    """Load the persistent cache of flat DeepSeek classifications."""
    if not cache_path.exists():
        return {}

    try:
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return {str(key): str(value) for key, value in raw_cache.items()}


def load_tree_llm_cache(cache_path: Path = TREE_LLM_CACHE_FILE) -> dict[str, dict[str, Any]]:
    """Load the cache used by the hierarchical traversal."""
    if not cache_path.exists():
        return {}

    try:
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw_cache, dict):
        return {}

    cache: dict[str, dict[str, Any]] = {}
    for key, value in raw_cache.items():
        if not isinstance(value, dict):
            continue
        cache[str(key)] = {
            "categoria": str(value.get("categoria", TREE_FALLBACK_LABEL)),
            "confianza": _coerce_confidence(value.get("confianza")),
            "motivo": str(value.get("motivo", "")).strip(),
            "modelo": str(value.get("modelo", "")).strip(),
        }
    return cache


def categorize_transactions(
    df: pd.DataFrame,
    *,
    cache_path: Path = LLM_CACHE_FILE,
    status_callback: StatusCallback | None = None,
) -> pd.Series:
    """Categorize a dataframe of movements using cached and remote results."""
    api_key, _, model = read_deepseek_config()
    if not api_key:
        raise RuntimeError("Configura DEEPSEEK_API_KEY en el entorno o en .env")

    llm_cache = load_llm_cache(cache_path)
    keys: list[tuple[int, str]] = []
    uncached: list[tuple[int, str, str, str]] = []

    for index, row in df.iterrows():
        concept = str(row.get("Concepto", ""))
        movement = str(row.get("Movimiento", ""))
        cache_key = f"{concept} || {movement}"
        keys.append((index, cache_key))
        if cache_key not in llm_cache:
            uncached.append((index, cache_key, concept, movement))

    if uncached:
        _request_missing_categories(
            uncached,
            llm_cache,
            cache_path=cache_path,
            model=model,
            status_callback=status_callback,
        )

    output = pd.Series(index=df.index, dtype="object")
    for index, cache_key in keys:
        output.at[index] = llm_cache.get(cache_key, "Otros")
    return output


def categorize_transactions_by_tree(
    df: pd.DataFrame,
    *,
    tree: dict[str, Any],
    root_path: str | None = None,
    cache_path: Path = TREE_LLM_CACHE_FILE,
    status_callback: StatusCallback | None = None,
    user_instruction: str | None = None,
) -> pd.DataFrame:
    """Categorize movements by traversing a taxonomy tree node by node."""
    api_key, _, model = read_deepseek_config()
    if not api_key:
        raise RuntimeError("Configura DEEPSEEK_API_KEY en el entorno o en .env")

    normalized_tree = normalize_category_tree(tree)
    start_node = find_node_by_path(normalized_tree, root_path)
    if start_node is None:
        raise ValueError(f"Nodo de taxonomia no encontrado: {root_path}")

    start_path = _normalize_path(root_path, start_node)
    manual_review_path = _resolve_manual_review_path(normalized_tree)
    llm_cache = load_tree_llm_cache(cache_path)

    assignments = _categorize_tree_node(
        df,
        tree=normalized_tree,
        node=start_node,
        node_path=start_path,
        manual_review_path=manual_review_path,
        llm_cache=llm_cache,
        model=model,
        status_callback=status_callback,
        user_instruction=_normalize_user_instruction(user_instruction),
        accumulated_confidence=None,
        accumulated_trace=None,
    )
    _persist_json_cache(llm_cache, cache_path)
    return assignments


def ask_deepseek_about_expenses(
    question: str,
    context: dict[str, Any],
    *,
    status_callback: StatusCallback | None = None,
) -> str:
    """Ask DeepSeek for advice using the current financial context."""
    api_key, _, model = read_deepseek_config()
    if not api_key:
        raise RuntimeError("Configura DEEPSEEK_API_KEY en el entorno o en .env")

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Escribe una pregunta para DeepSeek")

    system_prompt = (
        "Eres un asesor financiero personal para un usuario en Espana. "
        "Usa solo el contexto recibido; no inventes movimientos. "
        "Respeta el alcance_operativo: si dice vista filtrada o seleccion, no hables "
        "de datos fuera de ese alcance. "
        "Si recibes movimientos visibles, puedes citarlos por fecha, concepto, "
        "importe y categoria actual. "
        "Un importe negativo es salida y uno positivo es entrada. "
        "El gasto_total del contexto ya excluye transferencias a ahorro e inversion; "
        "no lo recalcules mezclando todos los negativos. "
        "Para gasto propio usa gasto_ajustado_por_bizums_compartidos cuando exista: "
        "los Bizums compartidos atenúan gastos grandes y se restan del gasto, "
        "no se meten como ingreso. "
        "Ningun Bizum recibido debe tratarse como salario o nomina. Solo puede "
        "ser clase particular si el concepto o el importe lo apoyan claramente; "
        "en caso contrario tratalo como reembolso compartido o gasto negativo. "
        "Analiza concepto, fecha e importes cercanos: un Bizum recibido cerca de "
        "un pago de tarjeta mayor en bar, restaurante, viaje, hotel, fiesta o "
        "grupo suele compensar ese gasto compartido. "
        "Ingreso real de input es nomina y clases particulares; otros positivos "
        "son entradas a revisar, no renta disponible automatica. "
        "Nunca digas 'ahorro negativo': si el margen o deficit_estimado es peor que cero, "
        "llamalo deficit o gasto superior a los ingresos detectados. "
        "Los Bizums compartidos no son ingresos netos. Los Bizums de clases "
        "si cuentan como ingreso extra. Usa la memoria confirmada del usuario "
        "Diferencia siempre entre ingreso, gasto, consumo, ahorro e inversion. "
        "No trates ahorro o inversion como consumo salvo evidencia clara en contra. "
        "como dato de mayor prioridad para viajes, quedadas, fiestas o gastos "
        "ya explicados. Prioriza patrones_recurrentes, suscripciones, gastos "
        "frecuentes y bizums_que_atenuan_gastos_compartidos cuando existan. "
        "Si detectas patrones por fechas o categorizaciones mejorables, senala "
        "cuales movimientos revisar o recategorizar. "
        "dudosas, termina con una seccion 'Dudas para confirmar' con preguntas "
        "concretas. Responde en espanol, con pasos accionables y sin extenderte "
        "mas de 12 lineas."
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "pregunta": normalized_question,
                        "contexto_gastos": context,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.2,
    }
    _emit_status(status_callback, "DeepSeek analizando resumen de gastos")
    response = llm_request_deepseek(payload)
    return str(response["choices"][0]["message"]["content"]).strip()


def _categorize_tree_node(
    df: pd.DataFrame,
    *,
    tree: dict[str, Any],
    node: dict[str, Any],
    node_path: str,
    manual_review_path: str,
    llm_cache: dict[str, dict[str, Any]],
    model: str,
    status_callback: StatusCallback | None,
    user_instruction: str | None,
    accumulated_confidence: pd.Series | None,
    accumulated_trace: dict[int, list[dict[str, Any]]] | None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=df.index)

    children = node.get("children", [])
    if not children:
        leaf_name = path_to_leaf(node_path)
        leaf_series = pd.Series(leaf_name, index=df.index, dtype="object")
        confidence = (
            accumulated_confidence
            if accumulated_confidence is not None
            else pd.Series(1.0, index=df.index, dtype="float64")
        )
        return build_assignments_from_leaf_series(
            leaf_series,
            tree,
            source=f"ia_arbol:{node_path}",
            confidence=confidence,
            reason=_trace_reason_series(node_path, df.index, accumulated_trace),
            trace=_trace_json_series(df.index, accumulated_trace),
        )

    options = [str(child["name"]) for child in children]
    decisions = _resolve_tree_decisions(
        df,
        node_path=node_path,
        options=options,
        llm_cache=llm_cache,
        model=model,
        status_callback=status_callback,
        user_instruction=user_instruction,
    )
    decisions["confianza"] = _combine_confidence(
        accumulated_confidence,
        decisions["confianza"],
    )
    next_trace = _append_trace_steps(
        df.index,
        accumulated_trace,
        decisions,
        node_path=node_path,
        options=options,
    )

    frames: list[pd.DataFrame] = []
    option_map = {normalize_key(option): option for option in options}

    for raw_choice, row_index in decisions.groupby("categoria").groups.items():
        selected_rows = df.loc[row_index]
        selected_decisions = decisions.loc[row_index]
        resolved_choice = option_map.get(normalize_key(str(raw_choice)))
        if resolved_choice is None:
            fallback_leaf = path_to_leaf(manual_review_path)
            fallback_reason = _build_fallback_reason(
                selected_decisions,
                node_path=node_path,
                manual_review_path=manual_review_path,
            )
            frames.append(
                build_assignments_from_leaf_series(
                    pd.Series(fallback_leaf, index=selected_rows.index, dtype="object"),
                    tree,
                    source="ia_arbol:fallback",
                    confidence=selected_decisions["confianza"],
                    reason=fallback_reason,
                    trace=_trace_json_series(selected_rows.index, next_trace),
                )
            )
            continue

        child_node = next(
            child
            for child in children
            if normalize_key(str(child["name"])) == normalize_key(resolved_choice)
        )
        child_path = _join_path(node_path, resolved_choice)
        frames.append(
            _categorize_tree_node(
                selected_rows,
                tree=tree,
                node=child_node,
                node_path=child_path,
                manual_review_path=manual_review_path,
                llm_cache=llm_cache,
                model=model,
                status_callback=status_callback,
                user_instruction=user_instruction,
                accumulated_confidence=selected_decisions["confianza"],
                accumulated_trace=next_trace,
            )
        )

    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames).sort_index()


def _resolve_tree_decisions(
    df: pd.DataFrame,
    *,
    node_path: str,
    options: list[str],
    llm_cache: dict[str, dict[str, Any]],
    model: str,
    status_callback: StatusCallback | None,
    user_instruction: str | None,
) -> pd.DataFrame:
    keys: list[tuple[int, str]] = []
    uncached: list[tuple[int, str, dict[str, Any]]] = []
    cache_hits: set[str] = set()

    for index, row in df.iterrows():
        item_payload = _build_tree_item_payload(row)
        legacy_key = (
            f"{node_path} || {item_payload['concepto']} || "
            f"{item_payload['movimiento']}"
        )
        cache_key = _build_tree_cache_key(
            node_path,
            item_payload,
            user_instruction=user_instruction,
        )
        local_decision = _resolve_local_tree_decision(
            node_path=node_path,
            options=options,
            item_payload=item_payload,
            user_instruction=user_instruction,
        )
        if local_decision is not None:
            llm_cache[cache_key] = local_decision
            keys.append((index, cache_key))
            continue

        resolved_cache_key = cache_key
        if cache_key in llm_cache:
            cache_hits.add(cache_key)
        elif user_instruction is None and legacy_key in llm_cache:
            resolved_cache_key = legacy_key
            cache_hits.add(legacy_key)
        else:
            uncached.append((index, cache_key, item_payload))
        keys.append((index, resolved_cache_key))

    if uncached:
        _request_missing_tree_decisions(
            uncached,
            llm_cache,
            node_path=node_path,
            options=options,
            model=model,
            status_callback=status_callback,
            user_instruction=user_instruction,
        )

    decisions = pd.DataFrame(index=df.index, columns=["categoria", "confianza"])
    decisions["motivo"] = ""
    decisions["cache"] = False
    for index, cache_key in keys:
        cached = llm_cache.get(cache_key, {})
        decisions.at[index, "categoria"] = cached.get("categoria", TREE_FALLBACK_LABEL)
        decisions.at[index, "confianza"] = _coerce_confidence(
            cached.get("confianza", 0.0)
        )
        decisions.at[index, "motivo"] = str(cached.get("motivo", "")).strip()
        decisions.at[index, "cache"] = cache_key in cache_hits

    decisions["categoria"] = decisions["categoria"].apply(
        lambda value: _normalize_tree_choice(value, options)
    )
    decisions["confianza"] = pd.to_numeric(
        decisions["confianza"],
        errors="coerce",
    ).fillna(0.0)
    decisions["cache"] = decisions["cache"].fillna(False).astype(bool)
    return decisions


def _resolve_local_tree_decision(
    *,
    node_path: str,
    options: list[str],
    item_payload: dict[str, Any],
    user_instruction: str | None,
) -> dict[str, Any] | None:
    if user_instruction is not None:
        return None

    if str(item_payload.get("signo", "")) != "gasto":
        return None

    food_choice = _resolve_food_tree_choice(node_path, options, item_payload)
    if food_choice is None:
        return None

    return {
        "categoria": food_choice,
        "confianza": 0.96,
        "motivo": "Heuristica local: separa supermercado de restaurante/bar.",
        "modelo": "local-heuristic",
    }


def _resolve_food_tree_choice(
    node_path: str,
    options: list[str],
    item_payload: dict[str, Any],
) -> str | None:
    normalized_options = {normalize_key(option): option for option in options}
    option_keys = set(normalized_options)
    text = normalize_key(
        " ".join(
            [
                str(item_payload.get("concepto", "")),
                str(item_payload.get("movimiento", "")),
                str(item_payload.get("categoria_actual", "")),
            ]
        )
    )

    if normalize_key(node_path) in {"root", ""}:
        if "alimentacion" in option_keys and _matches_any(
            text,
            _FOOD_ROOT_PATTERNS,
        ):
            return normalized_options["alimentacion"]
        return None

    if "alimentacion" not in normalize_key(node_path):
        return None

    ordered_rules = [
        ("supermercado", _SUPERMARKET_PATTERNS),
        ("fruteria", _FRUIT_SHOP_PATTERNS),
        ("panaderia", _BAKERY_PATTERNS),
        ("cafeteria", _CAFE_PATTERNS),
        ("comida rapida", _FAST_FOOD_PATTERNS),
        ("bar", _BAR_PATTERNS),
        ("restaurante", _RESTAURANT_PATTERNS),
        ("bebidas", _DRINKS_PATTERNS),
    ]
    for option_key, patterns in ordered_rules:
        if option_key in option_keys and _matches_any(text, patterns):
            return normalized_options[option_key]
    return None


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    padded_text = f" {text} "
    return any(pattern in padded_text for pattern in patterns)


def _request_missing_categories(
    uncached: list[tuple[int, str, str, str]],
    llm_cache: dict[str, str],
    *,
    cache_path: Path,
    model: str,
    status_callback: StatusCallback | None,
) -> None:
    batch_size = 120
    max_retries = 3
    total_batches = (len(uncached) + batch_size - 1) // batch_size
    system_prompt = (
        "Eres un asistente que clasifica movimientos bancarios espanoles "
        "en una unica categoria breve. Responde solo JSON con un array "
        "'categorias' de strings, en el mismo orden que 'items'."
    )

    for batch_index in range(total_batches):
        batch = uncached[batch_index * batch_size : (batch_index + 1) * batch_size]
        items_payload = [
            {"concepto": concept, "movimiento": movement}
            for _, _, concept, movement in batch
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"items": items_payload},
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(1, max_retries + 1):
            _emit_status(
                status_callback,
                (
                    f"DeepSeek lote {batch_index + 1}/{total_batches} "
                    f"(intento {attempt})"
                ),
            )
            try:
                response = llm_request_deepseek(payload)
                content = response["choices"][0]["message"]["content"]
                categories = json.loads(content).get("categorias", [])
                if len(categories) != len(items_payload):
                    raise RuntimeError(
                        "DeepSeek devolvio un numero inesperado de categorias"
                    )
                for (_, cache_key, _, _), category in zip(batch, categories):
                    llm_cache[cache_key] = str(category)
                _persist_json_cache(llm_cache, cache_path)
                break
            except Exception as exc:
                logger.warning("DeepSeek fallo en lote %s: %s", batch_index + 1, exc)
                if attempt == max_retries:
                    for _, cache_key, _, _ in batch:
                        llm_cache[cache_key] = "Otros"
                    _persist_json_cache(llm_cache, cache_path)


def _request_missing_tree_decisions(
    uncached: list[tuple[int, str, dict[str, Any]]],
    llm_cache: dict[str, dict[str, Any]],
    *,
    node_path: str,
    options: list[str],
    model: str,
    status_callback: StatusCallback | None,
    user_instruction: str | None,
) -> None:
    batch_size = 40
    max_retries = 3
    total_batches = (len(uncached) + batch_size - 1) // batch_size
    option_list = ", ".join(options)
    system_prompt = (
        "Clasificas movimientos bancarios siguiendo un arbol jerarquico. "
        f"Estas en el nodo '{node_path}'. Debes elegir una sola opcion exacta "
        f"de esta lista: {option_list}. "
        "Cada item incluye fecha, importe numerico y signo. Importes negativos "
        "son gasto; importes positivos son entradas, nomina, reembolsos o "
        "Bizums recibidos. No clasifiques una entrada positiva como gasto salvo "
        "que el arbol solo permita revision. Cada item tambien puede incluir "
        "una pista 'naturaleza_financiera' entre ingreso, gasto, consumo, ahorro, "
        "inversion y reembolso_compartido: usala como pista fuerte. "
        "Ingreso real suele ser nomina o clase particular. Bizums recibidos que "
        "no sean clases suelen ser reembolso_compartido: no los metas como "
        "ingreso neto y usalos para explicar gastos compartidos cercanos. "
        "Ningun Bizum recibido es salario o nomina; si no hay evidencia clara "
        "de clase particular, tratalo como compensacion de un pago compartido. "
        "Cruza concepto, fecha e importe con pagos cercanos de tarjeta en bares, "
        "restaurantes, viajes, hoteles, fiestas o grupos antes de decidir. "
        "No confundas ahorro o inversion con consumo ordinario. No marques un "
        "movimiento como consumo si encaja mejor con ahorro, inversion o "
        "reembolso compartido. No marques un ingreso como gasto. "
        "Cada item tambien puede incluir una 'categoria_actual'; mantenla solo "
        "si encaja bien con la evidencia. Si no encaja, corrigela. "
        "Los Bizums recibidos con 'clase' o 18/27 EUR suelen ser clases; otros "
        "Bizums recibidos suelen ser pagos compartidos. Si fecha e importes "
        "sugieren viaje, fiesta o quedada, usa "
        "esa pista y baja la confianza si no encaja claramente. "
        "Si recibes una 'instruccion_usuario', tratala como una correccion fuerte "
        "de analista humano y priorizala frente a heuristicas debiles. "
        "Responde solo JSON con una clave 'decisiones' que contenga un array "
        "del mismo tamano que 'items'. Cada elemento debe incluir "
        "'categoria', 'confianza' (0 a 1) y 'motivo' corto."
    )

    for batch_index in range(total_batches):
        batch = uncached[batch_index * batch_size : (batch_index + 1) * batch_size]
        items_payload = [item_payload for _, _, item_payload in batch]
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "node_path": node_path,
                            "options": options,
                            "instruccion_usuario": user_instruction or "",
                            "items": items_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(1, max_retries + 1):
            _emit_status(
                status_callback,
                (
                    f"IA arbol {node_path} lote {batch_index + 1}/{total_batches} "
                    f"(intento {attempt})"
                ),
            )
            try:
                response = llm_request_deepseek(payload)
                content = response["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                decisions = parsed.get("decisiones", [])
                if len(decisions) != len(items_payload):
                    raise RuntimeError(
                        "DeepSeek devolvio un numero inesperado de decisiones"
                    )
                for (_, cache_key, _), raw_decision in zip(batch, decisions):
                    llm_cache[cache_key] = _normalize_cached_tree_decision(
                        raw_decision,
                        options=options,
                        model=model,
                    )
                break
            except Exception as exc:
                logger.warning(
                    "DeepSeek fallo recorriendo %s en lote %s: %s",
                    node_path,
                    batch_index + 1,
                    exc,
                )
                if attempt == max_retries:
                    for _, cache_key, _ in batch:
                        llm_cache[cache_key] = {
                            "categoria": TREE_FALLBACK_LABEL,
                            "confianza": 0.0,
                            "motivo": f"Fallo al resolver el nodo {node_path}",
                            "modelo": model,
                        }


def _normalize_cached_tree_decision(
    raw_decision: Any,
    *,
    options: list[str],
    model: str = "",
) -> dict[str, Any]:
    if not isinstance(raw_decision, dict):
        return {
            "categoria": TREE_FALLBACK_LABEL,
            "confianza": 0.0,
            "motivo": "Respuesta invalida",
            "modelo": model,
        }

    return {
        "categoria": _normalize_tree_choice(raw_decision.get("categoria"), options),
        "confianza": _coerce_confidence(raw_decision.get("confianza")),
        "motivo": str(raw_decision.get("motivo", "")).strip(),
        "modelo": model,
    }


def _build_tree_item_payload(row: pd.Series) -> dict[str, Any]:
    concept = str(row.get("Concepto", "")).strip()
    movement = str(row.get("Movimiento", "")).strip()
    amount = _coerce_amount(row.get("Importe"))
    direction = "entrada" if amount > 0 else "gasto" if amount < 0 else "cero"
    nature = classify_cashflow_nature(pd.DataFrame([row])).iloc[0]
    treatment = (
        "restar_del_gasto_compartido; no ingreso"
        if str(nature) == "reembolso_compartido"
        else "ingreso_real" if str(nature) == "ingreso" else "salida"
    )
    return {
        "fecha": _format_payload_date(row.get("Fecha")),
        "concepto": concept,
        "movimiento": movement,
        "importe": round(amount, 2),
        "signo": direction,
        "categoria_actual": str(
            row.get("CategoriaPath", row.get("CategoriaLeaf", row.get("Grupo", "")))
        ).strip(),
        "naturaleza_financiera": str(nature),
        "tratamiento_contable": treatment,
    }


def _build_tree_cache_key(
    node_path: str,
    item_payload: dict[str, Any],
    *,
    user_instruction: str | None = None,
) -> str:
    key = (
        f"{node_path} || {item_payload['concepto']} || "
        f"{item_payload['movimiento']} || {item_payload['fecha']} || "
        f"{item_payload['importe']}"
    )
    if user_instruction is None:
        return key
    instruction_hash = hashlib.sha1(user_instruction.encode("utf-8")).hexdigest()[:12]
    return f"{key} || instruccion:{instruction_hash}"


def _normalize_user_instruction(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized or None


def _coerce_amount(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_payload_date(value: Any) -> str:
    try:
        parsed = pd.to_datetime(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _normalize_tree_choice(value: Any, options: list[str]) -> str:
    option_map = {normalize_key(option): option for option in options}
    raw_text = str(value).strip()
    if not raw_text:
        return TREE_FALLBACK_LABEL

    normalized = normalize_key(raw_text.split(" > ")[-1])
    return option_map.get(normalized, TREE_FALLBACK_LABEL)


def _persist_json_cache(llm_cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.write_text(
        json.dumps(llm_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _emit_status(status_callback: StatusCallback | None, message: str) -> None:
    if status_callback is not None:
        status_callback(message)
    else:
        logger.info(message)


def _normalize_path(root_path: str | None, node: dict[str, Any]) -> str:
    if root_path and str(root_path).strip():
        return str(root_path).strip()
    if str(node.get("name", "")).strip() == "Root":
        return "Root"
    return str(node.get("name", "")).strip()


def _join_path(parent: str, child: str) -> str:
    if not parent or parent == "Root":
        return child
    return f"{parent} > {child}"


def _resolve_manual_review_path(tree: dict[str, Any]) -> str:
    manual_node = find_node_by_path(tree, "Otros > Revision manual")
    if manual_node is not None:
        return "Otros > Revision manual"
    return "Revision manual"


def _combine_confidence(
    accumulated: pd.Series | None,
    current: pd.Series,
) -> pd.Series:
    current_series = pd.to_numeric(current, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if accumulated is None:
        return current_series

    accumulated_series = (
        pd.to_numeric(accumulated, errors="coerce")
        .reindex(current.index)
        .fillna(1.0)
        .clip(0.0, 1.0)
    )
    return pd.concat([accumulated_series, current_series], axis=1).min(axis=1)


def _coerce_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _append_trace_steps(
    row_index: pd.Index,
    accumulated_trace: dict[int, list[dict[str, Any]]] | None,
    decisions: pd.DataFrame,
    *,
    node_path: str,
    options: list[str],
) -> dict[int, list[dict[str, Any]]]:
    traces: dict[int, list[dict[str, Any]]] = {}
    for index in row_index:
        previous = list(accumulated_trace.get(index, [])) if accumulated_trace else []
        confidence = _coerce_confidence(decisions.at[index, "confianza"])
        traces[index] = previous + [
            {
                "nodo": node_path,
                "opciones": options,
                "eleccion": str(decisions.at[index, "categoria"]),
                "confianza": confidence,
                "motivo": str(decisions.at[index, "motivo"]).strip(),
                "cache": bool(decisions.at[index, "cache"]),
            }
        ]
    return traces


def _trace_json_series(
    row_index: pd.Index,
    traces: dict[int, list[dict[str, Any]]] | None,
) -> pd.Series:
    values: dict[int, str] = {}
    for index in row_index:
        steps = traces.get(index, []) if traces else []
        values[index] = json.dumps(steps, ensure_ascii=False, separators=(",", ":"))
    return pd.Series(values, index=row_index, dtype="object")


def _trace_reason_series(
    node_path: str,
    row_index: pd.Index,
    traces: dict[int, list[dict[str, Any]]] | None,
) -> pd.Series:
    values: dict[int, str] = {}
    for index in row_index:
        steps = traces.get(index, []) if traces else []
        motives = [
            str(step.get("motivo", "")).strip()
            for step in steps
            if str(step.get("motivo", "")).strip()
        ]
        suffix = f": {' | '.join(motives)}" if motives else ""
        values[index] = f"Ruta IA: {node_path}{suffix}"
    return pd.Series(values, index=row_index, dtype="object")


def _build_fallback_reason(
    decisions: pd.DataFrame,
    *,
    node_path: str,
    manual_review_path: str,
) -> pd.Series:
    values: dict[int, str] = {}
    for index, row in decisions.iterrows():
        category = str(row.get("categoria", "")).strip()
        confidence = _coerce_confidence(row.get("confianza"))
        motive = str(row.get("motivo", "")).strip()
        cause = "opcion fuera del arbol"
        if category == TREE_FALLBACK_LABEL and confidence <= 0:
            cause = "error API o respuesta invalida"
        elif category == TREE_FALLBACK_LABEL:
            cause = "baja confianza o revision manual solicitada"
        suffix = f": {motive}" if motive else ""
        values[index] = (
            f"Fallback a {manual_review_path} desde {node_path} "
            f"por {cause}{suffix}"
        )
    return pd.Series(values, index=decisions.index, dtype="object")
