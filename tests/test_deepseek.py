"""Tests for DeepSeek categorization helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core.insights import parse_ai_trace
from src.infra import deepseek


def test_ask_deepseek_about_expenses_sends_compact_context(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    captured_payload: dict[str, object] = {}

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {"choices": [{"message": {"content": "Revisa restaurantes."}}]}

    monkeypatch.setattr(deepseek, "llm_request_deepseek", fake_request)
    statuses: list[str] = []

    answer = deepseek.ask_deepseek_about_expenses(
        "Donde ahorro?",
        {"finanzas": {"gasto_total": 100.0}, "top_categorias_gasto": []},
        status_callback=statuses.append,
    )

    assert answer == "Revisa restaurantes."
    assert statuses == ["DeepSeek analizando resumen de gastos"]
    system_content = captured_payload["messages"][0]["content"]  # type: ignore[index]
    raw_content = captured_payload["messages"][1]["content"]  # type: ignore[index]
    assert "Dudas para confirmar" in str(system_content)
    assert "Bizums compartidos aten" in str(system_content)
    assert "Ingreso real de input" in str(system_content)
    assert "Ningun Bizum recibido debe tratarse como salario" in str(system_content)
    assert "top_categorias_gasto" in str(raw_content)


def test_categorize_transactions_uses_cache_and_persists_new_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "llm_cache.json"
    cache_path.write_text(
        json.dumps({"Mercadona || Compra": "Supermercado"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"categorias": ["Transferencias"]},
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    statuses: list[str] = []
    monkeypatch.setattr(deepseek, "llm_request_deepseek", fake_request)

    dataframe = pd.DataFrame(
        {
            "Concepto": ["Mercadona", "Bizum"],
            "Movimiento": ["Compra", "Transferencia"],
        }
    )
    categorized = deepseek.categorize_transactions(
        dataframe,
        cache_path=cache_path,
        status_callback=statuses.append,
    )

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert categorized.tolist() == ["Supermercado", "Transferencias"]
    assert persisted["Bizum || Transferencia"] == "Transferencias"
    assert statuses


def test_categorize_transactions_falls_back_to_otros_after_retries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "llm_cache.json"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def failing_request(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(deepseek, "llm_request_deepseek", failing_request)

    dataframe = pd.DataFrame(
        {"Concepto": ["Bizum"], "Movimiento": ["Transferencia"]}
    )
    categorized = deepseek.categorize_transactions(dataframe, cache_path=cache_path)

    assert categorized.tolist() == ["Otros"]
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "Bizum || Transferencia": "Otros"
    }


def test_categorize_transactions_by_tree_walks_taxonomy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "llm_tree_cache.json"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    taxonomy = {
        "name": "Root",
        "children": [
            {
                "name": "Gasto",
                "children": [
                    {"name": "Supermercado"},
                    {"name": "Restaurante"},
                ],
            },
            {
                "name": "Ingreso",
                "children": [{"name": "Nomina"}],
            },
            {
                "name": "Otros",
                "children": [{"name": "Revision manual"}],
            },
        ],
    }

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        user_payload = json.loads(str(payload["messages"][1]["content"]))
        node_path = user_payload["node_path"]
        items = user_payload["items"]
        if node_path == "Root":
            assert items[0]["importe"] == -15.3
            assert items[0]["signo"] == "gasto"
            assert items[0]["naturaleza_financiera"] == "consumo"
            assert items[1]["importe"] == 1200.0
            assert items[1]["signo"] == "entrada"
            assert items[1]["naturaleza_financiera"] == "ingreso"
            decisions = []
            for item in items:
                if item["concepto"] == "Mercadona":
                    decisions.append(
                        {"categoria": "Gasto", "confianza": 0.92, "motivo": "compra"}
                    )
                else:
                    decisions.append(
                        {"categoria": "Ingreso", "confianza": 0.95, "motivo": "nomina"}
                    )
        elif node_path == "Gasto":
            decisions = [
                {"categoria": "Supermercado", "confianza": 0.81, "motivo": "retail"}
            ]
        elif node_path == "Ingreso":
            decisions = [{"categoria": "Nomina", "confianza": 0.98, "motivo": "salario"}]
        else:  # pragma: no cover - defensive guard for unexpected nodes
            raise AssertionError(f"Nodo no esperado: {node_path}")

        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"decisiones": decisions},
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(deepseek, "llm_request_deepseek", fake_request)

    dataframe = pd.DataFrame(
        {
            "Concepto": ["Mercadona", "Empresa"],
            "Movimiento": ["Compra", "Ingreso"],
            "Fecha": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "Importe": [-15.30, 1200.0],
        }
    )
    categorized = deepseek.categorize_transactions_by_tree(
        dataframe,
        tree=taxonomy,
        cache_path=cache_path,
    )

    assert categorized.loc[0, "Grupo"] == "Supermercado"
    assert categorized.loc[0, "CategoriaPath"] == "Gasto > Supermercado"
    assert categorized.loc[0, "CategoriaConfianza"] == 0.81
    trace = parse_ai_trace(categorized.loc[0, "CategoriaTrazaIA"])
    assert [step["nodo"] for step in trace] == ["Root", "Gasto"]
    assert trace[0]["eleccion"] == "Gasto"
    assert trace[0]["cache"] is False
    assert categorized.loc[1, "Grupo"] == "Nomina"
    assert categorized.loc[1, "CategoriaPath"] == "Ingreso > Nomina"
    assert categorized.loc[1, "CategoriaConfianza"] == 0.95

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "Root || Mercadona || Compra || 2026-01-02 || -15.3" in persisted
    assert "Gasto || Mercadona || Compra || 2026-01-02 || -15.3" in persisted


def test_categorize_transactions_by_tree_splits_food_subcategories_locally(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "llm_tree_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                (
                    "Alimentacion || Restaurante La Plaza || Cena || "
                    "2026-01-02 || -32.5"
                ): {
                    "categoria": "Alimentacion",
                    "confianza": 0.7,
                    "motivo": "cache generica antigua",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def unexpected_request(_: dict[str, object]) -> dict[str, object]:
        raise AssertionError("food heuristics should avoid remote calls")

    monkeypatch.setattr(deepseek, "llm_request_deepseek", unexpected_request)

    taxonomy = {
        "name": "Root",
        "children": [
            {
                "name": "Alimentacion",
                "children": [
                    {"name": "Alimentacion"},
                    {"name": "Supermercado"},
                    {"name": "Restaurante"},
                    {"name": "Bar"},
                ],
            },
            {"name": "Otros", "children": [{"name": "Revision manual"}]},
        ],
    }
    dataframe = pd.DataFrame(
        {
            "Concepto": ["Mercadona", "Restaurante La Plaza", "Bar El Sol"],
            "Movimiento": ["Compra", "Cena", "Tapas"],
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "Importe": [-12.0, -32.5, -8.0],
        }
    )

    categorized = deepseek.categorize_transactions_by_tree(
        dataframe,
        tree=taxonomy,
        cache_path=cache_path,
    )

    assert categorized["Grupo"].tolist() == ["Supermercado", "Restaurante", "Bar"]
    assert categorized["CategoriaPath"].tolist() == [
        "Alimentacion > Supermercado",
        "Alimentacion > Restaurante",
        "Alimentacion > Bar",
    ]
    trace = parse_ai_trace(categorized.loc[1, "CategoriaTrazaIA"])
    assert trace[-1]["motivo"] == (
        "Heuristica local: separa supermercado de restaurante/bar."
    )

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert (
        persisted[
            "Alimentacion || Restaurante La Plaza || Cena || 2026-01-02 || -32.5"
        ]["categoria"]
        == "Restaurante"
    )


def test_categorize_transactions_by_tree_falls_back_to_manual_review(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "llm_tree_cache.json"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    taxonomy = {
        "name": "Root",
        "children": [
            {"name": "Compras", "children": [{"name": "Supermercado"}]},
            {"name": "Otros", "children": [{"name": "Revision manual"}]},
        ],
    }

    def failing_request(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(deepseek, "llm_request_deepseek", failing_request)

    dataframe = pd.DataFrame(
        {"Concepto": ["Desconocido"], "Movimiento": ["Compra"]}
    )
    categorized = deepseek.categorize_transactions_by_tree(
        dataframe,
        tree=taxonomy,
        cache_path=cache_path,
    )

    assert categorized.loc[0, "Grupo"] == "Revision manual"
    assert categorized.loc[0, "CategoriaPath"] == "Otros > Revision manual"
    assert categorized.loc[0, "CategoriaFuente"] == "ia_arbol:fallback"
    assert "error API" in categorized.loc[0, "CategoriaMotivoIA"]


def test_categorize_transactions_by_tree_marks_cache_hits_in_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "llm_tree_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "Root || Mercadona || Compra": {
                    "categoria": "Alimentacion",
                    "confianza": 0.9,
                    "motivo": "cache root",
                },
                "Alimentacion || Mercadona || Compra": {
                    "categoria": "Supermercado",
                    "confianza": 0.8,
                    "motivo": "cache leaf",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def unexpected_request(_: dict[str, object]) -> dict[str, object]:
        raise AssertionError("cache should avoid remote calls")

    monkeypatch.setattr(deepseek, "llm_request_deepseek", unexpected_request)

    taxonomy = {
        "name": "Root",
        "children": [
            {"name": "Alimentacion", "children": [{"name": "Supermercado"}]},
            {"name": "Otros", "children": [{"name": "Revision manual"}]},
        ],
    }
    dataframe = pd.DataFrame({"Concepto": ["Mercadona"], "Movimiento": ["Compra"]})

    categorized = deepseek.categorize_transactions_by_tree(
        dataframe,
        tree=taxonomy,
        cache_path=cache_path,
    )

    trace = parse_ai_trace(categorized.loc[0, "CategoriaTrazaIA"])
    assert categorized.loc[0, "CategoriaPath"] == "Alimentacion > Supermercado"
    assert all(step["cache"] is True for step in trace)


def test_categorize_transactions_by_tree_guidance_bypasses_unguided_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "llm_tree_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "Root || Transferencia realizada || Cuenta remunerada pibank || 2026-03-01 || -300.0": {
                    "categoria": "Otros",
                    "confianza": 0.1,
                    "motivo": "cache vieja",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    captured_payloads: list[dict[str, object]] = []

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        captured_payloads.append(payload)
        user_payload = json.loads(str(payload["messages"][1]["content"]))
        node_path = user_payload["node_path"]
        assert user_payload["instruccion_usuario"] == "Esto es ahorro propio"
        assert user_payload["items"][0]["categoria_actual"] == ""
        if node_path == "Root":
            decisions = [{"categoria": "Ahorro", "confianza": 0.94, "motivo": "traspaso"}]
        elif node_path == "Ahorro":
            decisions = [
                {
                    "categoria": "Cuenta remunerada",
                    "confianza": 0.91,
                    "motivo": "cuenta propia",
                }
            ]
        else:  # pragma: no cover - defensive guard
            raise AssertionError(f"Nodo no esperado: {node_path}")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"decisiones": decisions},
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(deepseek, "llm_request_deepseek", fake_request)

    taxonomy = {
        "name": "Root",
        "children": [
            {
                "name": "Ahorro",
                "children": [{"name": "Cuenta remunerada"}],
            },
            {"name": "Otros", "children": [{"name": "Revision manual"}]},
        ],
    }
    dataframe = pd.DataFrame(
        {
            "Concepto": ["Transferencia realizada"],
            "Movimiento": ["Cuenta remunerada pibank"],
            "Fecha": pd.to_datetime(["2026-03-01"]),
            "Importe": [-300.0],
        }
    )

    categorized = deepseek.categorize_transactions_by_tree(
        dataframe,
        tree=taxonomy,
        cache_path=cache_path,
        user_instruction="Esto es ahorro propio",
    )

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert categorized.loc[0, "CategoriaPath"] == "Ahorro > Cuenta remunerada"
    assert len(captured_payloads) == 2
    assert any("instruccion:" in key for key in persisted)
