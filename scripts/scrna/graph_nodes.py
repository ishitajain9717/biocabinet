"""LangGraph node wrappers for the scRNA-seq pipeline.

Each wrapper:
  1. Calls one work function from scripts.scrna.nodes
  2. Appends the returned NodeResult to state["node_history"]
  3. Updates state["adata_path"] to the new snapshot path on success
  4. Sets state["error"] on failure (so the router can divert to error_node)
  5. Optionally bubbles up specific metrics into named state fields
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scripts.scrna.config import collect_config_from_user
from scripts.scrna.graph_state import ScrnaState
from scripts.scrna.nodes import (
    node_cluster,
    node_filter,
    node_load_data,
    node_markers,
    node_normalize,
    node_pca,
    node_qc,
)


# ---------- entry: collect config ----------

def graph_node_collect_config(state: ScrnaState) -> dict:
    cfg = collect_config_from_user()
    return {
        "config":       cfg,
        "node_history": [],
        "adata_path":   "",
        "n_cells":      0,
        "n_genes":      0,
        "qc_metrics":   {},
        "n_clusters":   0,
        "markers_path": None,
        "error":        None,
    }


# ---------- shared boilerplate ----------

def _appended(state: ScrnaState, result) -> list:
    return list(state["node_history"]) + [result]


# ---------- work-node wrappers ----------

def graph_node_load_data(state: ScrnaState) -> dict:
    result, new_path = node_load_data(state["config"])
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    return {
        "node_history": _appended(state, result),
        "adata_path":   str(new_path),
        "n_cells":      int(result.metrics.get("n_cells", 0)),
        "n_genes":      int(result.metrics.get("n_genes", 0)),
    }


def graph_node_qc(state: ScrnaState) -> dict:
    result, new_path = node_qc(state["config"], Path(state["adata_path"]))
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    return {
        "node_history": _appended(state, result),
        "adata_path":   str(new_path),
        "qc_metrics": {
            "median_n_genes":      result.metrics.get("median_n_genes"),
            "median_pct_mt":       result.metrics.get("median_pct_mt"),
            "median_total_counts": result.metrics.get("median_total_counts"),
        },
    }


def graph_node_filter(state: ScrnaState) -> dict:
    result, new_path = node_filter(state["config"], Path(state["adata_path"]))
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    return {
        "node_history": _appended(state, result),
        "adata_path":   str(new_path),
        "n_cells":      int(result.metrics.get("n_cells_kept", 0)),
        "n_genes":      int(result.metrics.get("n_genes_kept", 0)),
    }


def graph_node_normalize(state: ScrnaState) -> dict:
    result, new_path = node_normalize(state["config"], Path(state["adata_path"]))
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    return {
        "node_history": _appended(state, result),
        "adata_path":   str(new_path),
    }


def graph_node_pca(state: ScrnaState) -> dict:
    result, new_path = node_pca(state["config"], Path(state["adata_path"]))
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    return {
        "node_history": _appended(state, result),
        "adata_path":   str(new_path),
    }


def graph_node_cluster(state: ScrnaState) -> dict:
    result, new_path = node_cluster(state["config"], Path(state["adata_path"]))
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    return {
        "node_history": _appended(state, result),
        "adata_path":   str(new_path),
        "n_clusters":   int(result.metrics.get("n_clusters", 0)),
    }


def graph_node_markers(state: ScrnaState) -> dict:
    result, new_path = node_markers(state["config"], Path(state["adata_path"]))
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    return {
        "node_history": _appended(state, result),
        "adata_path":   str(new_path),
        "markers_path": result.outputs.get("markers_csv"),
    }


# ---------- summarizer (LLM with fallback) ----------

def _make_llm():
    provider = (os.getenv("LLM_PROVIDER") or "ollama").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            temperature=0,
        )
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    return None


def _deterministic_summary(state: ScrnaState) -> str:
    history = state["node_history"]
    n_ok = sum(1 for r in history if r.ok)
    n_fail = sum(1 for r in history if not r.ok)
    return "\n".join([
        "scRNA-seq preprocessing run complete.",
        f"  Final shape   : {state.get('n_cells', '?')} cells x {state.get('n_genes', '?')} genes",
        f"  Clusters      : {state.get('n_clusters', '?')}",
        f"  Markers CSV   : {state.get('markers_path') or 'none'}",
        f"  Nodes ok/fail : {n_ok}/{n_fail}",
    ])


def graph_node_summarize(state: ScrnaState) -> dict:
    history_lines = "\n".join(
        f"  - {r.name}: {'OK' if r.ok else 'FAIL'} — {r.message}"
        for r in state["node_history"]
    )
    user_prompt = f"""Summarize this scRNA-seq preprocessing run.

Node results:
{history_lines}

Final shape: {state.get('n_cells', '?')} cells x {state.get('n_genes', '?')} genes
Clusters: {state.get('n_clusters', '?')}
Markers CSV: {state.get('markers_path') or 'none'}

Write 3-5 sentences. Be specific: how many cells/genes survived QC, how many
clusters, where the markers file is, any failures that need attention."""

    llm = _make_llm()
    if llm is None:
        return {"messages": [AIMessage(content=_deterministic_summary(state))]}

    try:
        response = llm.invoke([
            SystemMessage(content="You are a careful single-cell bioinformatics assistant."),
            HumanMessage(content=user_prompt),
        ])
        return {"messages": [response]}
    except Exception as exc:
        warning = (
            f"[LLM summary failed: {type(exc).__name__}: {exc}]\n\n"
            + _deterministic_summary(state)
        )
        return {"messages": [AIMessage(content=warning)]}


# ---------- error ----------

def graph_node_error(state: ScrnaState) -> dict:
    err = state.get("error") or "unknown error"
    return {"messages": [AIMessage(content=f"scRNA pipeline halted: {err}")]}
