"""LangGraph node wrappers for the enrichment (GNN-PPI) pipeline.

Each wrapper:
  1. Calls one work function from scripts.enrichment.nodes
  2. Appends the returned NodeResult to state["node_history"]
  3. On success, bubbles up specific metrics into named state fields
  4. On failure, sets state["error"] so the router can divert to error_node
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scripts.enrichment.config import collect_config_from_user
from scripts.enrichment.graph_state import EnrichmentState
from scripts.enrichment.nodes import (
    node_eval,
    node_infer,
    node_load_data,
    node_train,
)


# ---------- entry: collect config ----------

def graph_node_collect_config(state: EnrichmentState) -> dict:
    """Prompt the user for an EnrichmentConfig (skipped when state already has one)."""
    if state.get("config") is not None:
        # When chained from the orchestrator we receive a pre-built config; don't re-prompt.
        return {"node_history": list(state.get("node_history") or [])}
    cfg = collect_config_from_user()
    return {
        "config":            cfg,
        "node_history":      [],
        "n_proteins":        0,
        "n_edges":           0,
        "n_components":      0,
        "esm_dim":           0,
        "pathway_dim":       0,
        "pathway_coverage":  0.0,
        "best_ckpt_path":    None,
        "last_ckpt_path":    None,
        "best_val_f1":       None,
        "best_epoch":        None,
        "history_path":      None,
        "test_metrics_path": None,
        "bucket_metrics":    None,
        "inference_path":    None,
        "n_pairs_predicted": None,
        "n_pairs_skipped":   None,
        "error":             None,
    }


# ---------- shared boilerplate ----------

def _appended(state: EnrichmentState, result) -> list:
    return list(state.get("node_history") or []) + [result]


# ---------- work-node wrappers ----------

def graph_node_load_data(state: EnrichmentState) -> dict:
    cfg = state["config"]
    result = node_load_data(cfg)
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}
    m = result.metrics or {}
    return {
        "node_history":     _appended(state, result),
        "n_proteins":       int(m.get("n_proteins", 0)),
        "n_edges":          int(m.get("n_edges", 0)),
        "n_components":     int(m.get("n_components", 0)),
        "esm_dim":          int(m.get("esm_dim", 0)),
        "pathway_dim":      int(m.get("pathway_dim", 0)),
        "pathway_coverage": float(m.get("pathway_coverage", 0.0)),
    }


def graph_node_train(state: EnrichmentState) -> dict:
    cfg = state["config"]
    if cfg.skip_train:
        # caller has supplied a pre-trained checkpoint via state['best_ckpt_path']
        from scripts.common.node_result import NodeResult
        result = NodeResult(
            name="train",
            ok=True,
            message="skipped: cfg.skip_train=True (using existing checkpoint)",
            metrics={"skipped": True},
        )
        return {"node_history": _appended(state, result)}

    result = node_train(cfg)
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}

    out = result.outputs or {}
    m   = result.metrics or {}
    return {
        "node_history":   _appended(state, result),
        "best_ckpt_path": out.get("best_ckpt"),
        "last_ckpt_path": out.get("last_ckpt"),
        "history_path":   out.get("history"),
        "best_val_f1":    m.get("best_val_f1"),
        "best_epoch":     m.get("best_epoch"),
    }


def graph_node_eval(state: EnrichmentState) -> dict:
    cfg = state["config"]
    if cfg.skip_eval:
        from scripts.common.node_result import NodeResult
        result = NodeResult(name="eval", ok=True, message="skipped: cfg.skip_eval=True",
                            metrics={"skipped": True})
        return {"node_history": _appended(state, result)}

    ckpt_str = state.get("best_ckpt_path")
    if ckpt_str is None:
        from scripts.common.node_result import NodeResult
        result = NodeResult(name="eval", ok=False,
                            message="no best_ckpt_path in state — train must run first")
        return {"node_history": _appended(state, result), "error": result.message}

    result = node_eval(cfg, Path(ckpt_str))
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}

    out = result.outputs or {}
    m   = result.metrics or {}
    return {
        "node_history":      _appended(state, result),
        "test_metrics_path": out.get("metrics_path"),
        "bucket_metrics":    m.get("buckets"),
    }


def graph_node_infer(state: EnrichmentState) -> dict:
    cfg = state["config"]
    if cfg.skip_infer:
        from scripts.common.node_result import NodeResult
        result = NodeResult(name="infer", ok=True, message="skipped: cfg.skip_infer=True",
                            metrics={"skipped": True})
        return {"node_history": _appended(state, result)}

    ckpt_str = state.get("best_ckpt_path")
    if ckpt_str is None:
        from scripts.common.node_result import NodeResult
        result = NodeResult(name="infer", ok=False,
                            message="no best_ckpt_path in state — train must run first")
        return {"node_history": _appended(state, result), "error": result.message}

    result = node_infer(cfg, Path(ckpt_str))
    # node_infer returns ok=True with skipped flag when cfg.pairs_path is None;
    # only treat genuine failure as an error.
    if not result.ok:
        return {"node_history": _appended(state, result), "error": result.message}

    out = result.outputs or {}
    m   = result.metrics or {}
    return {
        "node_history":      _appended(state, result),
        "inference_path":    out.get("inference_path"),
        "n_pairs_predicted": m.get("n_predicted"),
        "n_pairs_skipped":   m.get("n_skipped"),
    }


# ---------- summarizer (LLM with deterministic fallback) ----------

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


def _deterministic_summary(state: EnrichmentState) -> str:
    history = state.get("node_history") or []
    n_ok   = sum(1 for r in history if r.ok)
    n_fail = sum(1 for r in history if not r.ok)
    bm = state.get("bucket_metrics") or {}
    f1_all   = bm.get("all",   {}).get("f1")
    f1_test1 = bm.get("test1", {}).get("f1")
    f1_test2 = bm.get("test2", {}).get("f1")
    f1_test3 = bm.get("test3", {}).get("f1")
    return "\n".join([
        "GNN-PPI enrichment run complete.",
        f"  Graph         : {state.get('n_proteins', '?')} proteins, "
        f"{state.get('n_edges', '?')} edges, {state.get('n_components', '?')} components",
        f"  Features      : ESM-dim={state.get('esm_dim', '?')}, "
        f"pathway-dim={state.get('pathway_dim', '?')} "
        f"(coverage {state.get('pathway_coverage', 0.0):.1%})",
        f"  Best epoch    : {state.get('best_epoch', '?')} (val_f1={state.get('best_val_f1')})",
        f"  Eval F1       : all={f1_all}, test1={f1_test1}, test2={f1_test2}, test3={f1_test3}",
        f"  Inference     : predicted={state.get('n_pairs_predicted')}, "
        f"skipped={state.get('n_pairs_skipped')}",
        f"  Best ckpt     : {state.get('best_ckpt_path')}",
        f"  Nodes ok/fail : {n_ok}/{n_fail}",
    ])


def graph_node_summarize(state: EnrichmentState) -> dict:
    history_lines = "\n".join(
        f"  - {r.name}: {'OK' if r.ok else 'FAIL'} — {r.message}"
        for r in (state.get("node_history") or [])
    )
    bm = state.get("bucket_metrics") or {}
    user_prompt = f"""Summarize this GNN-PPI enrichment run.

Node results:
{history_lines}

Graph: {state.get('n_proteins', '?')} proteins, {state.get('n_edges', '?')} edges
Pathway coverage: {state.get('pathway_coverage', 0.0):.1%}
Best val F1: {state.get('best_val_f1')} at epoch {state.get('best_epoch')}
Test1/2/3 F1: {bm.get('test1', {}).get('f1')} / {bm.get('test2', {}).get('f1')} / {bm.get('test3', {}).get('f1')}
Inference predictions: {state.get('n_pairs_predicted')} (skipped {state.get('n_pairs_skipped')})

Write 3-5 sentences. Be specific about how the model generalized
(test1 = both endpoints seen during training; test3 = both unseen).
Mention any failed nodes that need attention."""

    llm = _make_llm()
    if llm is None:
        return {"messages": [AIMessage(content=_deterministic_summary(state))]}

    try:
        response = llm.invoke([
            SystemMessage(content="You are a careful PPI/network biology assistant."),
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

def graph_node_error(state: EnrichmentState) -> dict:
    err = state.get("error") or "unknown error"
    return {"messages": [AIMessage(content=f"enrichment pipeline halted: {err}")]}
