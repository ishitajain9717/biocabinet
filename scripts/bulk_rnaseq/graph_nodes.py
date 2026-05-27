"""LangGraph node functions.

Each function takes the full PipelineState and returns a partial dict
of state updates. LangGraph merges that dict back into the state.

The LLM summarizer node dispatches between providers via LLM_PROVIDER:
  - LLM_PROVIDER=ollama (default) → local Ollama, model from OLLAMA_MODEL
  - LLM_PROVIDER=openai           → OpenAI, model from OPENAI_MODEL,
                                    needs OPENAI_API_KEY
Any failure (no provider, network down, no quota) falls back to a
deterministic text summary so the pipeline always completes.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scripts.bulk_rnaseq.config import collect_config_from_user
from scripts.bulk_rnaseq.graph_state import PipelineState
from scripts.bulk_rnaseq.nodes import node_deg, node_normalize
from scripts.bulk_rnaseq.run_preprocessing import _run_sample
from scripts.common.node_result import NodeResult

# ---------- node 1: ask user for config ----------


def graph_node_collect_config(state: PipelineState) -> dict:
    """Prompt the user for paths/options and seed the state.

    Returns initial empty values for every field the rest of the graph
    will read, so downstream nodes never hit a KeyError.
    """
    cfg = collect_config_from_user()
    return {
        "config": cfg,
        "node_history": [],
        "count_results": [],
        "failed_samples": [],
        "norm_outputs": {},
        "deg_full_path": None,
        "deg_sig_path": None,
        "deg_pairs_path": None,
        "n_deg_significant": None,
        "error": None,
    }


# ---------- node 2: run all samples through fastqc/trim/align/featurecounts ----------


def graph_node_run_samples(state: PipelineState) -> dict:
    """Loop over every sample in cfg.samples and run the per-sample nodes."""
    cfg = state["config"]
    # Make a fresh copy of the existing history rather than mutating in place.
    # Mutating state directly is allowed by LangGraph but risky for checkpointing.
    history: list[NodeResult] = list(state["node_history"])
    count_results: list[tuple[str, str]] = []
    failed: list[str] = []

    for sample in cfg.samples:
        print(f"\n--- Sample: {sample.name} ---")
        # _run_sample is the per-sample driver we already wrote in
        # run_preprocessing.py — it runs fastqc → trim → align → featurecounts
        # and returns (history, counts_txt | None).
        sample_history, counts_txt = _run_sample(cfg, sample)
        history.extend(sample_history)

        for r in sample_history:
            status = "OK  " if r.ok else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")

        if counts_txt is None:
            failed.append(sample.name)
            print(f"  => stopped early for {sample.name}")
        else:
            # Convert Path -> str so the SQLite checkpointer can JSON-serialise it.
            count_results.append((sample.name, str(counts_txt)))

    return {
        "node_history": history,
        "count_results": count_results,
        "failed_samples": failed,
    }


# ---------- node 3: merge counts and write TPM/FPKM/RPKM ----------


def graph_node_normalize(state: PipelineState) -> dict:
    """Run node_normalize on whatever samples produced counts.

    If no samples succeeded, sets state['error'] so the graph can route
    away from the summarizer to an error handler.
    """
    cfg = state["config"]

    if not state["count_results"]:
        return {"error": "No successful samples — nothing to normalize."}

    # Convert (str, str) tuples back to (str, Path) for node_normalize.
    count_results = [(name, Path(p)) for name, p in state["count_results"]]
    result = node_normalize(cfg, count_results)

    history = list(state["node_history"]) + [result]
    return {
        "node_history": history,
        "norm_outputs": result.outputs if result.ok else {},
        "error": None if result.ok else result.message,
    }


# ---------- node 4: differential expression (PyDESeq2) ----------


def graph_node_deg(state: PipelineState) -> dict:
    """Run DEG analysis on the raw counts matrix produced by `normalize`.

    No-ops (returns ok=True with a 'skipped' message) when the user opted
    out at config time. On success, populates state['deg_pairs_path'] which
    the orchestrator's enrichment auto-chain reads as candidate pairs.
    """
    cfg = state["config"]
    norm_outputs = state.get("norm_outputs") or {}
    raw_counts = norm_outputs.get("counts_raw")

    # Sanity: we need the raw counts matrix even when DEG is enabled. If it's
    # missing we skip rather than crash, so the rest of the pipeline still
    # finishes cleanly.
    if not raw_counts:
        from scripts.common.node_result import NodeResult

        result = NodeResult(
            name="deg",
            ok=True,
            message="skipped: no counts_raw produced upstream (nothing to test)",
            metrics={"skipped": True},
        )
        return {"node_history": list(state["node_history"]) + [result]}

    result = node_deg(cfg, Path(raw_counts))
    history = list(state["node_history"]) + [result]

    if not result.ok:
        return {
            "node_history": history,
            # node_deg failures are loud but should not halt the pipeline at
            # the LangGraph level; the summarize node will report it.
            "deg_full_path": None,
            "deg_sig_path": None,
            "deg_pairs_path": None,
            "n_deg_significant": None,
        }

    out = result.outputs or {}
    m = result.metrics or {}
    return {
        "node_history": history,
        "deg_full_path": out.get("deseq2_full"),
        "deg_sig_path": out.get("deseq2_significant"),
        "deg_pairs_path": out.get("deg_pairs"),
        "n_deg_significant": m.get("n_significant"),
    }


# ---------- node 5: summarize the whole run (LLM, with fallback) ----------


def _deterministic_summary(state: PipelineState) -> str:
    """Plain-Python text summary used when no OPENAI_API_KEY is set."""
    history = state["node_history"]
    n_ok = sum(1 for r in history if r.ok)
    n_fail = sum(1 for r in history if not r.ok)
    failed = state["failed_samples"]
    norms = list(state["norm_outputs"].keys())

    lines = [
        "RNA-seq preprocessing run complete.",
        f"  Nodes succeeded : {n_ok}",
        f"  Nodes failed    : {n_fail}",
        f"  Failed samples  : {failed if failed else 'none'}",
        f"  Normalizations  : {norms if norms else 'none'}",
    ]

    if n_fail:
        lines.append("\nFailures:")
        for r in history:
            if not r.ok:
                lines.append(f"  - {r.name}: {r.message}")

    return "\n".join(lines)


def _make_llm():
    """Return a LangChain chat model based on LLM_PROVIDER, or None to fall back.

    LLM_PROVIDER=ollama (default) → ChatOllama, model = OLLAMA_MODEL or qwen2.5:3b
    LLM_PROVIDER=openai           → ChatOpenAI, model = OPENAI_MODEL or gpt-4o-mini
                                     (requires OPENAI_API_KEY)
    Anything else / unset → None  → use deterministic summary
    """
    provider = (os.getenv("LLM_PROVIDER") or "ollama").lower()

    if provider == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        # Check server is up before attempting a connection that would hang.
        try:
            import urllib.request

            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            print(
                "[summarize] Ollama server not reachable — falling back to "
                "deterministic summary.  Start Ollama with: ollama serve",
                flush=True,
            )
            return None
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=ollama_model,
            temperature=0,
            keep_alive=-1,
            client_kwargs={"timeout": 60},
        )

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout=60,
        )

    return None


def graph_node_summarize(state: PipelineState) -> dict:
    """Use an LLM to write a friendly summary of the run, then append RAG
    pathway context for any significant DEGs that were found.

    Falls back to a deterministic text summary if no LLM is configured.
    The RAG augmentation is always attempted after the LLM step but never
    blocks pipeline completion (errors are silently swallowed).
    """
    history_lines = "\n".join(
        f"  - {r.name}: {'OK' if r.ok else 'FAIL'} — {r.message}"
        for r in state["node_history"]
    )
    user_prompt = f"""Summarize this RNA-seq preprocessing run.

Node results:
{history_lines}

Failed samples: {state['failed_samples'] or 'none'}
Normalizations produced: {list(state['norm_outputs'].keys()) or 'none'}

Write 3-5 sentences. Be specific: how many samples succeeded, what tools ran,
what output files exist, and which (if any) failures need attention."""

    llm = _make_llm()
    if llm is None:
        base = _deterministic_summary(state)
    else:
        print("[summarize] Generating summary with LLM (may take ~10 s)...", flush=True)
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content="You are a careful bioinformatics assistant."
                    ),
                    HumanMessage(content=user_prompt),
                ]
            )
            base = response.content
        except Exception as exc:
            base = (
                f"[LLM summary failed: {type(exc).__name__}: {exc}]\n\n"
                + _deterministic_summary(state)
            )

    # Phase 4a: append RAG pathway context (non-blocking)
    try:
        from scripts.rag.augment import rag_augment_bulk

        rag_extra = rag_augment_bulk(dict(state))
    except Exception:
        rag_extra = ""

    full = base + rag_extra if rag_extra else base
    return {"messages": [AIMessage(content=full)]}
