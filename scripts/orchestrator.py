"""Top-level orchestrator: asks what kind of data the user has, then dispatches
to the matching modality-specific pipeline (bulk_rnaseq / scrna / spatial).

Each modality pipeline is a compiled LangGraph with its own state shape; we
wrap each in a Python node that captures its final summary and error into the
orchestrator's own state.

A single SqliteSaver is shared between the orchestrator and all child
pipelines, so a Ctrl-C mid-run can be resumed by re-running with the same
--thread-id.

Run a fresh pipeline:
    python3 -m scripts.orchestrator

Resume a previous run:
    python3 -m scripts.orchestrator --thread-id orchestrator_20260428_094200
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

# Load .env from the project root so OLLAMA_MODEL etc. don't need manual export.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ---------- defaults ----------

DEFAULT_DB = Path("pipeline_runs.sqlite")
_VALID = ("bulk_rnaseq", "scrna", "spatial")


# ---------- shared state ----------


class OrchestratorState(TypedDict):
    data_type: str
    child_summary: str
    child_error: str | None
    bulk_deg_pairs: str | None  # path to deg_pairs.tsv if bulk DEG ran
    bulk_deg_full_path: str | None  # path to deseq2_full.tsv
    bulk_deg_sig_path: str | None  # path to deseq2_significant.tsv
    bulk_n_deg: int | None  # number of significant DEGs
    bulk_n_samples_ok: int | None  # samples that finished featurecounts
    bulk_conditions: dict | None  # {"treated": ..., "reference": ...}
    enrichment_summary: str
    enrichment_error: str | None
    enrichment_ran: bool
    enrichment_inf_path: str | None  # path to inference.json if ran
    pipeline_results: dict | None  # structured summary for RAG chat
    rag_chat_ran: bool  # True if the user started a RAG Q&A session
    messages: Annotated[list, add_messages]


# ---------- entry: pick a modality ----------


def _ask_data_type() -> str:
    print("=== RNA-seq agentic pipeline ===")
    print("What kind of data are you starting from?")
    print("  bulk_rnaseq  : paired/single FASTQ files")
    print("  scrna        : single-cell .h5ad / 10x output / pbmc3k")
    print("  spatial      : spatial transcriptomics .h5ad   [stub]")
    raw = input(f"Choice {list(_VALID)} [bulk_rnaseq]: ").strip().lower()
    if not raw:
        return "bulk_rnaseq"
    if raw not in _VALID:
        raise ValueError(f"Invalid choice: {raw}. Pick one of {_VALID}.")
    return raw


def graph_node_ask_data_type(state: OrchestratorState) -> dict:
    return {
        "data_type": _ask_data_type(),
        "child_summary": "",
        "child_error": None,
        "bulk_deg_pairs": None,
        "bulk_deg_full_path": None,
        "bulk_deg_sig_path": None,
        "bulk_n_deg": None,
        "bulk_n_samples_ok": None,
        "bulk_conditions": None,
        "enrichment_summary": "",
        "enrichment_error": None,
        "enrichment_ran": False,
        "enrichment_inf_path": None,
        "pipeline_results": None,
        "rag_chat_ran": False,
    }


# ---------- shared subgraph runner ----------


def _run_child(
    build_child_graph_fn,
    memory,
    parent_thread_id: str,
    label: str,
    initial_state: dict | None = None,
    return_state: bool = False,
):
    """Compile a child pipeline with the shared checkpointer and invoke it.

    By default returns ``(summary, error)``:
      * summary — the child's last AIMessage content, or "" if none
      * error   — the child's ``state.error`` field, OR the formatted
                  exception message if the child raised

    With ``return_state=True`` returns ``(summary, error, final_state)`` so
    the caller can pull additional fields out of the child's terminal state.

    Wrapping the whole call in try/except is what makes a child crash become
    data instead of taking down the whole orchestrator.
    """
    child_thread = f"{parent_thread_id}_{label}"
    try:
        child = build_child_graph_fn().compile(checkpointer=memory)
        final = child.invoke(
            initial_state or {},
            config={"configurable": {"thread_id": child_thread}},
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        return ("", msg, None) if return_state else ("", msg)

    summary = ""
    if final.get("messages"):
        summary = final["messages"][-1].content
    error = final.get("error")
    return (summary, error, final) if return_state else (summary, error)


# ---------- final report ----------


def graph_node_final_report(state: OrchestratorState) -> dict:
    parts: list[str] = []

    # primary modality
    if state.get("child_error"):
        parts.append(
            f"[{state['data_type']}] pipeline ended with error:\n"
            f"  {state['child_error']}\n\n"
            f"Last child message:\n  {state.get('child_summary') or '(none)'}"
        )
    else:
        parts.append(
            f"[{state['data_type']}] pipeline finished successfully.\n\n"
            f"{state.get('child_summary') or '(no summary returned)'}"
        )

    # enrichment (only printed when it ran)
    if state.get("enrichment_ran"):
        parts.append("=" * 60)
        if state.get("enrichment_error"):
            parts.append(
                f"[enrichment] pipeline ended with error:\n"
                f"  {state['enrichment_error']}\n\n"
                "Last enrichment message:\n"
                f"  {state.get('enrichment_summary') or '(none)'}"
            )
        else:
            parts.append(
                "[enrichment] GNN-PPI pipeline finished successfully.\n\n"
                f"{state.get('enrichment_summary') or '(no summary returned)'}"
            )

    return {"messages": [AIMessage(content="\n\n".join(parts))]}


# ---------- conditional routers ----------


def _route_data_type(state: OrchestratorState) -> str:
    return {
        "bulk_rnaseq": "run_bulk",
        "scrna": "run_scrna",
        "spatial": "run_spatial",
    }[state["data_type"]]


def _route_after_bulk(state: OrchestratorState) -> str:
    """Auto-chain: if bulk succeeded, run enrichment next; otherwise skip to report."""
    if state.get("child_error"):
        return "final_report"
    return "run_enrichment"


# ---------- graph builder ----------


def build_graph(memory, parent_thread_id: str) -> StateGraph:
    """Build the orchestrator graph.

    Runner nodes are defined as closures so they can capture the shared
    checkpointer and the parent thread_id (used to derive child thread_ids).
    Child graph imports are lazy: a missing dependency for one modality
    won't crash the orchestrator at import time, only when that branch runs.
    """

    def graph_node_run_bulk(state: OrchestratorState) -> dict:
        from scripts.bulk_rnaseq.graph import build_graph as build_bulk

        summary, error, child_state = _run_child(
            build_bulk,
            memory,
            parent_thread_id,
            "bulk",
            return_state=True,
        )
        cs = child_state or {}
        cfg = cs.get("config")

        # Derive conditions dict from config if available
        conditions: dict | None = None
        if cfg is not None:
            try:
                cond_map = getattr(cfg, "conditions", None) or {}
                if cond_map:
                    treated_samples = [s for s, c in cond_map.items() if c != "control"]
                    ref_samples = [s for s, c in cond_map.items() if c == "control"]
                    conditions = {
                        "treated": (
                            cond_map.get(treated_samples[0])
                            if treated_samples
                            else "treated"
                        ),
                        "reference": (
                            cond_map.get(ref_samples[0]) if ref_samples else "control"
                        ),
                        "all": cond_map,
                    }
            except Exception:
                pass

        # Count samples that completed featurecounts
        n_samples_ok: int | None = None
        count_results = cs.get("count_results") or []
        if count_results:
            n_samples_ok = len(count_results)

        return {
            "child_summary": summary,
            "child_error": error,
            "bulk_deg_pairs": cs.get("deg_pairs_path"),
            "bulk_deg_full_path": cs.get("deg_full_path"),
            "bulk_deg_sig_path": cs.get("deg_sig_path"),
            "bulk_n_deg": cs.get("n_deg_significant"),
            "bulk_n_samples_ok": n_samples_ok,
            "bulk_conditions": conditions,
        }

    def graph_node_run_scrna(state: OrchestratorState) -> dict:
        from scripts.scrna.graph import build_graph as build_scrna

        summary, error = _run_child(build_scrna, memory, parent_thread_id, "scrna")
        return {"child_summary": summary, "child_error": error}

    def graph_node_run_spatial(state: OrchestratorState) -> dict:
        from scripts.spatial.graph import build_graph as build_spatial

        summary, error = _run_child(build_spatial, memory, parent_thread_id, "spatial")
        return {"child_summary": summary, "child_error": error}

    def graph_node_run_enrichment(state: OrchestratorState) -> dict:
        """Auto-chained after a successful bulk RNA-seq run.

        Builds a default EnrichmentConfig pointing at the SHS27k benchmark
        dataset. If the bulk DEG node produced a candidate-pair file, that
        path is fed into the enrichment subgraph for inference. Otherwise
        inference self-skips.

        Uses ``split_method="bfs"`` so the test1/2/3 buckets are meaningful
        (see gnn_test): a random split on a connected biological graph
        leaves test3 nearly empty.
        """
        from scripts.enrichment.config import EnrichmentConfig
        from scripts.enrichment.graph import build_graph as build_enrichment

        deg_pairs_str = state.get("bulk_deg_pairs")
        deg_pairs = Path(deg_pairs_str) if deg_pairs_str else None

        cfg = EnrichmentConfig(
            ppi_path=Path("data/ppi_SHS27k.tsv"),
            esm_path=Path("data/esm2_embeddings_SHS27k.pt"),
            pathway_path=Path("data/pathway/06_pathway_embeddings_combined.pt"),
            out_dir=Path("runs") / f"{parent_thread_id}_enrichment",
            epochs=100,
            batch_size=512,
            device="cpu",
            split_method="bfs",
            test_size=0.2,
            seed=1,
            pairs_path=deg_pairs,
        )
        cfg.out_dir.mkdir(parents=True, exist_ok=True)

        summary, error, enrich_state = _run_child(
            build_enrichment,
            memory,
            parent_thread_id,
            "enrichment",
            initial_state={"config": cfg},
            return_state=True,
        )
        es = enrich_state or {}
        return {
            "enrichment_summary": summary,
            "enrichment_error": error,
            "enrichment_ran": True,
            "enrichment_inf_path": es.get("inference_path"),
        }

    def graph_node_offer_rag_chat(state: OrchestratorState) -> dict:
        """Phase 4b: after the pipelines finish, offer the user a Q&A session.

        Asks a single yes/no question.  If the user accepts, the RAG chat
        subgraph is compiled with the shared checkpointer and invoked — it
        loops internally until the user presses Enter to quit.

        All structured pipeline results (DEGs, conditions, inference) are
        assembled here and forwarded so the answerer can give grounded answers.
        """
        print("\n" + "=" * 60)
        print("Your pipelines are done.  Would you like to ask follow-up")
        print("biology questions about the results? (RAG Q&A)")
        ans = input("Start Q&A session? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            return {"rag_chat_ran": False}

        # Reconstruct deg_sig_path from pairs dir for backwards compatibility
        deg_sig = state.get("bulk_deg_sig_path")
        if not deg_sig:
            bp = state.get("bulk_deg_pairs")
            if bp:
                candidate = Path(bp).parent / "deseq2_significant.tsv"
                if candidate.exists():
                    deg_sig = str(candidate)

        # Build rich structured pipeline results for the RAG answerer
        from scripts.rag.pipeline_context import assemble_pipeline_results

        pipeline_results = assemble_pipeline_results(
            deg_full_path=state.get("bulk_deg_full_path"),
            deg_sig_path=deg_sig,
            n_deg_significant=state.get("bulk_n_deg"),
            inference_path=state.get("enrichment_inf_path"),
            n_samples_ok=state.get("bulk_n_samples_ok"),
            conditions=state.get("bulk_conditions"),
        )

        pipeline_ctx = {
            "deg_sig_path": deg_sig,
            "n_deg_significant": state.get("bulk_n_deg"),
            "child_summary": state.get("child_summary") or "",
        }

        from scripts.rag.graph import build_rag_chat_graph

        _run_child(
            build_rag_chat_graph,
            memory,
            parent_thread_id,
            "rag_chat",
            initial_state={
                "pipeline_context": pipeline_ctx,
                "pipeline_results": pipeline_results,
                "should_quit": False,
            },
        )
        return {"rag_chat_ran": True, "pipeline_results": pipeline_results}

    workflow = StateGraph(OrchestratorState)

    workflow.add_node("ask_data_type", graph_node_ask_data_type)
    workflow.add_node("run_bulk", graph_node_run_bulk)
    workflow.add_node("run_scrna", graph_node_run_scrna)
    workflow.add_node("run_spatial", graph_node_run_spatial)
    workflow.add_node("run_enrichment", graph_node_run_enrichment)
    workflow.add_node("final_report", graph_node_final_report)
    workflow.add_node("offer_rag_chat", graph_node_offer_rag_chat)

    workflow.set_entry_point("ask_data_type")

    workflow.add_conditional_edges(
        "ask_data_type",
        _route_data_type,
        {
            "run_bulk": "run_bulk",
            "run_scrna": "run_scrna",
            "run_spatial": "run_spatial",
        },
    )

    # bulk → enrichment (auto-chain) iff bulk succeeded; otherwise straight to report
    workflow.add_conditional_edges(
        "run_bulk",
        _route_after_bulk,
        {
            "run_enrichment": "run_enrichment",
            "final_report": "final_report",
        },
    )
    workflow.add_edge("run_enrichment", "final_report")
    workflow.add_edge("run_scrna", "final_report")
    workflow.add_edge("run_spatial", "final_report")

    # Phase 4b: every pipeline exits through offer_rag_chat → END
    workflow.add_edge("final_report", "offer_rag_chat")
    workflow.add_edge("offer_rag_chat", END)

    return workflow


# ---------- entry point ----------


def run(thread_id: str | None = None, db_path: Path = DEFAULT_DB) -> dict:
    if thread_id is None:
        thread_id = "orchestrator_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n=== Orchestrator (thread_id={thread_id}) ===")
    print(f"    checkpoint db: {db_path}")
    print(
        f"    to resume:     python3 -m scripts.orchestrator --thread-id {thread_id}\n"
    )

    with SqliteSaver.from_conn_string(str(db_path)) as memory:
        workflow = build_graph(memory, thread_id)
        graph = workflow.compile(checkpointer=memory)
        final = graph.invoke({}, config={"configurable": {"thread_id": thread_id}})

    if final.get("messages"):
        print("\n=== Orchestrator summary ===")
        print(final["messages"][-1].content)
    return final


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RNA-seq agentic pipeline orchestrator")
    p.add_argument(
        "--thread-id",
        default=None,
        help=(
            "Resume an existing run by its thread_id"
            " (printed at start of previous run)."
        ),
    )
    p.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"Path to the SQLite checkpoint file (default: {DEFAULT_DB}).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    final_state = run(thread_id=args.thread_id, db_path=Path(args.db))
    sys.exit(0 if final_state else 1)
