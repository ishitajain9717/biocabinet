"""Build the interactive RAG Q&A LangGraph subgraph.

Graph topology
--------------
    START → collect_query → _route_after_query ─── END
                                 ↓ (not quit)
                            rag_answer → collect_query  (loop)

Usage (standalone smoke test)
------------------------------
    python3 -m scripts.rag.graph

Usage (from orchestrator)
--------------------------
    from scripts.rag.graph import build_rag_chat_graph
    child = build_rag_chat_graph().compile(checkpointer=memory)
    child.invoke(
        {"pipeline_context": {...}, "should_quit": False},
        config={"configurable": {"thread_id": "..."}},
    )
"""

from __future__ import annotations

# Load .env from the project root so OLLAMA_MODEL etc. don't need manual export.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langgraph.graph import END, StateGraph

from scripts.rag.graph_nodes import (
    _route_after_query,
    graph_node_collect_query,
    graph_node_infer_data_type,
    graph_node_rag_answer,
)
from scripts.rag.graph_state import RagChatState


def build_rag_chat_graph() -> StateGraph:
    """Return an uncompiled StateGraph for the RAG chat loop."""
    workflow = StateGraph(RagChatState)

    workflow.add_node("infer_data_type", graph_node_infer_data_type)
    workflow.add_node("collect_query", graph_node_collect_query)
    workflow.add_node("rag_answer", graph_node_rag_answer)

    # infer_data_type runs once at startup, then hands off to collect_query
    workflow.set_entry_point("infer_data_type")
    workflow.add_edge("infer_data_type", "collect_query")

    # After collect_query: quit → END, or continue → rag_answer
    workflow.add_conditional_edges(
        "collect_query",
        _route_after_query,
        {"rag_answer": "rag_answer", "__end__": END},
    )

    # After rag_answer: always loop back to collect_query
    workflow.add_edge("rag_answer", "collect_query")

    return workflow


# ---------------------------------------------------------------------------
# standalone smoke test
# ---------------------------------------------------------------------------


def _find_latest_deg_run(runs_dir: "Path") -> "dict":
    """Scan runs/ and return paths for the most recent run that has DEGs."""
    import glob as _glob
    from pathlib import Path as _Path

    best: dict = {}
    for sig in sorted(
        _glob.glob(str(runs_dir / "**/deseq2_significant.tsv"), recursive=True),
        key=lambda p: _Path(p).stat().st_mtime,
        reverse=True,
    ):
        sig_path = _Path(sig)
        # Skip empty sig files (0 significant DEGs)
        try:
            lines = sig_path.read_text().strip().splitlines()
            if len(lines) <= 1:  # header only
                continue
        except OSError:
            continue
        run_dir = sig_path.parent.parent
        best["deg_sig"] = str(sig_path)
        full = sig_path.parent / "deseq2_full.tsv"
        best["deg_full"] = str(full) if full.exists() else None
        inf = run_dir / "inference.json"
        if not inf.exists():
            # Check enrichment sibling dir
            for inf_candidate in sorted(
                run_dir.parent.glob("*enrichment*/inference.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                inf = inf_candidate
                break
        best["inference"] = str(inf) if inf.exists() else None
        best["run_dir"] = str(run_dir)
        break
    return best


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from langgraph.checkpoint.memory import MemorySaver

    from scripts.rag.artifact_reader import scan_run_dir
    from scripts.rag.pipeline_context import assemble_pipeline_results

    p = argparse.ArgumentParser(description="Standalone RAG Q&A chat")
    p.add_argument("--deg-full", default=None)
    p.add_argument("--deg-sig", default=None)
    p.add_argument("--inference", default=None)
    p.add_argument(
        "--run-dir",
        default=None,
        help="Pipeline run directory to scan for all artifacts",
    )
    p.add_argument("--n-deg", type=int, default=None)
    args = p.parse_args()

    # Auto-detect the most recent run with significant DEGs if no files given
    if not args.deg_sig:
        latest = _find_latest_deg_run(Path("runs"))
        if latest:
            args.deg_sig = latest.get("deg_sig")
            args.deg_full = args.deg_full or latest.get("deg_full")
            args.inference = args.inference or latest.get("inference")
            args.run_dir = args.run_dir or latest.get("run_dir")
            print(f"[RAG] Auto-detected run: {latest.get('run_dir')}")
        else:
            print(
                "[RAG] No runs with significant DEGs found in runs/."
                " Proceeding without gene filter."
            )

    # Scan run directory for ALL artifacts (STAR, FastQC, featureCounts, GNN)
    run_artifacts: dict = {}
    if args.run_dir and Path(args.run_dir).exists():
        print(f"[RAG] Scanning artifacts in: {args.run_dir}")
        run_artifacts = scan_run_dir(args.run_dir)
        # Back-fill deg paths from artifact scan when not explicitly given
        args.deg_full = args.deg_full or run_artifacts.get("deg_full_path")
        args.deg_sig = args.deg_sig or run_artifacts.get("deg_sig_path")
        args.inference = args.inference or run_artifacts.get("inference_path")

    # Load rich pipeline results (DEG table + GNN inference)
    pipeline_results = assemble_pipeline_results(
        deg_full_path=(
            args.deg_full if args.deg_full and Path(args.deg_full).exists() else None
        ),
        deg_sig_path=(
            args.deg_sig if args.deg_sig and Path(args.deg_sig).exists() else None
        ),
        n_deg_significant=args.n_deg,
        inference_path=args.inference,
        n_samples_ok=run_artifacts.get("config", {}).get("n_samples_ok"),
        conditions=None,
    )

    pipeline_ctx = {
        "deg_sig_path": args.deg_sig,
        "n_deg_significant": (args.n_deg or pipeline_results.get("n_deg_significant")),
        "child_summary": "",
    }

    memory = MemorySaver()
    graph = build_rag_chat_graph().compile(checkpointer=memory)
    graph.invoke(
        {
            "pipeline_context": pipeline_ctx,
            "pipeline_results": pipeline_results,
            "run_artifacts": run_artifacts,
            "experiment_summary": "",
            "data_type_confirmed": False,
            "should_quit": False,
        },
        config={"configurable": {"thread_id": "rag_standalone"}},
    )
    print("=== Q&A session ended ===")
