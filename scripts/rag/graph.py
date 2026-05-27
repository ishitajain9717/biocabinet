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
    graph_node_rag_answer,
)
from scripts.rag.graph_state import RagChatState


def build_rag_chat_graph() -> StateGraph:
    """Return an uncompiled StateGraph for the RAG chat loop."""
    workflow = StateGraph(RagChatState)

    workflow.add_node("collect_query", graph_node_collect_query)
    workflow.add_node("rag_answer", graph_node_rag_answer)

    workflow.set_entry_point("collect_query")

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

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from langgraph.checkpoint.memory import MemorySaver

    from scripts.rag.pipeline_context import assemble_pipeline_results

    p = argparse.ArgumentParser(description="Standalone RAG Q&A chat")
    p.add_argument("--deg-full", default="runs/demo_bulk_v2/06_deg/deseq2_full.tsv")
    p.add_argument(
        "--deg-sig", default="runs/demo_bulk_v2/06_deg/deseq2_significant.tsv"
    )
    p.add_argument("--inference", default=None)
    p.add_argument("--n-deg", type=int, default=None)
    args = p.parse_args()

    # Load real pipeline results from the most recent demo run
    pipeline_results = assemble_pipeline_results(
        deg_full_path=args.deg_full if Path(args.deg_full).exists() else None,
        deg_sig_path=args.deg_sig if Path(args.deg_sig).exists() else None,
        n_deg_significant=args.n_deg,
        inference_path=args.inference,
        n_samples_ok=None,
        conditions=None,
    )

    pipeline_ctx = {
        "deg_sig_path": args.deg_sig if Path(args.deg_sig).exists() else None,
        "n_deg_significant": args.n_deg,
        "child_summary": "",
    }

    print("=== RAG Q&A (standalone) ===")
    if pipeline_results.get("n_genes_tested"):
        print(f"    Genes tested : {pipeline_results['n_genes_tested']}")
    if pipeline_results.get("n_deg_significant") is not None:
        print(f"    Sig DEGs     : {pipeline_results['n_deg_significant']}")
    if pipeline_results.get("inference_n_predicted") is not None:
        print(f"    PPI predicted: {pipeline_results['inference_n_predicted']}")
    print("    Type a biology question, or press Enter to quit.\n")

    memory = MemorySaver()
    graph = build_rag_chat_graph().compile(checkpointer=memory)
    graph.invoke(
        {
            "pipeline_context": pipeline_ctx,
            "pipeline_results": pipeline_results,
            "should_quit": False,
        },
        config={"configurable": {"thread_id": "rag_standalone"}},
    )
    print("=== Q&A session ended ===")
