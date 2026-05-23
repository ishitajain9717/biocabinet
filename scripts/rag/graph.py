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
    workflow.add_node("rag_answer",    graph_node_rag_answer)

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
    from langgraph.checkpoint.memory import MemorySaver

    print("=== RAG Q&A smoke test (type a question, Enter to quit) ===\n")
    memory = MemorySaver()
    graph  = build_rag_chat_graph().compile(checkpointer=memory)

    # Minimal pipeline context mimicking what the orchestrator would pass
    initial = {
        "pipeline_context": {
            "n_deg_significant": 23,
            "child_summary": "Bulk RNA-seq pipeline finished with 23 significant DEGs.",
        },
        "should_quit": False,
    }
    graph.invoke(initial, config={"configurable": {"thread_id": "rag_smoke"}})
    print("=== Q&A session ended ===")
