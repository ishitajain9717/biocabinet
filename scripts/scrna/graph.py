"""Build, compile, and run the scRNA-seq pipeline as a LangGraph.

Flow:
    collect_config → load_data → qc → filter → normalize → pca → cluster → markers
        → [error?] → summarize → END
                  ↘ error_node ↗

Conditional edges after every work node short-circuit to error_node at the
first failure, so we don't walk the rest of the chain pointlessly.

Run with:
    python3 -m scripts.scrna.graph
"""
from __future__ import annotations

import sys
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from scripts.scrna.graph_nodes import (
    graph_node_cluster,
    graph_node_collect_config,
    graph_node_error,
    graph_node_filter,
    graph_node_load_data,
    graph_node_markers,
    graph_node_normalize,
    graph_node_pca,
    graph_node_qc,
    graph_node_summarize,
)
from scripts.scrna.graph_state import ScrnaState


def _route_or_continue(next_step: str):
    """Build a conditional-edge router: error_node if state.error, else next_step."""
    def router(state: ScrnaState) -> str:
        return "error_node" if state.get("error") else next_step
    return router


def build_graph() -> StateGraph:
    """Construct the scRNA-seq StateGraph (not yet compiled)."""
    workflow = StateGraph(ScrnaState)

    workflow.add_node("collect_config", graph_node_collect_config)
    workflow.add_node("load_data",      graph_node_load_data)
    workflow.add_node("qc",             graph_node_qc)
    workflow.add_node("filter",         graph_node_filter)
    workflow.add_node("normalize",      graph_node_normalize)
    workflow.add_node("pca",            graph_node_pca)
    workflow.add_node("cluster",        graph_node_cluster)
    workflow.add_node("markers",        graph_node_markers)
    workflow.add_node("summarize",      graph_node_summarize)
    workflow.add_node("error_node",     graph_node_error)

    workflow.set_entry_point("collect_config")

    # collect_config never sets error itself → plain edge.
    workflow.add_edge("collect_config", "load_data")

    # After every work node: continue if ok, divert to error_node otherwise.
    for prev, nxt in [
        ("load_data",  "qc"),
        ("qc",         "filter"),
        ("filter",     "normalize"),
        ("normalize",  "pca"),
        ("pca",        "cluster"),
        ("cluster",    "markers"),
        ("markers",    "summarize"),
    ]:
        workflow.add_conditional_edges(
            prev,
            _route_or_continue(nxt),
            {nxt: nxt, "error_node": "error_node"},
        )

    workflow.add_edge("summarize",  END)
    workflow.add_edge("error_node", END)

    return workflow


def run_pipeline(thread_id: str | None = None) -> dict:
    """Compile and invoke the scRNA-seq graph. Returns the final state."""
    if thread_id is None:
        thread_id = "scrna_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    workflow = build_graph()
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Starting scRNA-seq pipeline (thread_id={thread_id}) ===\n")
    final_state = graph.invoke({}, config=config)

    if final_state.get("messages"):
        print("\n=== Run summary ===")
        print(final_state["messages"][-1].content)

    return final_state


if __name__ == "__main__":
    cli_thread_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(thread_id=cli_thread_id)
