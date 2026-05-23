"""Build, compile, and run the RNA-seq preprocessing pipeline as a LangGraph.

The graph wires the node functions from graph_nodes.py into the flow:

    collect_config -> run_samples -> normalize -> deg -> summarize -> END
                                                ↘ error_node ↗

The `deg` node self-skips when ``cfg.enable_deg=False`` so the same graph
shape covers both "preprocessing only" and "preprocessing + DEG" runs.

Run from the project root:
    python3 -m scripts.bulk_rnaseq.graph
"""
from __future__ import annotations

import sys
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from scripts.bulk_rnaseq.graph_nodes import (
    graph_node_collect_config,
    graph_node_deg,
    graph_node_normalize,
    graph_node_run_samples,
    graph_node_summarize,
)
from scripts.bulk_rnaseq.graph_state import PipelineState


# ---------- tiny inline error node ----------

def graph_node_error(state: PipelineState) -> dict:
    """Terminal node when state['error'] was set upstream.

    Records the error as an AIMessage so the final printed message is
    informative, then the graph routes to END.
    """
    err = state.get("error") or "unknown error"
    return {"messages": [AIMessage(content=f"Run halted: {err}")]}


# ---------- conditional router ----------

def _route_after_normalize(state: PipelineState) -> str:
    """Decide which node runs after normalize.

    Returns the *string name* of the next node. LangGraph looks up that
    name in the dict passed to add_conditional_edges.
    """
    return "error_node" if state.get("error") else "deg"


# ---------- graph builder ----------

def build_graph() -> StateGraph:
    """Construct the StateGraph (not yet compiled)."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("collect_config", graph_node_collect_config)
    workflow.add_node("run_samples",    graph_node_run_samples)
    workflow.add_node("normalize",      graph_node_normalize)
    workflow.add_node("deg",            graph_node_deg)
    workflow.add_node("summarize",      graph_node_summarize)
    workflow.add_node("error_node",     graph_node_error)

    workflow.set_entry_point("collect_config")

    workflow.add_edge("collect_config", "run_samples")
    workflow.add_edge("run_samples",    "normalize")

    # normalize → (DEG | error). deg always continues to summarize.
    workflow.add_conditional_edges(
        "normalize",
        _route_after_normalize,
        {"deg": "deg", "error_node": "error_node"},
    )
    workflow.add_edge("deg",        "summarize")

    workflow.add_edge("summarize",  END)
    workflow.add_edge("error_node", END)

    return workflow


# ---------- entry point ----------

def run_pipeline(thread_id: str | None = None) -> dict:
    """Compile the graph with an in-memory checkpointer and invoke it.

    Returns the final PipelineState dict.
    """
    if thread_id is None:
        thread_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    workflow = build_graph()

    # MemorySaver keeps state in a Python dict keyed by thread_id.
    # Lost when this Python process exits — fine for our run-once flow.
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Starting pipeline (thread_id={thread_id}) ===\n")
    final_state = graph.invoke({}, config=config)

    if final_state.get("messages"):
        print("\n=== Run summary ===")
        print(final_state["messages"][-1].content)

    return final_state


if __name__ == "__main__":
    cli_thread_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(thread_id=cli_thread_id)
