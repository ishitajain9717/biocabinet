"""Build, compile, and run the enrichment (GNN-PPI) pipeline as a LangGraph.

Flow:
    collect_config → load_data → train → eval → infer → summarize → END
                  ↘ error_node ↗ (taken on first failed work node)

Conditional edges after every work node short-circuit to error_node at the
first failure, so we don't keep training/evaluating after a load failure
(or evaluating after a training crash).

Run with:
    python3 -m scripts.enrichment.graph
    python3 -m scripts.enrichment.graph my_thread_id        # resume by thread id
"""
from __future__ import annotations

import sys
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from scripts.enrichment.graph_nodes import (
    graph_node_collect_config,
    graph_node_error,
    graph_node_eval,
    graph_node_infer,
    graph_node_load_data,
    graph_node_summarize,
    graph_node_train,
)
from scripts.enrichment.graph_state import EnrichmentState


def _route_or_continue(next_step: str):
    """Build a conditional-edge router: error_node if state.error, else next_step."""
    def router(state: EnrichmentState) -> str:
        return "error_node" if state.get("error") else next_step
    return router


def build_graph() -> StateGraph:
    """Construct the enrichment StateGraph (not yet compiled)."""
    workflow = StateGraph(EnrichmentState)

    workflow.add_node("collect_config", graph_node_collect_config)
    workflow.add_node("load_data",      graph_node_load_data)
    workflow.add_node("train",          graph_node_train)
    workflow.add_node("eval",           graph_node_eval)
    workflow.add_node("infer",          graph_node_infer)
    workflow.add_node("summarize",      graph_node_summarize)
    workflow.add_node("error_node",     graph_node_error)

    workflow.set_entry_point("collect_config")

    # collect_config never sets error itself → plain edge
    workflow.add_edge("collect_config", "load_data")

    # After every work node: continue if ok, otherwise divert to error_node
    for prev, nxt in [
        ("load_data", "train"),
        ("train",     "eval"),
        ("eval",      "infer"),
        ("infer",     "summarize"),
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
    """Compile and invoke the enrichment graph. Returns the final state."""
    if thread_id is None:
        thread_id = "enrichment_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    workflow = build_graph()
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Starting enrichment pipeline (thread_id={thread_id}) ===\n")
    final_state = graph.invoke({}, config=config)

    if final_state.get("messages"):
        print("\n=== Run summary ===")
        print(final_state["messages"][-1].content)

    return final_state


if __name__ == "__main__":
    cli_thread_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(thread_id=cli_thread_id)
