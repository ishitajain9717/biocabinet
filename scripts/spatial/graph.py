"""Stub spatial transcriptomics pipeline as a 1-node LangGraph.

Replaces the earlier raise-NotImplementedError pattern so the orchestrator
can compose this as a subgraph just like bulk/scrna without special-casing.

Real implementation will load Visium / Stereo-seq / Slide-seq .h5ad with
spatial coordinates, run spatial-aware QC, normalization, neighborhood
graphs, and produce region-level count summaries for downstream enrichment.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class _SpatialStubState(TypedDict):
    error:    str | None
    messages: Annotated[list, add_messages]


def _node_not_implemented(state: _SpatialStubState) -> dict:
    msg = (
        "Spatial transcriptomics pipeline is not built yet. "
        "Pick 'bulk_rnaseq' or 'scrna' for now."
    )
    return {"error": msg, "messages": [AIMessage(content=msg)]}


def build_graph() -> StateGraph:
    """Build the (currently stub) spatial StateGraph."""
    workflow = StateGraph(_SpatialStubState)
    workflow.add_node("not_implemented", _node_not_implemented)
    workflow.set_entry_point("not_implemented")
    workflow.add_edge("not_implemented", END)
    return workflow


def run_pipeline(thread_id: str | None = None) -> dict:
    """Stand-alone runner for the spatial stub (returns the not-implemented msg)."""
    from langgraph.checkpoint.memory import MemorySaver
    g = build_graph().compile(checkpointer=MemorySaver())
    return g.invoke({}, config={"configurable": {"thread_id": thread_id or "spatial"}})
