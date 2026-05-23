"""Shared state (the 'baton') passed between every node in the pipeline graph.

LangGraph requires state to be a TypedDict so it knows the exact shape
of the object it is checkpointing to SQLite after each node finishes.
"""
from __future__ import annotations

# Annotated lets us attach metadata to a type hint without changing the type.
# add_messages is LangGraph's reducer for the messages list — instead of
# replacing the list on every node update, it *appends* to it.
# This is how chat history accumulates across nodes rather than being overwritten.
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from scripts.bulk_rnaseq.config import PreprocessingConfig
from scripts.common.node_result import NodeResult


class PipelineState(TypedDict):
    # ----- run-wide config -----
    # Set once by the first node (collect_config) and read by every other node.
    # Holds all paths, sample list, threads, normalizations, etc.
    config: PreprocessingConfig

    # ----- per-node audit trail -----
    # Every node appends its NodeResult here.
    # The LLM summarizer reads this at the end to write a plain-English report.
    node_history: list[NodeResult]

    # ----- inter-node data handoffs -----
    # Populated by the featurecounts node; consumed by the normalize node.
    # Stored as list[tuple[str, str]] (sample_name, path_as_str) rather than
    # Path objects because the SQLite checkpointer serialises to JSON, which
    # does not know how to handle pathlib.Path.
    count_results: list[tuple[str, str]]

    # Written by the normalize node — maps norm name → output TSV path.
    # e.g. {"tpm": "/path/tpm.tsv", "fpkm": "/path/fpkm.tsv"}
    norm_outputs: dict[str, str]

    # Names of samples that did not reach featurecounts successfully.
    # Kept separate so the LLM can call them out explicitly in the summary.
    failed_samples: list[str]

    # ----- DEG outputs -----
    # Filled by graph_node_deg if cfg.enable_deg=True. None when DEG was
    # skipped (cfg.enable_deg=False) or failed. The orchestrator's enrichment
    # auto-chain reads `deg_pairs_path` and uses it as the candidate-pair
    # input for inference.
    deg_full_path:    str | None
    deg_sig_path:     str | None
    deg_pairs_path:   str | None
    n_deg_significant: int | None

    # ----- LLM conversation memory -----
    # Annotated[list, add_messages] is a LangGraph convention that means:
    # "this is a list, and when a node writes to it, append — don't replace."
    # Every HumanMessage / AIMessage / ToolMessage the LLM exchanges lives here.
    # Because of add_messages, the full conversation history builds up naturally
    # as the graph runs node → node.
    messages: Annotated[list, add_messages]

    # ----- error channel -----
    # Any node can write a string here to signal a hard, unrecoverable failure
    # (e.g. GTF file missing, genome index corrupt).
    # The graph router checks this field and short-circuits to an error node
    # instead of continuing with the next step.
    # None = no error, everything is fine.
    error: str | None
