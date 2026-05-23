"""Shared state for the enrichment (GNN-PPI) LangGraph subgraph."""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages

from scripts.common.node_result import NodeResult
from scripts.enrichment.config import EnrichmentConfig


class EnrichmentState(TypedDict):
    config:        EnrichmentConfig
    node_history:  list[NodeResult]

    # ---- populated by node_load_data ----
    # we never put the PyG Data object into state (too large; not serializable
    # for the SQLite checkpointer). Each node rebuilds the dataset from disk
    # using paths in `config`. We stash *summary stats* and the splits dict in
    # the state so downstream nodes can refer to them without recomputing.
    n_proteins:    int
    n_edges:       int
    n_components:  int
    esm_dim:       int
    pathway_dim:   int
    pathway_coverage:  float            # fraction of proteins with pathway features

    # ---- populated by node_train ----
    best_ckpt_path:    Optional[str]
    last_ckpt_path:    Optional[str]
    best_val_f1:       Optional[float]
    best_epoch:        Optional[int]
    history_path:      Optional[str]

    # ---- populated by node_eval ----
    test_metrics_path: Optional[str]
    bucket_metrics:    Optional[dict[str, Any]]   # name → {n, p, r, f1, loss}

    # ---- populated by node_infer (only if cfg.pairs_path provided) ----
    inference_path:    Optional[str]
    n_pairs_predicted: Optional[int]
    n_pairs_skipped:   Optional[int]

    # ---- LLM messages + error sentinel ----
    messages:      Annotated[list, add_messages]
    error:         Optional[str]
