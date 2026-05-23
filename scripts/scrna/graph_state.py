from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from scripts.common.node_result import NodeResult
from scripts.scrna.config import ScrnaConfig

class ScrnaState(TypedDict):
    config:        ScrnaConfig
    node_history:  list[NodeResult]

    # adata is too big for state — we save it to disk between nodes
    # and pass the *path* through state (same pattern as count_results)
    adata_path:    str

    # filled in progressively as nodes run
    n_cells:       int
    n_genes:       int
    qc_metrics:    dict
    n_clusters:    int
    markers_path:  str | None

    messages:      Annotated[list, add_messages]
    error:         str | None