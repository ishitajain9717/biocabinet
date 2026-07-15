"""Lightweight CI checks: verify core dependencies import (no data files, no GPU).

Optional extras ([gnn], [spatial], [trajectory]) are skipped when not installed
so the base CI matrix passes without the heavy optional deps.
"""

import importlib.util

import pytest


def test_langgraph_imports():
    import langgraph  # noqa: F401
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401


def test_bulk_stack():
    import pandas  # noqa: F401
    import pydeseq2  # noqa: F401


def test_scrna_stack():
    import scanpy  # noqa: F401


def test_rag_stack():
    import transformers  # noqa: F401


# ── Optional extras — skip cleanly when not installed ──────────────────────


@pytest.mark.skipif(
    importlib.util.find_spec("torch_geometric") is None,
    reason="optional [gnn] extra not installed",
)
def test_gnn_stack():
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401


@pytest.mark.skipif(
    importlib.util.find_spec("squidpy") is None,
    reason="optional [spatial] extra not installed",
)
def test_spatial_stack():
    import harmonypy  # noqa: F401
    import squidpy  # noqa: F401


@pytest.mark.skipif(
    importlib.util.find_spec("palantir") is None,
    reason="optional [trajectory] extra not installed",
)
def test_trajectory_stack():
    import palantir  # noqa: F401


# ── Core package structure ──────────────────────────────────────────────────


def test_package_modules_import():
    from scripts.bulk_rnaseq import graph_state  # noqa: F401
    from scripts.common import node_result  # noqa: F401
    from scripts.enrichment import graph_state as enrich_state  # noqa: F401
    from scripts.rag import graph_state as rag_state  # noqa: F401
    from scripts.scrna import graph_state as scrna_state  # noqa: F401
