"""Lightweight CI checks: verify core dependencies import (no data files, no GPU)."""


def test_langgraph_imports():
    import langgraph  # noqa: F401
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401


def test_bulk_stack():
    import pandas  # noqa: F401
    import pydeseq2  # noqa: F401


def test_scrna_stack():
    import scanpy  # noqa: F401


def test_enrichment_stack():
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401


def test_rag_stack():
    import transformers  # noqa: F401


def test_package_modules_import():
    from scripts.bulk_rnaseq import graph_state  # noqa: F401
    from scripts.common import node_result  # noqa: F401
    from scripts.enrichment import graph_state as enrich_state  # noqa: F401
    from scripts.rag import graph_state as rag_state  # noqa: F401
    from scripts.scrna import graph_state as scrna_state  # noqa: F401
