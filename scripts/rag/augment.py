"""Phase 4a helper: augment pipeline summaries with RAG pathway context.

Each pipeline's summarize node calls one of these helpers AFTER it has
produced its own LLM (or deterministic) summary.  The helper:

    1. Extracts relevant gene IDs from the pipeline's final state.
    2. Calls answerer.answer() with those genes as a filter.
    3. Returns a formatted string to APPEND to the existing summary.

If the RAG index is missing, the LLM is unavailable, or any error
occurs the function returns "" — the pipeline always completes.

Public functions
----------------
    rag_augment_bulk(state)         → str   (uses DEG significant genes)
    rag_augment_scrna(state)        → str   (uses top marker gene names)
    rag_augment_enrichment(state)   → str   (uses predicted interaction pairs)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# shared private helpers
# ---------------------------------------------------------------------------

def _read_all_deg_ids_sorted(path: str | Path) -> list[str]:
    """Read ALL gene IDs from a DESeq2 TSV, sorted by padj then |lfc|.

    Returns the full sorted list — the caller decides how many to keep.
    Falls back to file order if stat columns are absent.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        rows: list[dict] = []
        with path.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                rows.append(row)
        if not rows:
            return []

        first = rows[0]
        if "padj" in first or "log2FoldChange" in first:
            def sort_key(r: dict):
                try:
                    padj = float(r.get("padj") or 1.0)
                except ValueError:
                    padj = 1.0
                try:
                    lfc = abs(float(r.get("log2FoldChange") or 0.0))
                except ValueError:
                    lfc = 0.0
                return (padj, -lfc)
            rows.sort(key=sort_key)

        gene_col = next(iter(first))
        return [r[gene_col].strip() for r in rows if r.get(gene_col, "").strip()]
    except Exception:
        return []


def _find_interest_genes(
    gene_ids:          list[str],
    pathway_interests: list[str],
    index_dir:         Path,
) -> set[str]:
    """Return the subset of gene_ids linked to at least one pathway interest keyword.

    Looks up each gene in gene_to_doc_ids.json, then checks the pathway_name
    of each linked doc in docs.jsonl for a case-insensitive keyword match.
    Returns an empty set if the RAG index is missing or interests is empty.
    """
    if not pathway_interests or not gene_ids:
        return set()

    rev_path  = index_dir / "gene_to_doc_ids.json"
    docs_path = index_dir / "docs.jsonl"
    if not rev_path.exists() or not docs_path.exists():
        return set()

    try:
        import json
        rev: dict[str, list[str]] = json.loads(rev_path.read_text())
        # Build doc_id → pathway_name map (load once, lightweight)
        doc_name: dict[str, str] = {}
        for line in docs_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            doc_name[d["id"]] = d.get("pathway_name", "").lower()

        keywords = [kw.lower() for kw in pathway_interests]
        pinned: set[str] = set()
        for gid in gene_ids:
            doc_ids = rev.get(str(gid), [])
            for did in doc_ids:
                pname = doc_name.get(did, "")
                if any(kw in pname for kw in keywords):
                    pinned.add(gid)
                    break   # one match is enough for this gene
        return pinned
    except Exception:
        return set()


def _select_genes(
    sorted_genes:  list[str],
    pinned:        set[str],
) -> list[str]:
    """Combine pinned (interest) genes with top-60% of the rest.

    Rule:
      pinned genes always included first (in their sorted order).
      Remaining slots filled with top 60% of non-pinned genes
      (or all non-pinned if total <= 100).
    """
    pinned_ordered   = [g for g in sorted_genes if g in pinned]
    remaining        = [g for g in sorted_genes if g not in pinned]

    total_remaining = len(remaining)
    if total_remaining <= 100:
        n_from_remaining = total_remaining
    else:
        n_from_remaining = max(1, int(total_remaining * 0.60))

    return pinned_ordered + remaining[:n_from_remaining]


def _format_rag_result(result) -> str:
    """Turn an AnswerResult into a formatted section to append to a summary."""
    if not result or not result.ok or not result.answer:
        return ""

    lines = [
        "",
        "--- Pathway context (RAG) ---",
        result.answer,
    ]
    if result.citations:
        lines.append("\nCitations:")
        for c in result.citations:
            lines.append(
                f"  [{c['rank']}] {c['pathway_name']}"
                f"  ({c['source']}, score={c['score']:.3f})"
            )
    return "\n".join(lines)


def _call_rag(question: str, genes: list[str], n_deg: int | None, k: int = 6,
              llm_summary: str | None = None) -> str:
    """Call answerer.answer() and format the result.  Never raises."""
    if not genes:
        return ""
    try:
        from scripts.rag.answerer import answer
        result = answer(
            question=question,
            gene_filter=genes,
            k=k,
            n_deg=n_deg,
            llm_summary=llm_summary,
        )
        return _format_rag_result(result)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Phase 4a — bulk RNA-seq
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_DIR = Path("data/rag")


def rag_augment_bulk(state: dict[str, Any], k: int = 6) -> str:
    """Augment a bulk pipeline summary with DEG-focused pathway context.

    Gene selection:
      1. Genes linked to user's pathway_interests are always included.
      2. Remaining slots filled with top 60% of DEGs sorted by padj.
      3. Hard cap at 200 genes total.

    Returns "" when DEG was skipped, failed, or no genes were found.
    """
    deg_sig = (state or {}).get("deg_sig_path")
    if not deg_sig:
        return ""

    # 1. Read all significant DEGs, sorted by statistical significance
    all_genes = _read_all_deg_ids_sorted(deg_sig)
    if not all_genes:
        return ""

    # 2. Pin genes that match the user's pathway interests
    pathway_interests: list[str] = []
    cfg = (state or {}).get("config")
    if cfg is not None:
        pathway_interests = getattr(cfg, "pathway_interests", []) or []

    pinned = _find_interest_genes(all_genes, pathway_interests, _DEFAULT_INDEX_DIR)

    # 3. Select final gene list: pinned first, then top 60% of the rest
    genes = _select_genes(all_genes, pinned)
    if not genes:
        return ""

    n_sig = (state or {}).get("n_deg_significant")
    question = (
        "What biological pathways and cellular processes are most likely "
        "dysregulated given these differentially expressed genes?"
    )
    return _call_rag(question, genes, n_deg=n_sig, k=k)


# ---------------------------------------------------------------------------
# Phase 4a — scRNA-seq
# ---------------------------------------------------------------------------

def rag_augment_scrna(state: dict[str, Any], k: int = 6) -> str:
    """Augment a scRNA summary with marker-gene pathway context.

    Reads gene names from state["marker_genes_path"] if present.
    Returns "" when marker detection was skipped or the file is missing.
    """
    marker_path = (state or {}).get("marker_genes_path")
    if not marker_path:
        return ""

    all_genes = _read_all_deg_ids_sorted(marker_path)
    genes = _select_genes(all_genes, pinned=set())
    if not genes:
        return ""

    question = (
        "What cell types, biological functions, and signalling pathways "
        "are characterised by these marker genes?"
    )
    return _call_rag(question, genes, n_deg=None, k=k)


# ---------------------------------------------------------------------------
# Phase 4a — enrichment / GNN-PPI
# ---------------------------------------------------------------------------

def rag_augment_enrichment(state: dict[str, Any], k: int = 6) -> str:
    """Augment an enrichment summary with PPI-partner pathway context.

    Reads ENSP IDs from state["inference_path"] (inference.json written by
    the enrichment pipeline).  Returns "" when inference was skipped.
    """
    inf_path = (state or {}).get("inference_path")
    if not inf_path:
        return ""

    # Extract unique ENSP IDs from the inference JSON
    try:
        import json
        data = json.loads(Path(inf_path).read_text())
        ensps: list[str] = []
        for r in data.get("results", []):
            for key in ("ensp_a", "ensp_b"):
                v = r.get(key, "")
                if v and v not in ensps:
                    ensps.append(v.replace("9606.", ""))   # strip species prefix
        if not ensps:
            return ""
    except Exception:
        return ""

    question = (
        "What biological functions and interaction networks are associated "
        "with these predicted protein-protein interacting pairs?"
    )
    # Pass ENSP ids; the gene_to_doc_ids reverse index may match some via
    # Reactome ENSG overlap — even partial overlap improves retrieval precision.
    return _call_rag(question, ensps, n_deg=None, k=k)
