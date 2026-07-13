"""Assemble rich pipeline context for the RAG chat loop.

Called by the orchestrator after all pipeline nodes finish.  Reads structured
artefacts from disk and returns a flat dict that can be serialised to SQLite
and passed into RagChatState.pipeline_results.

The dict has this shape (all keys optional / None when the step did not run):

    {
        "conditions": {"treated": str, "reference": str},
        "n_samples_ok": int,
        "n_genes_tested": int,
        "n_deg_significant": int,
        "deg_top": [                          # up to 20 rows, sorted by padj
            {"gene_id": str,
             "log2fc": float,
             "padj":   float,
             "direction": "up"|"down"},
            ...
        ],
        "inference_n_predicted": int,
        "inference_n_skipped":   int,
        "inference_top": [                    # up to 10 predicted interactions
            {"ensp_a": str, "ensp_b": str,
             "classes": [str, ...],
             "top_prob": float},
            ...
        ],
    }
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# DEG reader
# ---------------------------------------------------------------------------


def _ensg_to_gene_info(ensg_ids: list[str]) -> dict[str, dict]:
    """Map ENSG IDs to {entrez_id, symbol} via mygene (single batched call).

    Returns {ensg_id: {"entrez": "7157", "symbol": "TP53"}}.
    Missing mappings are silently omitted; network/import failures return {}.
    """
    if not ensg_ids:
        return {}
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        hits = mg.querymany(
            ensg_ids,
            scopes="ensembl.gene",
            fields="entrezgene,symbol",
            species="human",
            verbose=False,
        )
        mapping: dict[str, dict] = {}
        for h in hits:
            if h.get("notfound"):
                continue
            entrez = h.get("entrezgene")
            symbol = h.get("symbol")
            if entrez:
                mapping[h["query"]] = {
                    "entrez": str(int(entrez)),
                    "symbol": symbol or h["query"],
                }
        return mapping
    except Exception:
        return {}


def _read_deg(
    deg_full_path: str | Path | None, deg_sig_path: str | Path | None
) -> dict[str, Any]:
    """Read DESeq2 result TSVs into structured dicts.

    Also translates ENSG IDs to Entrez IDs (used as gene_filter keys in the
    RAG index, which is keyed by Entrez ID).
    """
    result: dict[str, Any] = {
        "n_genes_tested": None,
        "n_deg_significant": None,
        "deg_top": [],
        "deg_entrez_ids": [],  # Entrez IDs for RAG gene_filter
    }

    # Count total genes tested from the full table
    full = Path(deg_full_path) if deg_full_path else None
    if full and full.exists():
        try:
            with full.open() as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            result["n_genes_tested"] = len(rows)
        except Exception:
            pass

    # Read significant genes
    sig = Path(deg_sig_path) if deg_sig_path else None
    if sig and sig.exists():
        try:
            with sig.open() as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            result["n_deg_significant"] = len(rows)

            top: list[dict] = []
            ensg_ids: list[str] = []
            for r in rows[:20]:
                try:
                    lfc = float(r.get("log2FoldChange") or 0)
                    padj = float(r.get("padj") or 1)
                    gid = r.get("gene_id", r.get("", ""))
                    ensg_ids.append(gid)
                    top.append(
                        {
                            "gene_id": gid,
                            "log2fc": round(lfc, 4),
                            "padj": round(padj, 6),
                            "direction": "up" if lfc >= 0 else "down",
                        }
                    )
                except (ValueError, KeyError):
                    continue
            result["deg_top"] = top

            # Translate ENSG → Entrez + symbol for RAG gene filter and prompt
            print(
                f"[pipeline_context] looking up symbols for"
                f" {len(ensg_ids)} DEGs via mygene...",
                flush=True,
            )
            gene_info = _ensg_to_gene_info(ensg_ids)
            result["deg_entrez_ids"] = [v["entrez"] for v in gene_info.values()]
            # Annotate each DEG entry with Entrez ID and symbol
            for entry in top:
                info = gene_info.get(entry["gene_id"], {})
                entry["entrez_id"] = info.get("entrez")
                entry["symbol"] = info.get("symbol", entry["gene_id"])
            print(
                f"[pipeline_context] resolved"
                f" {len(result['deg_entrez_ids'])}/{len(ensg_ids)} genes"
                " (Entrez + symbol)",
                flush=True,
            )
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# GNN inference reader
# ---------------------------------------------------------------------------


def _read_inference(inference_path: str | Path | None) -> dict[str, Any]:
    """Read inference.json into structured dicts."""
    result: dict[str, Any] = {
        "inference_n_predicted": None,
        "inference_n_skipped": None,
        "inference_top": [],
    }
    inf = Path(inference_path) if inference_path else None
    if not inf or not inf.exists():
        return result

    try:
        data = json.loads(inf.read_text())
        result["inference_n_predicted"] = data.get("n_predicted")
        result["inference_n_skipped"] = data.get("n_skipped")

        top: list[dict] = []
        for r in data.get("results") or []:
            classes = r.get("predicted_classes") or []
            probs = r.get("probabilities") or {}
            if not classes:
                continue
            top_prob = max(probs.values()) if probs else 0.0
            top.append(
                {
                    "ensp_a": r.get("ensp_a", ""),
                    "ensp_b": r.get("ensp_b", ""),
                    "classes": classes,
                    "top_prob": round(top_prob, 4),
                }
            )
            if len(top) >= 10:
                break
        result["inference_top"] = sorted(top, key=lambda x: -x["top_prob"])
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# scRNA reader
# ---------------------------------------------------------------------------


def _read_scrna_markers(markers_path: str | Path | None) -> dict[str, Any]:
    """Read the markers.csv produced by node_markers into a structured dict.

    Returns:
        {
            "n_clusters_with_markers": int,
            "markers_top": [
                {"cluster": str, "gene": str, "logfoldchange": float,
                 "pval_adj": float},
                ...  # up to 5 genes × up to 10 clusters
            ],
        }
    """
    result: dict[str, Any] = {"n_clusters_with_markers": None, "markers_top": []}
    p = Path(markers_path) if markers_path else None
    if not p or not p.exists():
        return result

    try:
        with p.open() as fh:
            rows = list(csv.DictReader(fh))

        clusters_seen: dict[str, int] = {}  # cluster → count of genes taken
        top: list[dict] = []
        for r in rows:
            cluster = r.get("cluster", "?")
            taken = clusters_seen.get(cluster, 0)
            if taken >= 5:
                continue
            try:
                top.append(
                    {
                        "cluster": cluster,
                        "gene": r.get("gene", "?"),
                        "logfoldchange": round(float(r.get("logfoldchange") or 0), 3),
                        "pval_adj": round(float(r.get("pval_adj") or 1), 5),
                    }
                )
                clusters_seen[cluster] = taken + 1
            except (ValueError, KeyError):
                continue

        result["n_clusters_with_markers"] = len(clusters_seen)
        result["markers_top"] = top
    except Exception:
        pass

    return result


def assemble_scrna_results(
    n_cells: int | None,
    n_genes: int | None,
    n_clusters: int | None,
    markers_path: str | None,
) -> dict[str, Any]:
    """Build the pipeline_results dict for an scRNA run.

    Shape is parallel to assemble_pipeline_results so the RAG answerer
    receives the same ``pipeline_results`` key regardless of modality.
    """
    results: dict[str, Any] = {
        "data_type": "scrna",
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_clusters": n_clusters,
    }
    results.update(_read_scrna_markers(markers_path))
    return results


# ---------------------------------------------------------------------------
# public entry point (bulk)
# ---------------------------------------------------------------------------


def assemble_pipeline_results(
    deg_full_path: str | None,
    deg_sig_path: str | None,
    n_deg_significant: int | None,
    inference_path: str | None,
    n_samples_ok: int | None,
    conditions: dict | None,
) -> dict[str, Any]:
    """Build the full pipeline_results dict from all available artefacts."""
    results: dict[str, Any] = {
        "conditions": conditions or {},
        "n_samples_ok": n_samples_ok,
    }

    deg_data = _read_deg(deg_full_path, deg_sig_path)
    # Prefer explicitly passed count (already in state) over re-reading file
    if n_deg_significant is not None:
        deg_data["n_deg_significant"] = n_deg_significant
    results.update(deg_data)

    results.update(_read_inference(inference_path))

    return results


# ---------------------------------------------------------------------------
# format for LLM prompt
# ---------------------------------------------------------------------------


def format_pipeline_results_for_prompt(pr: dict[str, Any]) -> str:
    """Return a concise text block to inject into the LLM prompt."""
    if not pr:
        return "(no pipeline results available)"

    lines: list[str] = ["=== Pipeline run summary ==="]

    conds = pr.get("conditions") or {}
    if conds:
        lines.append(
            f"Comparison: {conds.get('treated', '?')} vs "
            f"{conds.get('reference', '?')}"
        )

    if pr.get("n_samples_ok") is not None:
        lines.append(f"Samples that completed: {pr['n_samples_ok']}")

    n_tested = pr.get("n_genes_tested")
    n_sig = pr.get("n_deg_significant")
    if n_tested is not None:
        lines.append(f"Genes tested for differential expression: {n_tested}")
    if n_sig is not None:
        lines.append(
            f"Significant DEGs (padj < threshold, |log2FC| > threshold): {n_sig}"
        )

    deg_top = pr.get("deg_top") or []
    if deg_top:
        lines.append("\nTop differentially expressed genes:")
        for g in deg_top[:10]:
            arrow = "▲" if g["direction"] == "up" else "▼"
            # Show symbol if available, fall back to ENSG ID
            label = g.get("symbol") or g["gene_id"]
            if label != g["gene_id"]:
                label = f"{label} ({g['gene_id']})"
            lines.append(
                f"  {arrow} {label}"
                f"  log2FC={g['log2fc']:+.2f}  padj={g['padj']:.2e}"
            )

    n_pred = pr.get("inference_n_predicted")
    n_skip = pr.get("inference_n_skipped")
    inf_top = pr.get("inference_top") or []
    if n_pred is not None:
        lines.append(
            f"\nGNN protein-protein interaction inference: "
            f"{n_pred} predicted, {n_skip} skipped (not in graph)"
        )
    if inf_top:
        lines.append("Top predicted interactions:")
        for i in inf_top[:5]:
            lines.append(
                f"  {i['ensp_a']} — {i['ensp_b']}: "
                f"{', '.join(i['classes'])} (p={i['top_prob']:.2f})"
            )

    return "\n".join(lines)


def format_scrna_results_for_prompt(pr: dict[str, Any]) -> str:
    """Return a concise text block for scRNA pipeline_results
    to inject into the LLM prompt."""
    if not pr:
        return "(no scRNA pipeline results available)"

    lines: list[str] = ["=== scRNA-seq run summary ==="]

    if pr.get("n_cells") is not None:
        lines.append(f"Cells after QC + filtering: {pr['n_cells']}")
    if pr.get("n_genes") is not None:
        lines.append(f"Genes after filtering: {pr['n_genes']}")
    if pr.get("n_clusters") is not None:
        lines.append(f"Leiden clusters identified: {pr['n_clusters']}")

    n_clust_markers = pr.get("n_clusters_with_markers")
    if n_clust_markers is not None:
        lines.append(f"Clusters with marker genes: {n_clust_markers}")

    markers = pr.get("markers_top") or []
    if markers:
        lines.append("\nTop marker genes per cluster (up to 5 per cluster):")
        current_cluster = None
        for m in markers:
            if m["cluster"] != current_cluster:
                current_cluster = m["cluster"]
                lines.append(f"  Cluster {current_cluster}:")
            direction = "▲" if m["logfoldchange"] >= 0 else "▼"
            lines.append(
                f"    {direction} {m['gene']}"
                f"  log2FC={m['logfoldchange']:+.2f}"
                f"  padj={m['pval_adj']:.2e}"
            )

    return "\n".join(lines)
