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


def _read_deg(
    deg_full_path: str | Path | None, deg_sig_path: str | Path | None
) -> dict[str, Any]:
    """Read DESeq2 result TSVs into structured dicts."""
    result: dict[str, Any] = {
        "n_genes_tested": None,
        "n_deg_significant": None,
        "deg_top": [],
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
            for r in rows[:20]:
                try:
                    lfc = float(r.get("log2FoldChange") or 0)
                    padj = float(r.get("padj") or 1)
                    top.append(
                        {
                            "gene_id": r.get("gene_id", r.get("", "")),
                            "log2fc": round(lfc, 4),
                            "padj": round(padj, 6),
                            "direction": "up" if lfc >= 0 else "down",
                        }
                    )
                except (ValueError, KeyError):
                    continue
            result["deg_top"] = top
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
# public entry point
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
            lines.append(
                f"  {arrow} {g['gene_id']}"
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
