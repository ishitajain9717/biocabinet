"""Predict interaction types for novel candidate edges (e.g. DEG pairs).

Workflow:
    1. Load trained checkpoint + dataset (provides PPI graph context).
    2. Read candidate pairs (TSV with two ENSP columns, OR list[(str, str)]).
    3. Map each ENSP to the dataset's internal node index.
    4. Run model.fuse + model.gin_convs once to get per-node post-GIN features.
    5. For each candidate pair, gather the two node vectors, fuse (mul or concat),
       and pass through model.fc2 to get class logits.
    6. Apply sigmoid; threshold at 0.5 for predicted class set.
    7. Save JSON: list of {ensp_a, ensp_b, probabilities, predicted_classes,
       in_graph_a, in_graph_b}.

Why a custom forward
    The model.forward() expects an edge_index lookup. For novel pairs we
    don't have a slot in edge_index, so we replicate the GIN stack manually
    here, then do edge prediction by direct indexing.

Usage:
    python -m scripts.enrichment.inference \\
        --ppi data/ppi_SHS27k.tsv \\
        --esm data/esm2_embeddings_SHS27k.pt \\
        --pathway data/pathway/06_pathway_embeddings.pt \\
        --ckpt runs/.../gnn_model_valid_best.ckpt \\
        --pairs my_pairs.tsv \\
        --output runs/.../inference.json

PPI label classes (7) — STRING evidence channels:
    0: reaction
    1: binding
    2: ptmod (post-translational modification)
    3: activation
    4: inhibition
    5: catalysis
    6: expression
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from scripts.enrichment.gnn_data import GNNDataset
from scripts.enrichment.gnn_model import GIN_PPI
from scripts.enrichment.gnn_train import _resolve_device


# STRING/SHS27k label channel order (from the GNN-PPI paper)
PPI_CLASS_NAMES = ["reaction", "binding", "ptmod", "activation",
                   "inhibition", "catalysis", "expression"]


# ---------------------------------------------------------------------------
# pair loading
# ---------------------------------------------------------------------------

def read_pairs(path: Path) -> list[tuple[str, str]]:
    """Read candidate pairs from a TSV. Two columns expected: ENSP_a, ENSP_b.
    Header row optional; rows starting with '#' or empty are ignored."""
    pairs: list[tuple[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(",")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            # skip a header row if present (any column name we recognize)
            header_tokens = {"protein1", "protein2", "protein_a", "protein_b",
                             "ensp_a", "ensp_b", "ensp1", "ensp2", "a", "b"}
            if a.lower() in header_tokens or b.lower() in header_tokens:
                continue
            pairs.append((a, b))
    return pairs


def _normalize_protein_id(pid: str, known: set[str]) -> str | None:
    """Try a few naming conventions to match a candidate ENSP to known nodes."""
    if pid in known:
        return pid
    if not pid.startswith("9606.") and f"9606.{pid}" in known:
        return f"9606.{pid}"
    bare = pid.split(".", 1)[-1] if "." in pid else pid
    if bare in known:
        return bare
    return None


# ---------------------------------------------------------------------------
# model forward — exposes per-node post-GIN features
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_post_gin_node_features(
    model:     GIN_PPI,
    x:         torch.Tensor,
    pathway_x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Replicates model.forward() up to (but not including) the edge gather.
    Returns (N, num_classes) per-node logit-space feature."""
    model.eval()
    node_feat = torch.cat([x, pathway_x], dim=1)
    node_feat = model.fuse(node_feat)

    xs: list[torch.Tensor] = []
    for gin in model.gin_convs:
        node_feat = gin(node_feat, edge_index)
        xs.append(node_feat)
    if model.use_jk:
        node_feat = model.jump(xs)

    node_feat = F.relu(model.lin1(node_feat))
    node_feat = model.lin2(node_feat)              # (N, num_classes)
    return node_feat


# ---------------------------------------------------------------------------
# main inference function
# ---------------------------------------------------------------------------

def predict_pairs(
    ppi_path:     Path,
    esm_path:     Path,
    pathway_path: Optional[Path],
    ckpt_path:    Path,
    pairs:        list[tuple[str, str]],
    out_path:     Optional[Path] = None,
    device:       str = "auto",
    threshold:    float = 0.5,
    model_kwargs: Optional[dict] = None,
) -> dict:
    dev = _resolve_device(device)
    print(f"[infer] device: {dev}")

    # ---- dataset (graph context) ----
    dataset = GNNDataset(
        ppi_path=ppi_path,
        esm_emb_path=esm_path,
        pathway_emb_path=pathway_path,
        verbose=True,
    )
    data = dataset.to_pyg_data().to(dev)

    # ---- model + checkpoint ----
    mk = model_kwargs or {}
    if dataset.pathway_x is not None:
        mk.setdefault("pathway_dim", dataset.pathway_x.shape[1])
    mk.setdefault("esm_dim", dataset.esm_dim)

    model = GIN_PPI(**mk).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"[infer] loaded checkpoint: {ckpt_path.name}")

    # ---- per-node features (GIN forward, no edge gather) ----
    node_feat = compute_post_gin_node_features(
        model, data.x, data.pathway_x, data.edge_index,
    )

    # ---- map candidate pairs to graph indices ----
    known = set(dataset.protein_to_idx.keys())
    print(f"[infer] {len(pairs)} candidate pairs, {len(known)} graph proteins")

    results: list[dict] = []
    n_skipped = 0
    n_predicted = 0

    for ensp_a, ensp_b in pairs:
        norm_a = _normalize_protein_id(ensp_a, known)
        norm_b = _normalize_protein_id(ensp_b, known)
        rec = {
            "ensp_a":     ensp_a,
            "ensp_b":     ensp_b,
            "in_graph_a": norm_a is not None,
            "in_graph_b": norm_b is not None,
        }
        if norm_a is None or norm_b is None:
            n_skipped += 1
            results.append(rec)
            continue

        idx_a = dataset.protein_to_idx[norm_a]
        idx_b = dataset.protein_to_idx[norm_b]

        with torch.no_grad():
            x1 = node_feat[idx_a]
            x2 = node_feat[idx_b]
            if model.feature_fusion == "concat":
                edge_feat = torch.cat([x1, x2], dim=0)
            else:
                edge_feat = torch.mul(x1, x2)
            logits = model.fc2(edge_feat)              # (num_classes,)
            probs = torch.sigmoid(logits).cpu().tolist()

        rec["probabilities"] = {
            PPI_CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)
        }
        rec["predicted_classes"] = [
            PPI_CLASS_NAMES[i] for i, p in enumerate(probs) if p > threshold
        ]
        n_predicted += 1
        results.append(rec)

    print(f"[infer] predicted={n_predicted}, skipped (not in graph)={n_skipped}")

    summary = {
        "ckpt_path":  str(ckpt_path),
        "ppi_path":   str(ppi_path),
        "n_pairs":    len(pairs),
        "n_predicted": n_predicted,
        "n_skipped":  n_skipped,
        "threshold":  threshold,
        "results":    results,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[infer] wrote → {out_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict interaction types for novel ENSP pairs.")
    p.add_argument("--ppi",     required=True, type=Path)
    p.add_argument("--esm",     required=True, type=Path)
    p.add_argument("--pathway", type=Path, default=None)
    p.add_argument("--ckpt",    required=True, type=Path)
    p.add_argument("--pairs",   required=True, type=Path,
                   help="TSV/CSV with 2 columns: ENSP_a, ENSP_b")
    p.add_argument("--output",  type=Path, default=None,
                   help="JSON output (default: <ckpt-dir>/inference.json)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device",    default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    pairs = read_pairs(args.pairs)
    print(f"[infer] read {len(pairs)} candidate pairs from {args.pairs}")
    out = args.output or (args.ckpt.parent / "inference.json")
    predict_pairs(
        ppi_path=args.ppi,
        esm_path=args.esm,
        pathway_path=args.pathway,
        ckpt_path=args.ckpt,
        pairs=pairs,
        out_path=out,
        device=args.device,
        threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
