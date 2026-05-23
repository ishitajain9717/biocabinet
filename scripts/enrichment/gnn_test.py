"""Evaluate a trained GIN_PPI checkpoint on the validation split.

Splits validation edges into 3 buckets to expose how the model generalizes:

  * test1 (both seen)    : both endpoints appeared in training edges
  * test2 (one seen)     : exactly one endpoint appeared in training
  * test3 (both unseen)  : neither endpoint appeared in training (hardest)

This is the classic GNN-PPI evaluation pattern.

Usage:
    python -m scripts.enrichment.gnn_test \\
        --ppi data/ppi_SHS27k.tsv \\
        --esm data/esm2_embeddings_SHS27k.pt \\
        --pathway data/pathway/06_pathway_embeddings.pt \\
        --ckpt runs/esm_smoke/gnn_model_valid_best.ckpt \\
        --split random --test-size 0.2 --seed 1

Output:
    runs/.../test_metrics.json   per-bucket P/R/F1 + counts
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from scripts.enrichment.gnn_data import GNNDataset
from scripts.enrichment.gnn_model import GIN_PPI
from scripts.enrichment.gnn_train import _resolve_device, _set_seeds, _iter_batches
from scripts.enrichment.utils import Metrictor_PPI


# ---------------------------------------------------------------------------
# bucket classification
# ---------------------------------------------------------------------------

def classify_val_edges(
    train_ids: list[int],
    val_ids:   list[int],
    edge_list: list[tuple[int, int]],
) -> dict[str, list[int]]:
    """Split val edges by node visibility.

    A node is 'seen' if it appears in any training edge.
    For each val edge, count how many endpoints are seen → 0/1/2.
    """
    seen_nodes: set[int] = set()
    for k in train_ids:
        i, j = edge_list[k]
        seen_nodes.add(i)
        seen_nodes.add(j)

    test1, test2, test3 = [], [], []
    for k in val_ids:
        i, j = edge_list[k]
        n_seen = (i in seen_nodes) + (j in seen_nodes)
        if   n_seen == 2: test1.append(k)
        elif n_seen == 1: test2.append(k)
        else:             test3.append(k)

    return {
        "all":   list(val_ids),
        "test1": test1,
        "test2": test2,
        "test3": test3,
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@dataclass
class BucketResult:
    name:      str
    n_edges:   int
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    loss:      float = 0.0


@torch.no_grad()
def evaluate_bucket(
    model:      GIN_PPI,
    data,
    edge_ids:   list[int],
    batch_size: int,
    loss_fn:    nn.Module,
    device:     torch.device,
    name:       str,
) -> BucketResult:
    """Run validation pass over one bucket. Concat all preds before scoring."""
    res = BucketResult(name=name, n_edges=len(edge_ids))
    if not edge_ids:
        return res

    model.eval()
    all_preds:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    loss_sum = 0.0
    n_batches = 0

    for batch_ids in _iter_batches(edge_ids, batch_size, shuffle=False):
        batch_ids = batch_ids.to(device)
        logits = model(data.x, data.pathway_x, data.edge_index, batch_ids)
        labels = data.edge_attr[batch_ids].float()
        loss = loss_fn(logits, labels)

        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        loss_sum += loss.item()
        n_batches += 1

    preds  = torch.cat(all_preds,  dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = Metrictor_PPI(preds, labels)
    res.precision = metrics.Precision
    res.recall    = metrics.Recall
    res.f1        = metrics.F1
    res.loss      = loss_sum / max(n_batches, 1)
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    ppi_path:     Path,
    esm_path:     Path,
    pathway_path: Optional[Path],
    ckpt_path:    Path,
    out_path:     Optional[Path] = None,
    batch_size:   int  = 512,
    device:       str  = "auto",
    split_method: str  = "random",
    test_size:    float = 0.2,
    seed:         int  = 1,
    model_kwargs: Optional[dict] = None,
    split_file:   Optional[Path] = None,
) -> dict:
    """Load checkpoint + dataset, split (or load split), evaluate each bucket."""
    _set_seeds(seed)
    dev = _resolve_device(device)
    print(f"[test] device: {dev}")

    # ---- dataset ----
    dataset = GNNDataset(
        ppi_path=ppi_path,
        esm_emb_path=esm_path,
        pathway_emb_path=pathway_path,
        verbose=True,
    )
    data = dataset.to_pyg_data().to(dev)

    # ---- split ----
    if split_file is not None and split_file.exists():
        sp = json.loads(split_file.read_text())
        train_ids = sp["train_index"]
        val_ids   = sp["valid_index"]
        print(f"[test] split loaded from file: train={len(train_ids)}, val={len(val_ids)}")
    else:
        sp = dataset.split(method=split_method, test_size=test_size, seed=seed)
        train_ids = sp["train_index"]
        val_ids   = sp["valid_index"]

    buckets = classify_val_edges(train_ids, val_ids, dataset.edge_list)
    print(f"[test] buckets: all={len(buckets['all'])}, "
          f"test1={len(buckets['test1'])} (both seen), "
          f"test2={len(buckets['test2'])} (one seen), "
          f"test3={len(buckets['test3'])} (both unseen)")

    # ---- model ----
    mk = model_kwargs or {}
    if dataset.pathway_x is not None:
        mk.setdefault("pathway_dim", dataset.pathway_x.shape[1])
    mk.setdefault("esm_dim", dataset.esm_dim)

    model = GIN_PPI(**mk).to(dev)

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"[test] loaded checkpoint: {ckpt_path} "
          f"(epoch={ckpt.get('epoch', '?')}, val_f1={ckpt.get('val_f1', '?')})")

    # ---- evaluate each bucket ----
    loss_fn = nn.BCEWithLogitsLoss()
    results: dict[str, BucketResult] = {}
    for name in ("all", "test1", "test2", "test3"):
        res = evaluate_bucket(model, data, buckets[name], batch_size, loss_fn, dev, name)
        print(f"  [{name:5s}] n={res.n_edges:>6d}  loss={res.loss:.4f}  "
              f"p={res.precision:.4f}  r={res.recall:.4f}  f1={res.f1:.4f}")
        results[name] = res

    summary = {
        "ckpt_path":    str(ckpt_path),
        "ppi_path":     str(ppi_path),
        "esm_path":     str(esm_path),
        "pathway_path": str(pathway_path) if pathway_path else None,
        "n_train":      len(train_ids),
        "n_val":        len(val_ids),
        "buckets":      {k: asdict(v) for k, v in results.items()},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[test] wrote → {out_path}")

    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained GIN_PPI checkpoint.")
    p.add_argument("--ppi",     required=True, type=Path)
    p.add_argument("--esm",     required=True, type=Path)
    p.add_argument("--pathway", type=Path,     default=None)
    p.add_argument("--ckpt",    required=True, type=Path,
                   help="path to gnn_model_valid_best.ckpt or similar")
    p.add_argument("--output",  type=Path, default=None,
                   help="output JSON file (default: <ckpt-dir>/test_metrics.json)")
    p.add_argument("--split-file", type=Path, default=None,
                   help="optional split JSON (must contain train_index, valid_index)")
    p.add_argument("--split",      default="random", choices=["random", "bfs", "dfs"])
    p.add_argument("--test-size",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=1)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--device",     default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    out = args.output or (args.ckpt.parent / "test_metrics.json")
    evaluate_checkpoint(
        ppi_path=args.ppi,
        esm_path=args.esm,
        pathway_path=args.pathway,
        ckpt_path=args.ckpt,
        out_path=out,
        batch_size=args.batch_size,
        device=args.device,
        split_method=args.split,
        test_size=args.test_size,
        seed=args.seed,
        split_file=args.split_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
