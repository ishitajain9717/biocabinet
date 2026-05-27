"""Training loop for GIN_PPI.

Usage as a library:
    from scripts.enrichment.gnn_train import train
    result = train(dataset, out_dir=Path("runs/exp1"), epochs=50)

Usage as a CLI:
    python -m scripts.enrichment.gnn_train \\
        --ppi data/ppi_SHS27k.tsv \\
        --seq data/protein_seq_SHS27k.tsv \\
        --vec data/amino_acid_vec.tsv \\
        --pathway data/pathway/06_pathway_embeddings.pt \\
        --out-dir runs/shs27k_v1 \\
        --epochs 50 \\
        --batch-size 512 \\
        --device auto

Saves under out_dir:
    gnn_model_valid_best.ckpt   model at best val F1
    gnn_model_train_last.ckpt   model at last completed epoch
    training_history.json       per-epoch metric dicts
    tensorboard/                optional TB logs (if tensorboard installed)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from scripts.enrichment.gnn_data import GNNDataset
from scripts.enrichment.gnn_model import GIN_PPI, count_parameters
from scripts.enrichment.utils import Metrictor_PPI, print_file

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> torch.device:
    """Pick a torch.device.

    spec:
        'auto'   → cuda if available, else mps if available, else cpu
        'cuda'   → cuda:0
        'mps'    → Apple Silicon GPU
        'cpu'    → cpu
    """
    spec = spec.lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_batches(
    ids: list[int], batch_size: int, shuffle: bool = True
) -> Iterable[torch.Tensor]:
    """Yield 1-D LongTensors of edge ids, batch_size at a time."""
    ids = list(ids)
    if shuffle:
        random.shuffle(ids)
    n = len(ids)
    n_batches = math.ceil(n / batch_size)
    for i in range(n_batches):
        chunk = ids[i * batch_size : (i + 1) * batch_size]
        yield torch.as_tensor(chunk, dtype=torch.long)


# ---------------------------------------------------------------------------
# core epoch logic
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: GIN_PPI,
    data,
    train_ids: list[int],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """Run one training epoch.

    Returns dict with avg loss + F1/Precision/Recall on the full epoch.
    """
    model.train()
    epoch_preds: list[torch.Tensor] = []
    epoch_labels: list[torch.Tensor] = []
    loss_sum = 0.0
    n_batches = 0

    for batch_ids in _iter_batches(train_ids, batch_size, shuffle=True):
        batch_ids = batch_ids.to(device)
        logits = model(data.x, data.pathway_x, data.edge_index, batch_ids)  # (B, C)
        labels = data.edge_attr[batch_ids].float()  # (B, C)

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
        epoch_preds.append(preds.detach().cpu())
        epoch_labels.append(labels.detach().cpu())

        loss_sum += loss.item()
        n_batches += 1

    all_preds = torch.cat(epoch_preds, dim=0)
    all_labels = torch.cat(epoch_labels, dim=0)
    metrics = Metrictor_PPI(all_preds, all_labels)
    return {
        "loss": loss_sum / max(n_batches, 1),
        "precision": metrics.Precision,
        "recall": metrics.Recall,
        "f1": metrics.F1,
        "n_batches": n_batches,
        "n_samples": int(all_labels.shape[0]),
    }


@torch.no_grad()
def validate(
    model: GIN_PPI,
    data,
    val_ids: list[int],
    batch_size: int,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """Validation pass. Concats all batches before computing metrics (correct way)."""
    model.eval()
    val_preds: list[torch.Tensor] = []
    val_labels: list[torch.Tensor] = []
    loss_sum = 0.0
    n_batches = 0

    for batch_ids in _iter_batches(val_ids, batch_size, shuffle=False):
        batch_ids = batch_ids.to(device)
        logits = model(data.x, data.pathway_x, data.edge_index, batch_ids)
        labels = data.edge_attr[batch_ids].float()
        loss = loss_fn(logits, labels)

        preds = (torch.sigmoid(logits) > 0.5).float()
        val_preds.append(preds.cpu())
        val_labels.append(labels.cpu())

        loss_sum += loss.item()
        n_batches += 1

    all_preds = torch.cat(val_preds, dim=0)
    all_labels = torch.cat(val_labels, dim=0)
    metrics = Metrictor_PPI(all_preds, all_labels)
    return {
        "loss": loss_sum / max(n_batches, 1),
        "precision": metrics.Precision,
        "recall": metrics.Recall,
        "f1": metrics.F1,
        "n_batches": n_batches,
        "n_samples": int(all_labels.shape[0]),
    }


# ---------------------------------------------------------------------------
# top-level training entrypoint
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    best_f1: float
    best_epoch: int
    best_ckpt_path: Path
    last_ckpt_path: Path
    history_path: Path
    history: list[dict] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    n_params: int = 0
    device: str = "cpu"


def train(
    dataset: GNNDataset,
    out_dir: Path,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    split_method: str = "random",  # "random" | "bfs" | "dfs"
    test_size: float = 0.2,
    seed: int = 1,
    use_tensorboard: bool = True,
    early_stop_patience: Optional[int] = 20,
    scheduler_patience: int = 5,
    model_kwargs: Optional[dict] = None,
) -> TrainResult:
    """Full training loop.

    Args:
        dataset: built GNNDataset (already loaded; .split() will be called inside)
        out_dir: where to write checkpoints + history
        epochs, batch_size, lr, weight_decay: standard
        device: 'auto'|'cuda'|'mps'|'cpu'
        split_method: 'random'|'bfs'|'dfs'
        test_size: validation fraction (0..1)
        seed: reproducibility
        use_tensorboard: log to out_dir/tensorboard (silently skipped if TB missing)
        early_stop_patience: stop after N epochs of no val_f1 improvement
            (None = disabled)
        scheduler_patience: ReduceLROnPlateau patience
        model_kwargs: passed straight to GIN_PPI(...)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seeds(seed)

    dev = _resolve_device(device)
    print(f"[train] device: {dev}", flush=True)

    # ---- split + tensors ----
    splits = dataset.split(method=split_method, test_size=test_size, seed=seed)
    train_ids = splits["train_index"]
    val_ids = splits["valid_index"]
    print(
        f"[train] split={split_method}  n_train={len(train_ids)}  n_val={len(val_ids)}",
        flush=True,
    )

    data = dataset.to_pyg_data().to(dev)
    print(
        f"[train] data on device: x={tuple(data.x.shape)}  "
        f"pathway_x={tuple(data.pathway_x.shape)}  "
        f"edge_index={tuple(data.edge_index.shape)}",
        flush=True,
    )

    # ---- model + optim + loss ----
    model_kwargs = model_kwargs or {}
    model = GIN_PPI(**model_kwargs).to(dev)
    n_params = count_parameters(model)
    print(f"[train] model params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=scheduler_patience,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # ---- optional tensorboard ----
    writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        except ImportError:
            print("[train] tensorboard not installed, skipping TB logs")

    best_ckpt_path = out_dir / "gnn_model_valid_best.ckpt"
    last_ckpt_path = out_dir / "gnn_model_train_last.ckpt"
    history_path = out_dir / "training_history.json"

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, data, train_ids, batch_size, optimizer, loss_fn, dev
        )
        val_metrics = validate(model, data, val_ids, batch_size, loss_fn, dev)
        scheduler.step(val_metrics["f1"])
        cur_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": cur_lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        if writer is not None:
            for split, m in [("train", train_metrics), ("val", val_metrics)]:
                writer.add_scalar(f"{split}/loss", m["loss"], epoch)
                writer.add_scalar(f"{split}/precision", m["precision"], epoch)
                writer.add_scalar(f"{split}/recall", m["recall"], epoch)
                writer.add_scalar(f"{split}/f1", m["f1"], epoch)
            writer.add_scalar("lr", cur_lr, epoch)

        print_file(
            f"epoch {epoch:>3}/{epochs}  lr={cur_lr:.2e}  "
            f"train: loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f}  "
            f"val: loss={val_metrics['loss']:.4f} p={val_metrics['precision']:.4f} "
            f"r={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
        )

        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, last_ckpt_path)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_f1": best_f1},
                best_ckpt_path,
            )
            print_file(
                f"           ↳ new best val_f1={best_f1:.4f}"
                f"  (saved {best_ckpt_path.name})"
            )
        else:
            no_improve += 1
            if early_stop_patience is not None and no_improve >= early_stop_patience:
                print_file(
                    "           ↳ early stop: no improvement for"
                    f" {early_stop_patience} epochs"
                )
                break

        with history_path.open("w") as f:
            json.dump(history, f, indent=2)

    if writer is not None:
        writer.close()

    print_file(f"[train] done. best val_f1={best_f1:.4f} at epoch {best_epoch}")

    return TrainResult(
        best_f1=best_f1,
        best_epoch=best_epoch,
        best_ckpt_path=best_ckpt_path,
        last_ckpt_path=last_ckpt_path,
        history_path=history_path,
        history=history,
        n_train=len(train_ids),
        n_val=len(val_ids),
        n_params=n_params,
        device=str(dev),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GIN_PPI")
    p.add_argument("--ppi", required=True, type=Path, help="STRING PPI tsv")
    p.add_argument(
        "--esm", required=True, type=Path, help="precomputed ESM-2 embeddings .pt"
    )
    p.add_argument(
        "--pathway", type=Path, default=None, help="pathway embeddings .pt (optional)"
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--split", default="random", choices=["random", "bfs", "dfs"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--no-tensorboard", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    print("[main] loading dataset from:")
    print(f"         ppi={args.ppi}")
    print(f"         esm={args.esm}")
    print(f"         pathway={args.pathway}")

    dataset = GNNDataset(
        ppi_path=args.ppi,
        esm_emb_path=args.esm,
        pathway_emb_path=args.pathway,
        verbose=True,
    )
    print(
        f"[main] proteins: {len(dataset.protein_to_idx)}, "
        f"unique edges: {len(dataset.edge_attr or [])}, "
        f"components: {dataset.count_components()}"
    )

    result = train(
        dataset=dataset,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        split_method=args.split,
        test_size=args.test_size,
        seed=args.seed,
        use_tensorboard=not args.no_tensorboard,
        early_stop_patience=args.early_stop,
    )
    print(f"[main] best_f1={result.best_f1:.4f}  best_epoch={result.best_epoch}")
    print(f"[main] best ckpt:    {result.best_ckpt_path}")
    print(f"[main] last ckpt:    {result.last_ckpt_path}")
    print(f"[main] history json: {result.history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
