"""Precompute frozen ESM-2 mean-pooled per-protein embeddings.

Run once per protein-set. Output is a small dict {ENSP_id: tensor(esm_dim)} on disk
that the GNN training loop can load directly — no ESM model needed at training time.

Usage:
    python -m scripts.enrichment.precompute_esm \
        --seq-tsv data/protein_seq_SHS27k.tsv \
        --output  data/esm2_embeddings.pt \
        --model   esm2_t6_8M_UR50D \
        --batch-size 4 \
        --device  auto

Output format:
    {
      "ENSP00000xxxxxx": tensor(320,),   # mean-pooled, float32
      ...
    }

Behavior:
  * sequences truncated to model's context limit (warns once per truncation)
  * resumable: if --output exists, missing IDs are appended (no recompute)
  * deterministic on the same input set
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import torch


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def _load_protein_sequences(path: Path) -> dict[str, str]:
    """Parse a tsv with columns: ENSP_id<TAB>sequence (header optional)."""
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pid, seq = parts[0].strip(), parts[1].strip()
            if pid.lower() in {"protein", "id", "ensp"}:    # skip header row
                continue
            if pid and seq:
                out[pid] = seq
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

_MODEL_TO_LAYER_AND_DIM = {
    "esm2_t6_8M_UR50D":    (6,   320),
    "esm2_t12_35M_UR50D":  (12,  480),
    "esm2_t30_150M_UR50D": (30,  640),
    "esm2_t33_650M_UR50D": (33, 1280),
    "esm2_t36_3B_UR50D":   (36, 2560),
    "esm2_t48_15B_UR50D":  (48, 5120),
}


def _resolve_device(spec: str) -> torch.device:
    spec = spec.lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _load_esm_model(name: str, device: torch.device):
    """Load ESM-2 model, alphabet, and batch_converter."""
    import esm
    if name not in _MODEL_TO_LAYER_AND_DIM:
        raise ValueError(f"unknown ESM-2 model: {name!r}. "
                         f"Choices: {sorted(_MODEL_TO_LAYER_AND_DIM)}")
    print(f"[esm] loading {name} ...")
    fn = getattr(esm.pretrained, name)
    model, alphabet = fn()
    model.eval()
    model = model.to(device)
    return model, alphabet, alphabet.get_batch_converter()


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

def _iter_batches(items: list[tuple[str, str]], n: int) -> Iterable[list[tuple[str, str]]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def precompute_embeddings(
    sequences:  dict[str, str],
    model_name: str,
    device:     torch.device,
    batch_size: int  = 4,
    max_len:    int  = 1022,
    existing:   dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute frozen ESM-2 mean-pooled embeddings for each sequence.

    Returns a dict {ENSP_id: float32 tensor of shape (esm_dim,)}.
    If `existing` is provided, only sequences not in `existing` are computed,
    and the result is merged.
    """
    layer_idx, esm_dim = _MODEL_TO_LAYER_AND_DIM[model_name]
    model, alphabet, batch_converter = _load_esm_model(model_name, device)
    print(f"[esm] model loaded. layer={layer_idx}, dim={esm_dim}, device={device}")

    out: dict[str, torch.Tensor] = dict(existing) if existing else {}
    todo = [(pid, s) for pid, s in sequences.items() if pid not in out]
    print(f"[esm] todo: {len(todo)} of {len(sequences)} proteins "
          f"(skipping {len(sequences) - len(todo)} already present)")

    if not todo:
        return out

    n_truncated = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(_iter_batches(todo, batch_size)):
        # truncate any oversize sequences
        clean_batch: list[tuple[str, str]] = []
        for pid, seq in batch:
            if len(seq) > max_len:
                n_truncated += 1
                seq = seq[:max_len]
            clean_batch.append((pid, seq))

        # tokenize
        labels, _, tokens = batch_converter(clean_batch)
        tokens = tokens.to(device)

        # forward
        with torch.no_grad():
            result = model(tokens, repr_layers=[layer_idx])
        rep = result["representations"][layer_idx]    # (B, L+2, D)

        # mean-pool over real residues only (drop BOS at index 0 and EOS / padding via mask)
        # tokens.shape == (B, L+2). padding_idx is alphabet.padding_idx.
        pad_idx = alphabet.padding_idx
        bos_idx = alphabet.cls_idx     # <cls> behaves as BOS in ESM
        eos_idx = alphabet.eos_idx
        # mask: True where token is a real residue (not BOS/EOS/PAD)
        residue_mask = (tokens != pad_idx) & (tokens != bos_idx) & (tokens != eos_idx)  # (B, L+2)
        # weighted mean over residue dimension
        residue_mask_f = residue_mask.unsqueeze(-1).float()                              # (B, L+2, 1)
        rep_sum  = (rep * residue_mask_f).sum(dim=1)                                     # (B, D)
        rep_cnt  = residue_mask_f.sum(dim=1).clamp(min=1.0)                              # (B, 1)
        rep_mean = (rep_sum / rep_cnt).cpu().float()                                     # (B, D)

        for label, vec in zip(labels, rep_mean):
            out[label] = vec

        # progress every 25 batches
        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == (len(todo) + batch_size - 1) // batch_size:
            done = min((batch_idx + 1) * batch_size, len(todo))
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (len(todo) - done) / max(rate, 1e-6)
            print(f"[esm] {done}/{len(todo)} proteins  "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, {rate:.1f} prot/s]")

    if n_truncated > 0:
        print(f"[esm] WARNING: truncated {n_truncated} sequence(s) to max_len={max_len}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute frozen ESM-2 per-protein embeddings.")
    p.add_argument("--seq-tsv",  required=True, type=Path, help="TSV: ENSP_id<TAB>sequence")
    p.add_argument("--output",   required=True, type=Path, help="output .pt file")
    p.add_argument("--model",    default="esm2_t6_8M_UR50D",
                   choices=sorted(_MODEL_TO_LAYER_AND_DIM.keys()))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len",    type=int, default=1022,
                   help="truncate sequences longer than this (excl. BOS/EOS)")
    p.add_argument("--device",     default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--rebuild",    action="store_true",
                   help="ignore existing output, recompute from scratch")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sequences = _load_protein_sequences(args.seq_tsv)
    print(f"[main] loaded {len(sequences)} sequences from {args.seq_tsv}")

    existing: dict[str, torch.Tensor] | None = None
    if args.output.exists() and not args.rebuild:
        existing = torch.load(args.output, weights_only=False)
        print(f"[main] resuming: {len(existing)} embeddings already in {args.output}")

    device = _resolve_device(args.device)

    embeddings = precompute_embeddings(
        sequences=sequences,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        existing=existing,
    )

    torch.save(embeddings, args.output)
    sample = next(iter(embeddings.values()))
    print(f"[main] wrote {len(embeddings)} embeddings → {args.output}")
    print(f"[main] tensor shape: {tuple(sample.shape)}, dtype: {sample.dtype}")
    print(f"[main] file size: {args.output.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
