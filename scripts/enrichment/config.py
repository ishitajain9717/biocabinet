"""Configuration + interactive prompts for the enrichment (GNN-PPI) pipeline.

The enrichment subgraph trains a GIN_PPI model on a STRING PPI dataset using
precomputed ESM-2 + (optional) BioBERT pathway embeddings, then evaluates the
checkpoint and runs inference on a candidate-pair list (typically the
combinatorial DEG pairs coming out of the bulk RNA-seq pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------- prompt helpers (mirrors scrna/config.py) ----------


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or default


def _ask_path(prompt: str, must_exist: bool = True, default: str = "") -> Path:
    raw = _ask(prompt, default).strip('"').strip("'")
    p = Path(raw).expanduser().resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def _ask_optional_path(prompt: str, default: str = "") -> Optional[Path]:
    raw = _ask(prompt, default).strip('"').strip("'")
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def _ask_int(prompt: str, default: int) -> int:
    return int(_ask(prompt, str(default)))


def _ask_float(prompt: str, default: float) -> float:
    return float(_ask(prompt, str(default)))


def _ask_bool(prompt: str, default: bool) -> bool:
    raw = _ask(prompt, "Y/n" if default else "y/N").lower()
    if raw in {"y", "yes", "1", "true", "y/n"}:
        return True
    if raw in {"n", "no", "0", "false"}:
        return False
    return default


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    raw = _ask(f"{prompt} {choices}", default).lower()
    if raw not in choices:
        raise ValueError(f"Invalid choice: {raw}. Pick one of {choices}.")
    return raw


# ---------- config dataclass ----------


@dataclass
class EnrichmentConfig:
    """Parameters for the full enrichment subgraph (load → train → eval → infer)."""

    # ---- inputs ----
    ppi_path: Path  # STRING/SHS27k PPI tsv
    esm_path: Path  # precomputed ESM-2 embeddings .pt
    pathway_path: Optional[
        Path
    ]  # combined pathway embeddings .pt (None ⇒ no pathway features)

    # ---- output / resume ----
    out_dir: Path  # checkpoint + metric dump directory

    # ---- training ----
    epochs: int = 50  # was 100; 50 + early-stop=10 is sufficient
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop: int = 10  # was 20; tighter patience speeds up wall time
    split_method: str = "random"  # "random" | "bfs" | "dfs"
    test_size: float = 0.2
    seed: int = 1
    # "cpu" is the right default for SHS27k (1690 proteins, 7624 edges):
    # MPS / CUDA have kernel-launch overhead that hurts on small graphs, and
    # MPS on Apple Silicon shares the GPU with the display server, which causes
    # the whole UI to stall during training.  Set to "mps" or "cuda" explicitly
    # only when you have a large dataset that benefits from GPU parallelism.
    device: str = "cpu"
    use_tensorboard: bool = False

    # ---- evaluation ----
    eval_batch_size: int = 512

    # ---- inference ----
    pairs_path: Optional[Path] = (
        None  # TSV of candidate ENSP pairs (None ⇒ skip inference)
    )
    inference_threshold: float = 0.5

    # ---- skip flags (used by orchestrator when chaining) ----
    skip_train: bool = False
    skip_eval: bool = False
    skip_infer: bool = False


# ---------- top-level collector ----------


def collect_config_from_user() -> EnrichmentConfig:
    print("=== enrichment (GNN-PPI) config ===")

    ppi_path = _ask_path("Path to PPI TSV", default="data/ppi_SHS27k.tsv")
    esm_path = _ask_path(
        "Path to ESM-2 embeddings .pt", default="data/esm2_embeddings_SHS27k.pt"
    )
    pathway_path = _ask_optional_path(
        "Path to pathway embeddings .pt (empty for no pathway features)",
        default="data/pathway/06_pathway_embeddings_combined.pt",
    )

    out_dir_raw = _ask(
        "Output directory (will be created)", default="runs/enrichment_run"
    )
    out_dir = Path(out_dir_raw).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_defaults = _ask_bool("Use default training hyperparameters?", default=True)
    if use_defaults:
        cfg = EnrichmentConfig(
            ppi_path=ppi_path,
            esm_path=esm_path,
            pathway_path=pathway_path,
            out_dir=out_dir,
        )
    else:
        cfg = EnrichmentConfig(
            ppi_path=ppi_path,
            esm_path=esm_path,
            pathway_path=pathway_path,
            out_dir=out_dir,
            epochs=_ask_int("epochs", 50),
            batch_size=_ask_int("train batch_size", 512),
            lr=_ask_float("learning rate", 1e-3),
            weight_decay=_ask_float("weight_decay", 1e-4),
            early_stop=_ask_int("early-stop patience (epochs)", 10),
            split_method=_ask_choice(
                "split method", ["random", "bfs", "dfs"], "random"
            ),
            test_size=_ask_float("val fraction", 0.2),
            seed=_ask_int("seed", 1),
            device=_ask_choice("device", ["cpu", "mps", "cuda", "auto"], "cpu"),
            use_tensorboard=_ask_bool("write TensorBoard logs?", default=False),
            eval_batch_size=_ask_int("eval batch_size", 512),
        )

    pairs_path = _ask_optional_path(
        "Path to candidate-pair TSV for inference (empty to skip inference)",
        default="",
    )
    cfg.pairs_path = pairs_path
    if pairs_path is not None:
        cfg.inference_threshold = _ask_float("inference probability threshold", 0.5)

    return cfg
