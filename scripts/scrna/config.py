"""Configuration + interactive prompts for the scRNA-seq pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------- prompt helpers ----------


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or default


def _ask_path(prompt: str, must_exist: bool = True) -> Path:
    raw = _ask(prompt).strip('"').strip("'")
    p = Path(raw).expanduser().resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def _ask_int(prompt: str, default: int) -> int:
    raw = _ask(prompt, str(default))
    return int(raw)


def _ask_float(prompt: str, default: float) -> float:
    raw = _ask(prompt, str(default))
    return float(raw)


def _ask_bool(prompt: str, default: bool) -> bool:
    raw = _ask(prompt, "Y/n" if default else "y/N").lower()
    if raw in {"y/n", "y/n", "y", "yes", "1", "true"}:
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
class ScrnaConfig:
    """Configuration for the scRNA-seq pipeline."""

    # ---- input ----
    input_kind: str  # "h5ad" | "10x" | "pbmc3k"
    out_dir: Path
    input_path: Optional[Path] = None  # None when input_kind == "pbmc3k"

    # ---- filter thresholds ----
    min_cells_per_gene: int = 3
    min_genes_per_cell: int = 200
    max_pct_mt: float = 5.0

    # ---- normalization ----
    target_sum: float = 1e4
    n_top_hvg: int = 2000

    # ---- pca ----
    n_pcs: int = 50
    scale_clip: bool = True  # clip scaled values during sc.pp.scale
    scale_max_value: float = 10.0  # clip threshold (only used if scale_clip)

    # ---- clustering ----
    n_neighbors: int = 15
    leiden_resolution: float = 0.5

    # ---- markers ----
    n_top_markers: int = 25

    # ---- trajectory + pseudotime ----
    run_trajectory: bool = False  # PAGA graph + PAGA-init UMAP
    n_waypoints: int = 500  # Palantir waypoints; more = smoother but slower

    # ---- benchmarking ----
    run_benchmark: bool = False  # ARI/NMI vs ground truth + per-node timing
    true_labels_col: Optional[str] = None  # obs column with true labels;
    # None = auto-load pbmc3k reference


# ---------- top-level collector ----------


def collect_config_from_user() -> ScrnaConfig:
    print("=== scRNA-seq config ===")

    input_kind = _ask_choice(
        "Input kind",
        ["h5ad", "10x", "pbmc3k"],
        default="pbmc3k",
    )

    input_path: Optional[Path] = None
    if input_kind == "h5ad":
        input_path = _ask_path("Path to .h5ad file")
    elif input_kind == "10x":
        input_path = _ask_path(
            "Path to 10x folder (containing matrix.mtx + barcodes.tsv + features.tsv)"
        )

    out_dir_raw = _ask("Output directory (will be created)")
    out_dir = Path(out_dir_raw).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_benchmark = _ask_bool(
        "Run benchmarking (ARI/NMI vs ground truth labels + per-node timing)?",
        default=False,
    )
    true_labels_col: Optional[str] = None
    if run_benchmark and input_kind != "pbmc3k":
        col = _ask(
            "Column in adata.obs with true labels " "(press Enter to skip ARI/NMI)",
            default="",
        )
        true_labels_col = col or None

    run_trajectory = _ask_bool(
        "Run PAGA trajectory inference after clustering?", default=False
    )

    use_defaults = _ask_bool(
        "Use default thresholds for filter/normalize/pca/cluster?", default=True
    )
    if use_defaults:
        return ScrnaConfig(
            input_kind=input_kind,
            input_path=input_path,
            out_dir=out_dir,
            run_trajectory=run_trajectory,
            run_benchmark=run_benchmark,
            true_labels_col=true_labels_col,
        )

    scale_clip = _ask_bool("Clip scaled gene values? (recommended)", default=True)
    scale_max_value = (
        _ask_float("Clip max_value (z-score)", default=10.0) if scale_clip else 10.0
    )

    return ScrnaConfig(
        input_kind=input_kind,
        input_path=input_path,
        out_dir=out_dir,
        run_trajectory=run_trajectory,
        run_benchmark=run_benchmark,
        true_labels_col=true_labels_col,
        min_cells_per_gene=_ask_int("min_cells_per_gene", 3),
        min_genes_per_cell=_ask_int("min_genes_per_cell", 200),
        max_pct_mt=_ask_float("max_pct_mt", 5.0),
        target_sum=_ask_float("normalize_total target_sum", 1e4),
        n_top_hvg=_ask_int("n_top_hvg", 2000),
        n_pcs=_ask_int("n_pcs", 50),
        scale_clip=scale_clip,
        scale_max_value=scale_max_value,
        n_neighbors=_ask_int("n_neighbors", 15),
        leiden_resolution=_ask_float("leiden_resolution", 0.5),
        n_top_markers=_ask_int("n_top_markers per cluster", 25),
    )
