"""Configuration + interactive prompts for the spatial (imaging) pipeline.

v1 scope: MERSCOPE/Vizgen, starting from post-segmentation cell tables.
Segmentation is the user's responsibility — we ingest the two CSVs that
MERSCOPE emits per region:

    <region>_cell_by_gene.csv   counts matrix (genes + Blank-* columns)
    <region>_cell_metadata.csv  per-cell centroid, volume, fov

squidpy.read.vizgen() reads both, pulls the Blank-* columns into
obsm['blank_genes'], and puts centroids into obsm['spatial'].
"""

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


def _ask_bool(prompt: str, default: bool) -> bool:
    raw = _ask(prompt, "Y/n" if default else "y/N").lower()
    if raw in {"y/n", "y", "yes", "1", "true"}:
        return True
    if raw in {"n/y", "n", "no", "0", "false"}:
        return False
    return default


# ---------- filename auto-detection ----------


def _autodetect(dir_path: Path, suffix: str) -> Optional[str]:
    """Return the basename of the first file in dir_path ending with suffix.

    MERSCOPE prefixes the region name, e.g. region_0_cell_by_gene.csv, so we
    match on the suffix rather than an exact filename.
    """
    hits = sorted(dir_path.glob(f"*{suffix}"))
    return hits[0].name if hits else None


# ---------- config dataclass ----------


@dataclass
class SpatialConfig:
    """Configuration for the imaging-based spatial pipeline (Vizgen v1)."""

    # ---- input ----
    input_dir: Path  # directory holding the Vizgen CSVs
    counts_file: str  # *_cell_by_gene.csv (basename)
    meta_file: str  # *_cell_metadata.csv (basename)
    out_dir: Path
    transformation_file: Optional[str] = None  # micron→pixel matrix (optional)
    library_id: str = "library"
    platform: str = "vizgen"  # fixed for v1; future: xenium, cosmx

    # ---- QC thresholds (sensible MERSCOPE defaults; panels are small) ----
    min_counts_per_cell: int = 20  # drop near-empty segmentations
    min_genes_per_cell: int = 5  # drop cells expressing too few genes
    volume_pct_low: float = 1.0  # drop bottom-1% cell volumes (debris)
    volume_pct_high: float = 99.0  # drop top-1% cell volumes (doublets)
    max_blank_fdr: float = 0.05  # dataset warning if misID rate > 5%
    fov_min_median_ratio: float = 0.5  # flag FOVs below 0.5x global median


# ---------- top-level collector ----------


def collect_config_from_user() -> SpatialConfig:
    print("=== Spatial (imaging / Vizgen) config ===")

    input_dir = _ask_path("Path to the Vizgen output directory")

    # Auto-detect the two required CSVs; let the user override.
    counts_guess = _autodetect(input_dir, "cell_by_gene.csv")
    meta_guess = _autodetect(input_dir, "cell_metadata.csv")

    if counts_guess:
        print(f"  detected counts file   : {counts_guess}")
    if meta_guess:
        print(f"  detected metadata file : {meta_guess}")

    counts_file = _ask("counts CSV (cell_by_gene)", counts_guess or "")
    meta_file = _ask("metadata CSV (cell_metadata)", meta_guess or "")

    if not counts_file or not meta_file:
        raise ValueError(
            "Both a cell_by_gene.csv and a cell_metadata.csv are required."
        )

    out_dir_raw = _ask("Output directory (will be created)")
    out_dir = Path(out_dir_raw).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    return SpatialConfig(
        input_dir=input_dir,
        counts_file=counts_file,
        meta_file=meta_file,
        out_dir=out_dir,
    )
