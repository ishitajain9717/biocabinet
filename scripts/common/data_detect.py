"""Automatic detection of the input data modality.

Given a file or directory, inspect its contents and decide whether it looks
like bulk RNA-seq, single-cell RNA-seq, or spatial transcriptomics data.

The detector is heuristic and file-structure based — no heavy dependencies
required.  For ambiguous ``.h5ad`` files (which can be either scRNA or
spatial) it optionally peeks inside the HDF5 file for an ``obsm/spatial``
array, which is the canonical marker that an AnnData object carries spatial
coordinates.

Public API
----------
    from scripts.common.data_detect import detect_data_type

    result = detect_data_type("/path/to/data")
    result.data_type     # "bulk_rnaseq" | "scrna" | "spatial" | "unknown"
    result.confidence    # "high" | "medium" | "low"
    result.platform      # e.g. "visium", "xenium", "merscope", or None
    result.evidence      # list[str] human-readable reasons
    result.scores        # {data_type: score} raw signal counts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# file-pattern signal tables
# ---------------------------------------------------------------------------

_FASTQ_EXTS = (".fastq", ".fastq.gz", ".fq", ".fq.gz")
_PAIRED_TOKENS = ("_r1", "_r2", "_1.", "_2.", ".r1.", ".r2.")

# 10x / scRNA markers (filenames or dir names, matched case-insensitively)
_TENX_FILES = ("matrix.mtx", "barcodes.tsv", "features.tsv", "genes.tsv")
_TENX_DIRS = ("filtered_feature_bc_matrix", "raw_feature_bc_matrix")
_SCRNA_NAME_HINTS = ("pbmc3k", "pbmc", "scrna", "single_cell", "singlecell")

# Spatial markers, grouped by platform → list of signal files/dirs
_SPATIAL_PLATFORM_SIGNALS: dict[str, tuple[str, ...]] = {
    "visium": (
        "scalefactors_json.json",
        "tissue_positions.csv",
        "tissue_positions_list.csv",
        "tissue_hires_image.png",
        "tissue_lowres_image.png",
    ),
    "xenium": (
        "transcripts.parquet",
        "cell_feature_matrix.h5",
        "experiment.xenium",
        "cell_boundaries.parquet",
    ),
    "merscope": (
        "cell_by_gene.csv",
        "cell_metadata.csv",
        "detected_transcripts.csv",
    ),
    "stereo_seq": (),  # handled by extension below (.gem / .gef)
}
_SPATIAL_EXTS = (".gem", ".gef", ".vzg")
_SPATIAL_NAME_HINTS = (
    "spatial",
    "visium",
    "xenium",
    "merscope",
    "merfish",
    "stereo",
    "slide_seq",
    "slideseq",
    "cosmx",
)


# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    data_type: str  # bulk_rnaseq | scrna | spatial | unknown
    confidence: str  # high | medium | low
    platform: str | None = None
    evidence: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _iter_files(root: Path, max_files: int = 5000) -> list[Path]:
    """Return up to max_files paths under root (recursive). Single file → [file]."""
    if root.is_file():
        return [root]
    out: list[Path] = []
    for p in root.rglob("*"):
        out.append(p)
        if len(out) >= max_files:
            break
    return out


def _h5ad_has_spatial(path: Path) -> bool | None:
    """Peek into a .h5ad (HDF5) file for an obsm/spatial array.

    Returns True/False if we could inspect, or None if h5py is unavailable
    or the file could not be read.
    """
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            obsm = f.get("obsm")
            if obsm is None:
                return False
            # AnnData stores coordinates under obsm/spatial
            return "spatial" in obsm.keys()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def detect_data_type(path: str | Path) -> DetectionResult:
    """Inspect a file/directory and guess the data modality."""
    root = Path(path).expanduser()
    if not root.exists():
        return DetectionResult(
            data_type="unknown",
            confidence="low",
            evidence=[f"path does not exist: {root}"],
        )

    files = _iter_files(root)
    names_lower = [f.name.lower() for f in files]
    full_lower = str(root).lower()

    scores = {"bulk_rnaseq": 0, "scrna": 0, "spatial": 0}
    evidence: list[str] = []
    platform: str | None = None

    # ---- bulk: FASTQ files ----
    fastqs = [n for n in names_lower if n.endswith(_FASTQ_EXTS)]
    if fastqs:
        scores["bulk_rnaseq"] += 3 + min(len(fastqs), 5)
        evidence.append(f"found {len(fastqs)} FASTQ file(s)")
        if any(tok in n for n in fastqs for tok in _PAIRED_TOKENS):
            scores["bulk_rnaseq"] += 2
            evidence.append("paired-end naming (_R1/_R2) detected")

    # ---- spatial platform signal files ----
    for plat, signals in _SPATIAL_PLATFORM_SIGNALS.items():
        hits = [s for s in signals if s in names_lower]
        if hits:
            scores["spatial"] += 3 + len(hits)
            platform = plat
            evidence.append(f"{plat} signal files: {', '.join(hits)}")

    # ---- spatial by extension (.gem/.gef/.vzg) ----
    spatial_ext_hits = [n for n in names_lower if n.endswith(_SPATIAL_EXTS)]
    if spatial_ext_hits:
        scores["spatial"] += 3
        if platform is None:
            platform = (
                "stereo_seq"
                if any(n.endswith((".gem", ".gef")) for n in spatial_ext_hits)
                else "merscope"
            )
        evidence.append(f"spatial extension files: {len(spatial_ext_hits)}")

    # ---- a 'spatial' subfolder is the Visium hallmark ----
    if any(f.is_dir() and f.name.lower() == "spatial" for f in files):
        scores["spatial"] += 3
        platform = platform or "visium"
        evidence.append("'spatial/' directory present (Visium)")

    # ---- scrna: 10x matrix files ----
    tenx_hits = [s for s in _TENX_FILES if any(s in n for n in names_lower)]
    if tenx_hits:
        scores["scrna"] += 2 + len(tenx_hits)
        evidence.append(f"10x matrix files: {', '.join(tenx_hits)}")
    if any(d in full_lower or d in names_lower for d in _TENX_DIRS):
        scores["scrna"] += 3
        evidence.append("10x feature-barcode matrix directory")
    if any(n.endswith(".loom") for n in names_lower):
        scores["scrna"] += 3
        evidence.append("loom file (single-cell)")

    # ---- name-based hints ----
    if any(h in full_lower for h in _SCRNA_NAME_HINTS):
        scores["scrna"] += 1
        evidence.append("path name hints at single-cell")
    if any(h in full_lower for h in _SPATIAL_NAME_HINTS):
        scores["spatial"] += 1
        evidence.append("path name hints at spatial")

    # ---- .h5ad: ambiguous scRNA vs spatial → peek inside ----
    h5ads = [f for f in files if f.name.lower().endswith(".h5ad")]
    if h5ads:
        has_spatial = _h5ad_has_spatial(h5ads[0])
        if has_spatial is True:
            scores["spatial"] += 4
            evidence.append(".h5ad contains obsm/spatial coordinates")
        elif has_spatial is False:
            scores["scrna"] += 3
            evidence.append(".h5ad with no spatial coordinates (single-cell)")
        else:
            # Couldn't inspect — weak scRNA prior, but note ambiguity
            scores["scrna"] += 1
            evidence.append(
                ".h5ad found (install h5py to auto-distinguish scRNA/spatial)"
            )

    # ---- decide ----
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]
    sorted_scores = sorted(scores.values(), reverse=True)
    runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0

    if best_score == 0:
        return DetectionResult(
            data_type="unknown",
            confidence="low",
            platform=None,
            evidence=evidence or ["no recognisable data signals found"],
            scores=scores,
        )

    margin = best_score - runner_up
    if best_score >= 5 and margin >= 3:
        confidence = "high"
    elif best_score >= 3 and margin >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    return DetectionResult(
        data_type=best,
        confidence=confidence,
        platform=platform if best == "spatial" else None,
        evidence=evidence,
        scores=scores,
    )
