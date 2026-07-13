"""Run-level config + per-sample inputs for the preprocessing pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------- prompt helpers ----------


def ask_path(prompt: str, must_exist: bool = True) -> Path:
    raw = input(prompt).strip().strip('"').strip("'")
    p = Path(raw).expanduser().resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def ask_optional_path(prompt: str) -> Optional[Path]:
    raw = input(prompt).strip().strip('"').strip("'")
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else default


def ask_bool(prompt: str, default: bool) -> bool:
    raw = input(f"{prompt} [{default}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"yes", "y", "1", "true"}


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    raw = input(f"{prompt} {choices} [{default}]: ").strip().lower()
    if not raw:
        return default
    if raw not in choices:
        raise ValueError(f"Invalid choice: {raw}. Pick one of {choices}.")
    return raw


def ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    return float(raw) if raw else default


def ask_str(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or default


def ask_multi_choice(prompt: str, choices: list[str], default: list[str]) -> list[str]:
    """Comma-separated multi-pick. Empty input = default."""
    raw = (
        input(f"{prompt} {choices} (comma-separated) [{','.join(default)}]: ")
        .strip()
        .lower()
    )
    if not raw:
        return list(default)
    picks = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in picks if x not in choices]
    if bad:
        raise ValueError(f"Invalid choices: {bad}. Pick from {choices}.")
    return picks


# ---------- data classes ----------


@dataclass
class Sample:
    name: str
    r1: Path
    r2: Optional[Path] = None
    condition: Optional[str] = None  # e.g., "control" / "treated" — required for DEG


@dataclass
class PreprocessingConfig:
    samples: list[Sample]
    gtf: Path
    genome_dir: Path
    out_dir: Path
    threads: int = 4
    skip_trim: bool = True
    trimmomatic_jar: Optional[Path] = None
    aligner: str = "star"
    strand: str = "unstranded"
    normalizations: list[str] = field(default_factory=lambda: ["tpm", "fpkm", "rpkm"])

    # ---- DEG (PyDESeq2) ----
    enable_deg: bool = False
    reference_condition: Optional[str] = (
        None  # which condition is the baseline for log2FC
    )
    treated_condition: Optional[str] = (
        None  # which condition is the foreground (None = pick the other one)
    )
    padj_threshold: float = 0.05
    lfc_threshold: float = 1.0  # |log2FC| cut for "significant"
    build_pairs_for_enrichment: bool = (
        True  # if True, also write deg_pairs.tsv (ENSP pairs)
    )

    # ---- ENSP pair building (STRING + orphan fallback) ----
    string_confidence_threshold: int = (
        400  # STRING combined score 0–1000 (400 = medium confidence)
    )
    pairs_fallback_max: int = (
        500  # max combinatorial pairs for orphan ENSPs (no STRING edge)
    )

    # ---- RAG pathway interests (optional) ----
    pathway_interests: list[str] = field(default_factory=list)
    # e.g. ["cell cycle", "apoptosis", "DNA repair"]
    # DEGs belonging to these pathways are always included in RAG retrieval
    # regardless of the 60% selection rule.


# ---------- sample discovery ----------

_R1_MARKERS = ("_R1", ".R1")
_R2_MARKERS = ("_R2", ".R2")
_FASTQ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq", ".fq")


def _strip_fastq_suffix(name: str) -> str:
    for s in _FASTQ_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def _sample_name_from_r1(r1: Path) -> str:
    base = _strip_fastq_suffix(r1.name)
    for m in _R1_MARKERS:
        if m in base:
            return base.split(m)[0].rstrip("._-")
    return base


def _r2_from_r1(r1: Path) -> Optional[Path]:
    for m in _R1_MARKERS:
        if m in r1.name:
            r2_name = r1.name.replace(m, m.replace("R1", "R2"), 1)
            r2 = r1.with_name(r2_name)
            if r2.exists():
                return r2
    return None


def discover_samples_in_dir(folder: Path) -> list[Sample]:
    samples: list[Sample] = []
    for f in sorted(folder.iterdir()):
        if not any(f.name.endswith(s) for s in _FASTQ_SUFFIXES):
            continue
        if any(m in f.name for m in _R2_MARKERS):
            # R2 is picked up via its R1 partner; skip standalone R2.
            continue
        if any(m in f.name for m in _R1_MARKERS):
            samples.append(
                Sample(
                    name=_sample_name_from_r1(f),
                    r1=f.resolve(),
                    r2=_r2_from_r1(f),
                )
            )
        else:
            # No R1/R2 marker => treat as single-end.
            samples.append(Sample(name=_strip_fastq_suffix(f.name), r1=f.resolve()))
    if not samples:
        raise FileNotFoundError(f"No FASTQ files found in {folder}")
    return samples


def load_samplesheet(path: Path) -> list[Sample]:
    """CSV/TSV with header: sample,r1,r2,condition (r2 + condition optional)."""
    delim = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    samples: list[Sample] = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        for row in reader:
            r1 = Path(row["r1"]).expanduser().resolve()
            r2_raw = (row.get("r2") or "").strip()
            r2 = Path(r2_raw).expanduser().resolve() if r2_raw else None
            cond_raw = (row.get("condition") or "").strip() or None
            samples.append(
                Sample(name=row["sample"].strip(), r1=r1, r2=r2, condition=cond_raw)
            )
    if not samples:
        raise ValueError(f"Samplesheet is empty: {path}")
    return samples


def assign_conditions_interactive(samples: list[Sample]) -> list[Sample]:
    """For samples whose `condition` is None, prompt the user to assign one."""
    print("Assign a condition label to each sample (e.g. 'control', 'treated').")
    print("Use the SAME label for biological replicates of the same condition.")
    for s in samples:
        if s.condition:
            print(f"  - {s.name}: condition '{s.condition}' (from samplesheet, kept)")
            continue
        cond = input(f"  - {s.name}: condition: ").strip()
        if not cond:
            raise ValueError(
                f"Sample '{s.name}' has no condition. "
                "DEG analysis needs a condition for every sample."
            )
        s.condition = cond
    return samples


def collect_samples_from_user() -> list[Sample]:
    mode = ask_choice("Sample input", ["folder", "samplesheet"], default="folder")
    if mode == "folder":
        folder = ask_path("Folder containing FASTQs: ")
        samples = discover_samples_in_dir(folder)
    else:
        sheet = ask_path("Path to samplesheet (CSV or TSV with header sample,r1,r2): ")
        samples = load_samplesheet(sheet)
    print(f"Discovered {len(samples)} sample(s):")
    for s in samples:
        r2 = f", R2={s.r2}" if s.r2 else ""
        print(f"  - {s.name}: R1={s.r1}{r2}")
    return samples


# ---------- top-level config collector ----------


def collect_config_from_user() -> PreprocessingConfig:
    print("=== Preprocessing config ===")
    samples = collect_samples_from_user()
    gtf = ask_path("Path to GTF: ")
    genome_dir = ask_path("Path to STAR genomeDir (index): ")
    out_dir_raw = (
        input("Output project directory (will be created): ")
        .strip()
        .strip('"')
        .strip("'")
    )
    out_dir = Path(out_dir_raw).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = ask_int("Threads", default=4)
    skip_trim = ask_bool("Skip trimming for this run?", default=True)
    trimmomatic_jar = None
    if not skip_trim:
        trimmomatic_jar = ask_path("Path to Trimmomatic .jar: ")
    aligner = ask_choice("Aligner", ["star"], default="star")
    strand = ask_choice(
        "Strandedness", ["unstranded", "forward", "reverse"], default="unstranded"
    )
    normalizations = ask_multi_choice(
        "Normalizations to compute",
        ["tpm", "fpkm", "rpkm"],
        default=["tpm", "fpkm", "rpkm"],
    )

    # ---- DEG (PyDESeq2) — optional ----
    enable_deg = ask_bool(
        "Run differential expression analysis (PyDESeq2)? "
        "Needs >=2 conditions with >=2 replicates each",
        default=False,
    )
    reference_condition: Optional[str] = None
    treated_condition: Optional[str] = None
    padj_threshold = 0.05
    lfc_threshold = 1.0
    build_pairs = True
    if enable_deg:
        samples = assign_conditions_interactive(samples)
        condition_set = sorted({s.condition for s in samples if s.condition})
        if len(condition_set) < 2:
            raise ValueError(
                f"Need at least 2 distinct conditions for DEG; got {condition_set}."
            )
        reference_condition = ask_choice(
            "Reference condition (the baseline; log2FC is treated/reference)",
            condition_set,
            default=condition_set[0],
        )
        # Default the treated to the only remaining condition if 2-way comparison;
        # otherwise let the user pick which non-reference condition to compare against.
        non_ref = [c for c in condition_set if c != reference_condition]
        treated_condition = (
            non_ref[0]
            if len(non_ref) == 1
            else ask_choice(
                "Treated condition (foreground)", non_ref, default=non_ref[0]
            )
        )
        padj_threshold = ask_float("padj threshold for 'significant'", 0.05)
        lfc_threshold = ask_float("|log2FC| threshold for 'significant'", 1.0)
        build_pairs = ask_bool(
            "Also build candidate ENSP-pair TSV for the enrichment subgraph?",
            default=True,
        )

    # Optional: pathway interests for RAG-grounded summary
    raw_interests = input(
        "\nAny biological pathway categories of interest for RAG summary?\n"
        "  (comma-separated, e.g. 'cell cycle, apoptosis, DNA repair')\n"
        "  Press Enter to skip: "
    ).strip()
    pathway_interests = (
        [p.strip() for p in raw_interests.split(",") if p.strip()]
        if raw_interests
        else []
    )

    return PreprocessingConfig(
        samples=samples,
        gtf=gtf,
        genome_dir=genome_dir,
        out_dir=out_dir,
        threads=threads,
        skip_trim=skip_trim,
        trimmomatic_jar=trimmomatic_jar,
        aligner=aligner,
        strand=strand,
        normalizations=normalizations,
        enable_deg=enable_deg,
        reference_condition=reference_condition,
        treated_condition=treated_condition,
        padj_threshold=padj_threshold,
        lfc_threshold=lfc_threshold,
        build_pairs_for_enrichment=build_pairs,
        pathway_interests=pathway_interests,
    )
