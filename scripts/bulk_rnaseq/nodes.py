"""Pipeline nodes: each function runs one tool and returns a NodeResult."""

from __future__ import annotations

import math
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from scripts.bulk_rnaseq.config import PreprocessingConfig, Sample
from scripts.common.node_result import NodeResult

_STRAND_MAP = {"unstranded": "0", "forward": "1", "reverse": "2"}


# ---------- internal helpers ----------


def _run(cmd: list[str]) -> tuple[str, str, int]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout, proc.stderr, proc.returncode


def _tail(text: str, n: int = 200) -> str:
    return text[-n:] if text else ""


# ---------- FastQC report parser ----------


def _parse_fastqc_report(qc_dir: Path, sample_name: str) -> dict:  # type: ignore
    """Extract module statuses and key metrics from a FastQC zip report.

    Returns a dict with:
        module_statuses : {module_name: "pass"|"warn"|"fail"}
        total_sequences : int
        pct_gc          : float
        mean_quality    : float  (median of Per Sequence Quality Scores peak)
        pct_duplicates  : float
        pct_adapter     : float  (max adapter content across bases)
        raw_summary     : str    (first 2000 chars of fastqc_data.txt)
    """
    result: dict = {
        "module_statuses": {},
        "total_sequences": None,
        "pct_gc": None,
        "mean_quality": None,
        "pct_duplicates": None,
        "pct_adapter": None,
        "raw_summary": "",
    }

    # FastQC creates <stem>_fastqc.zip for each input file
    zips = list(qc_dir.glob("*_fastqc.zip"))
    if not zips:
        return result

    # Use the first zip (R1); R2 parsed separately if needed
    zip_path = zips[0]
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # fastqc_data.txt is inside <stem>_fastqc/fastqc_data.txt
            data_files = [n for n in zf.namelist() if n.endswith("fastqc_data.txt")]
            if not data_files:
                return result
            content = zf.read(data_files[0]).decode("utf-8", errors="replace")
    except Exception as e:
        result["raw_summary"] = f"Error reading FastQC zip: {e}"
        return result

    result["raw_summary"] = content[:2000]

    # Parse >>Module Name\tstatus lines
    for line in content.splitlines():
        if line.startswith(">>") and not line.startswith(">>END"):
            parts = line[2:].split("\t")
            if len(parts) == 2:
                module, status = parts
                result["module_statuses"][module] = status.strip()

    # Extract Total Sequences from Basic Statistics table
    m = re.search(r"Total Sequences\s+(\d+)", content)
    if m:
        result["total_sequences"] = int(m.group(1))

    # Extract %GC
    m = re.search(r"%GC\s+(\d+)", content)
    if m:
        result["pct_gc"] = float(m.group(1))

    # Extract % duplication (Sequence Duplication Levels block)
    m = re.search(r"#Total Deduplicated Percentage\s+([\d.]+)", content)
    if m:
        result["pct_duplicates"] = round(100 - float(m.group(1)), 1)

    # Extract mean quality — peak of Per Sequence Quality Scores
    in_quality_block = False
    quality_counts: list[tuple[float, float]] = []
    for line in content.splitlines():
        if line.startswith(">>Per sequence quality scores"):
            in_quality_block = True
            continue
        if in_quality_block and line.startswith(">>END MODULE"):
            break
        if in_quality_block and not line.startswith("#") and line.strip():
            parts = line.split("\t")
            if len(parts) == 2:
                try:
                    quality_counts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    if quality_counts:
        peak_q = max(quality_counts, key=lambda x: x[1])[0]
        result["mean_quality"] = peak_q

    # Extract max adapter content (Adapter Content block)
    in_adapter_block = False
    max_adapter = 0.0
    for line in content.splitlines():
        if line.startswith(">>Adapter Content"):
            in_adapter_block = True
            continue
        if in_adapter_block and line.startswith(">>END MODULE"):
            break
        if in_adapter_block and not line.startswith("#") and line.strip():
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    vals = [float(p) for p in parts[1:]]
                    max_adapter = max(max_adapter, max(vals))
                except ValueError:
                    pass
    result["pct_adapter"] = round(max_adapter, 2)

    return result


# ---------- nodes ----------


def node_fastqc(cfg: PreprocessingConfig, sample: Sample) -> NodeResult:
    out_dir = cfg.out_dir / "01_qc" / sample.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["fastqc", "-o", str(out_dir), str(sample.r1)]
    if sample.r2 is not None:
        cmd.append(str(sample.r2))

    _, err, rc = _run(cmd)
    if rc != 0:
        return NodeResult(
            name="fastqc",
            ok=False,
            message=_tail(err),
            outputs={"qc_dir": str(out_dir)},
            metrics={"returncode": rc},
        )

    report = _parse_fastqc_report(out_dir, sample.name)

    # If FastQC ran but we couldn't parse the zip/quality scores, treat it
    # as a failed node so the QC gate doesn't silently pass a sample with
    # no readable metrics.
    metrics_missing = report["mean_quality"] is None and not report["module_statuses"]
    if metrics_missing:
        return NodeResult(
            name="fastqc",
            ok=False,
            message=(
                "FastQC ran (rc=0) but quality report could not be parsed "
                "— zip may be missing or corrupt"
            ),
            outputs={"qc_dir": str(out_dir)},
            metrics={"returncode": rc},
        )

    return NodeResult(
        name="fastqc",
        ok=True,
        message="ok",
        outputs={"qc_dir": str(out_dir)},
        metrics={
            "returncode": rc,
            "module_statuses": report["module_statuses"],
            "total_sequences": report["total_sequences"],
            "pct_gc": report["pct_gc"],
            "mean_quality": report["mean_quality"],
            "pct_duplicates": report["pct_duplicates"],
            "pct_adapter": report["pct_adapter"],
            "raw_summary": report["raw_summary"],
        },
    )


def node_trim(
    cfg: PreprocessingConfig,
    sample: Sample,
    trim_steps: list[str] | None = None,
) -> NodeResult:
    """Run Trimmomatic (or skip) for a single sample.

    Parameters
    ----------
    trim_steps:
        Trimmomatic step tokens, e.g.
        ["ILLUMINACLIP:adapters.fa:2:30:10", "LEADING:20",
         "TRAILING:20", "SLIDINGWINDOW:4:15", "MINLEN:36"].
        When None the hardcoded defaults are used.
        When empty-list the run is treated as skip.
    """
    # Determine whether to skip:
    #   trim_steps=None  → honour cfg.skip_trim (no LLM gate involved)
    #   trim_steps=[]    → LLM decided no trimming needed
    #   trim_steps=[...] → LLM generated steps; run them if jar is available,
    #                      otherwise warn and fall back to skipping so the
    #                      pipeline can continue rather than hard-failing just
    #                      because the user set skip_trim=True at config time.
    llm_wants_trim = trim_steps is not None and len(trim_steps) > 0

    if llm_wants_trim and cfg.trimmomatic_jar is None:
        print(
            f"  [trim] WARNING: LLM requested trimming for {sample.name} but "
            "trimmomatic_jar is not configured — skipping trim and continuing.",
            flush=True,
        )
        outs: dict[str, str] = {"r1": str(sample.r1)}
        if sample.r2 is not None:
            outs["r2"] = str(sample.r2)
        return NodeResult(
            name="trim",
            ok=True,
            message="skipped (LLM requested trim but trimmomatic_jar not configured)",
            outputs=outs,
            metrics={"trim_steps_requested": trim_steps},
        )

    effective_skip = cfg.skip_trim if trim_steps is None else (len(trim_steps) == 0)

    if effective_skip:
        outs = {"r1": str(sample.r1)}
        if sample.r2 is not None:
            outs["r2"] = str(sample.r2)
        return NodeResult(name="trim", ok=True, message="skipped", outputs=outs)

    if cfg.trimmomatic_jar is None:
        return NodeResult(
            name="trim",
            ok=False,
            message="trimmomatic_jar is required when skip_trim=False",
        )

    # Default steps when none provided (conservative safe defaults)
    steps = (
        trim_steps
        if trim_steps
        else [
            "LEADING:3",
            "TRAILING:3",
            "SLIDINGWINDOW:4:15",
            "MINLEN:36",
        ]
    )

    trim_dir = cfg.out_dir / "02_trim" / sample.name
    trim_dir.mkdir(parents=True, exist_ok=True)

    if sample.r2 is not None:
        r1_out = trim_dir / f"{sample.name}.R1_trimmed.fastq.gz"
        r2_out = trim_dir / f"{sample.name}.R2_trimmed.fastq.gz"
        r1_unp = trim_dir / f"{sample.name}.R1_unpaired.fastq.gz"
        r2_unp = trim_dir / f"{sample.name}.R2_unpaired.fastq.gz"
        cmd = [
            "java",
            "-jar",
            str(cfg.trimmomatic_jar),
            "PE",
            "-threads",
            str(cfg.threads),
            str(sample.r1),
            str(sample.r2),
            str(r1_out),
            str(r1_unp),
            str(r2_out),
            str(r2_unp),
        ] + steps
        _, err, rc = _run(cmd)
        return NodeResult(
            name="trim",
            ok=(rc == 0 and r1_out.exists() and r2_out.exists()),
            message="ok" if rc == 0 else _tail(err),
            outputs={"r1": str(r1_out), "r2": str(r2_out)},
            metrics={"returncode": rc, "trim_steps": steps},
        )

    r1_out = trim_dir / f"{sample.name}.R1_trimmed.fastq.gz"
    cmd = [
        "java",
        "-jar",
        str(cfg.trimmomatic_jar),
        "SE",
        "-threads",
        str(cfg.threads),
        str(sample.r1),
        str(r1_out),
    ] + steps
    _, err, rc = _run(cmd)
    return NodeResult(
        name="trim",
        ok=(rc == 0 and r1_out.exists()),
        message="ok" if rc == 0 else _tail(err),
        outputs={"r1": str(r1_out)},
        metrics={"returncode": rc, "trim_steps": steps},
    )


def _parse_star_log(log_path: Path) -> dict:
    """Parse STAR's Log.final.out into a flat metrics dict.

    STAR writes lines in the form:
        "   <label> |\\t<value>"
    We strip both sides and convert numeric values to int/float where possible.

    Key fields returned (None when not found):
        n_input_reads          int   — total reads fed to STAR
        n_uniquely_mapped      int   — reads aligned exactly once
        pct_uniquely_mapped    float — % uniquely mapped
        pct_multi_mapped       float — % mapped to multiple loci
        pct_unmapped_tooshort  float — % too short to map (common quality signal)
        pct_unmapped_mismatch  float — % unmapped: too many mismatches
        pct_unmapped_other     float — % unmapped: other
        n_splices_total        int
        mismatch_rate_pct      float
        deletion_rate_pct      float
        insertion_rate_pct     float
    """
    metrics: dict = {
        "n_input_reads": None,
        "n_uniquely_mapped": None,
        "pct_uniquely_mapped": None,
        "pct_multi_mapped": None,
        "pct_unmapped_tooshort": None,
        "pct_unmapped_mismatch": None,
        "pct_unmapped_other": None,
        "n_splices_total": None,
        "mismatch_rate_pct": None,
        "deletion_rate_pct": None,
        "insertion_rate_pct": None,
        "star_log_path": str(log_path),
    }

    if not log_path.exists():
        return metrics

    # Map STAR label substrings → our metric keys + whether to strip "%" from value
    _FIELD_MAP = [
        ("Number of input reads", "n_input_reads", False),
        ("Uniquely mapped reads number", "n_uniquely_mapped", False),
        ("Uniquely mapped reads %", "pct_uniquely_mapped", True),
        ("% of reads mapped to multiple loci", "pct_multi_mapped", True),
        ("% of reads unmapped: too short", "pct_unmapped_tooshort", True),
        ("% of reads unmapped: too many mismatches", "pct_unmapped_mismatch", True),
        ("% of reads unmapped: other", "pct_unmapped_other", True),
        ("Number of splices: Total", "n_splices_total", False),
        ("Mismatch rate per base, %", "mismatch_rate_pct", True),
        ("Deletion rate per base", "deletion_rate_pct", True),
        ("Insertion rate per base", "insertion_rate_pct", True),
    ]

    try:
        content = log_path.read_text(errors="replace")
    except OSError:
        return metrics

    for line in content.splitlines():
        if "|" not in line:
            continue
        label_raw, _, value_raw = line.partition("|")
        label = label_raw.strip()
        value = value_raw.strip().rstrip("%")
        for star_label, key, _strip_pct in _FIELD_MAP:
            if star_label in label:
                try:
                    parsed: int | float = (
                        int(value) if "." not in value else float(value)
                    )
                    metrics[key] = parsed
                except ValueError:
                    pass
                break

    # Sanity-check: flag low mapping rate so the RAG / summary can highlight it
    pct = metrics.get("pct_uniquely_mapped")
    metrics["mapping_rate_ok"] = pct is not None and pct >= 70.0

    return metrics


def node_align(
    cfg: PreprocessingConfig, sample: Sample, r1: Path, r2: Optional[Path]
) -> NodeResult:
    if cfg.aligner != "star":
        return NodeResult(
            name="align",
            ok=False,
            message=f"aligner not supported yet: {cfg.aligner}",
        )

    bam_dir = cfg.out_dir / "03_bam" / sample.name
    bam_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(bam_dir / sample.name)

    cmd = [
        "STAR",
        "--runThreadN",
        str(cfg.threads),
        "--genomeDir",
        str(cfg.genome_dir),
        "--readFilesIn",
        str(r1),
    ]
    if r2 is not None:
        cmd.append(str(r2))
    if str(r1).endswith(".gz"):
        cmd += ["--readFilesCommand", "zcat"]
    cmd += [
        "--outSAMtype",
        "BAM",
        "SortedByCoordinate",
        "--outFileNamePrefix",
        str(prefix),
    ]

    _, err, rc = _run(cmd)
    bam_path = bam_dir / f"{sample.name}Aligned.sortedByCoord.out.bam"
    log_path = bam_dir / f"{sample.name}Log.final.out"

    if rc != 0 or not bam_path.exists():
        star_metrics = _parse_star_log(log_path)
        return NodeResult(
            name="align",
            ok=False,
            message=_tail(err) or "STAR did not produce expected BAM",
            metrics={"returncode": rc, **star_metrics},
        )

    _, err_idx, rc_idx = _run(["samtools", "index", str(bam_path)])
    if rc_idx != 0:
        return NodeResult(
            name="align",
            ok=False,
            message=f"samtools index failed: {_tail(err_idx)}",
            outputs={"bam": str(bam_path), "star_log": str(log_path)},
            metrics={"returncode": rc_idx},
        )

    _, err_qc, rc_qc = _run(["samtools", "quickcheck", str(bam_path)])

    star_metrics = _parse_star_log(log_path)
    pct = star_metrics.get("pct_uniquely_mapped")
    mapping_note = f" ({pct:.1f}% uniquely mapped)" if pct is not None else ""

    return NodeResult(
        name="align",
        ok=(rc_qc == 0),
        message=(
            f"ok{mapping_note}" if rc_qc == 0 else f"quickcheck failed: {_tail(err_qc)}"
        ),
        outputs={"bam": str(bam_path), "star_log": str(log_path)},
        metrics={"returncode": rc_qc, **star_metrics},
    )


def node_featurecounts(
    cfg: PreprocessingConfig, sample: Sample, bam_path: Path
) -> NodeResult:
    counts_dir = cfg.out_dir / "04_counts" / sample.name
    counts_dir.mkdir(parents=True, exist_ok=True)
    out_file = counts_dir / "counts.tsv"

    strand_flag = _STRAND_MAP.get(cfg.strand, "0")
    cmd = [
        "featureCounts",
        "-T",
        str(cfg.threads),
        "-a",
        str(cfg.gtf),
        "-F",
        "GTF",
        "-t",
        "exon",
        "-g",
        "gene_id",
        "-s",
        strand_flag,
        "-o",
        str(out_file),
    ]
    if sample.r2 is not None:
        cmd.append("-p")
    cmd.append(str(bam_path))

    _, err, rc = _run(cmd)
    if rc != 0 or not out_file.exists():
        return NodeResult(
            name="featurecounts",
            ok=False,
            message=_tail(err) or "featureCounts did not produce output",
            metrics={"returncode": rc},
        )

    n_lines = sum(1 for _ in out_file.open())
    ok = n_lines > 2
    return NodeResult(
        name="featurecounts",
        ok=ok,
        message="ok" if ok else "counts file unexpectedly small",
        outputs={"counts_tsv": str(out_file)},
        metrics={"returncode": rc, "n_lines": n_lines},
    )


# ---------- normalization ----------


def _parse_counts(counts_tsv: Path) -> tuple[list[str], list[int], list[int]]:
    """Parse a featureCounts output file.

    Returns (gene_ids, lengths, counts).
    Skips the leading comment line (starts with '#') and the column header row.
    Column 5 is Length; the last column is the count for the single BAM.
    """
    gene_ids: list[str] = []
    lengths: list[int] = []
    counts: list[int] = []
    with counts_tsv.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "Geneid":
                continue
            gene_ids.append(parts[0])
            lengths.append(int(parts[5]))
            counts.append(int(parts[-1]))
    return gene_ids, lengths, counts


def _compute_tpm(counts: list[int], lengths: list[int]) -> list[float]:
    rpk = [c / (l / 1000) if l > 0 else 0.0 for c, l in zip(counts, lengths)]
    scale = sum(rpk) / 1e6
    return [r / scale if scale > 0 else 0.0 for r in rpk]


def _compute_fpkm_rpkm(counts: list[int], lengths: list[int]) -> list[float]:
    # FPKM (paired-end) and RPKM (single-end) share the same formula.
    # We compute once and write to whichever output files were requested.
    lib_size = sum(counts)
    if lib_size == 0:
        return [0.0] * len(counts)
    return [c * 1e9 / (l * lib_size) if l > 0 else 0.0 for c, l in zip(counts, lengths)]


def _write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")


def node_normalize(
    cfg: PreprocessingConfig,
    count_results: list[tuple[str, Path]],
) -> NodeResult:
    """Merge per-sample counts and write normalised matrices to 05_normalized/.

    count_results: list of (sample_name, path_to_counts_tsv)
    """
    norm_dir = cfg.out_dir / "05_normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    all_gene_ids: list[str] = []
    all_lengths: list[int] = []
    sample_counts: dict[str, list[int]] = {}

    for sname, counts_tsv in count_results:
        gene_ids, lengths, counts = _parse_counts(counts_tsv)
        if not all_gene_ids:
            all_gene_ids = gene_ids
            all_lengths = lengths
        elif all_gene_ids != gene_ids:
            return NodeResult(
                name="normalize",
                ok=False,
                message=f"Gene list mismatch for sample '{sname}' vs first sample",
            )
        sample_counts[sname] = counts

    sample_names = [s for s, _ in count_results]
    n_genes = len(all_gene_ids)
    written: dict[str, str] = {}

    # Raw counts matrix
    raw_tsv = norm_dir / "counts_raw.tsv"
    header = ["gene_id", "length"] + sample_names
    rows: list[list] = [
        [all_gene_ids[i], all_lengths[i]] + [sample_counts[s][i] for s in sample_names]
        for i in range(n_genes)
    ]
    _write_tsv(raw_tsv, header, rows)
    written["counts_raw"] = str(raw_tsv)

    norm_header = ["gene_id"] + sample_names

    if "tpm" in cfg.normalizations:
        tpm_by_sample = {
            s: _compute_tpm(sample_counts[s], all_lengths) for s in sample_names
        }
        tpm_tsv = norm_dir / "tpm.tsv"
        tpm_rows: list[list] = [
            [all_gene_ids[i]] + [round(tpm_by_sample[s][i], 6) for s in sample_names]
            for i in range(n_genes)
        ]
        _write_tsv(tpm_tsv, norm_header, tpm_rows)
        written["tpm"] = str(tpm_tsv)

    if "fpkm" in cfg.normalizations or "rpkm" in cfg.normalizations:
        fpkm_by_sample = {
            s: _compute_fpkm_rpkm(sample_counts[s], all_lengths) for s in sample_names
        }
        for norm in ("fpkm", "rpkm"):
            if norm not in cfg.normalizations:
                continue
            out_tsv = norm_dir / f"{norm}.tsv"
            out_rows: list[list] = [
                [all_gene_ids[i]]
                + [round(fpkm_by_sample[s][i], 6) for s in sample_names]
                for i in range(n_genes)
            ]
            _write_tsv(out_tsv, norm_header, out_rows)
            written[norm] = str(out_tsv)

    return NodeResult(
        name="normalize",
        ok=True,
        message=f"wrote {list(written.keys())} for {len(sample_names)} sample(s)",
        outputs=written,
        metrics={"n_genes": n_genes, "n_samples": len(sample_names)},
    )


# ---------- DEG (PyDESeq2) ----------


def _strip_ensembl_version(gene_id: str) -> str:
    """ENSG00000123456.5 → ENSG00000123456 (mygene lookups want unversioned IDs)."""
    return gene_id.split(".", 1)[0]


def _read_counts_raw(raw_tsv: Path) -> tuple[list[str], list[str], list[list[int]]]:
    """Parse counts_raw.tsv (gene_id, length, sample1, sample2, ...).

    Returns (gene_ids, sample_names, matrix as gene-major rows of ints).
    """
    gene_ids: list[str] = []
    matrix: list[list[int]] = []
    sample_names: list[str] = []
    with raw_tsv.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        # header: ["gene_id", "length", sample1, sample2, ...]
        sample_names = header[2:]
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            gene_ids.append(parts[0])
            matrix.append([int(x) for x in parts[2:]])
    return gene_ids, sample_names, matrix


def _fetch_string_pairs(
    ensps: list[str],
    confidence: int = 400,
) -> set[tuple[str, str]]:
    """Query the STRING DB REST API for known interactions among the given ENSPs.

    POSTs to https://string-db.org/api/json/network with 9606-prefixed
    identifiers and the required_score threshold (0–1000).

    Returns a set of (stringId_A, stringId_B) tuples where
    stringId_A < stringId_B so each pair appears exactly once.
    Returns an empty set on any network or parsing failure so the
    pipeline never crashes when STRING is unreachable.
    """
    import json
    import urllib.parse
    import urllib.request

    if not ensps:
        return set()

    identifiers = "\n".join(f"9606.{e}" for e in ensps)
    payload = urllib.parse.urlencode(
        {
            "identifiers": identifiers,
            "species": 9606,
            "required_score": confidence,
            "caller_identity": "rnaseq_package",
        }
    ).encode()

    try:
        req = urllib.request.Request(
            "https://string-db.org/api/json/network",
            data=payload,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            records = json.loads(resp.read().decode())
    except Exception as exc:
        print(
            f"  [deg] WARNING: STRING API call failed "
            f"({type(exc).__name__}: {exc}) — skipping STRING pairs, "
            "all DEG pairs will use significance-ranked fallback.",
            flush=True,
        )
        return set()

    pairs: set[tuple[str, str]] = set()
    for rec in records:
        a = rec.get("stringId_A", "")
        b = rec.get("stringId_B", "")
        if a and b and a != b:
            pairs.add((min(a, b), max(a, b)))

    return pairs


def _build_deg_pairs_tsv(
    significant_genes: list[str],
    gene_scores: dict[str, float],
    out_path: Path,
    string_confidence: int = 400,
    fallback_max_pairs: int = 500,
) -> tuple[int, int, int, int]:
    """Build ENSP interaction pairs using STRING first, orphan fallback second.

    Steps:
      1. Map ENSG → ENSP via mygene (unversioned IDs).
      2. Query STRING for known interactions among all mapped ENSPs.
      3. Write every STRING-backed pair (no cap) tagged source=string.
      4. Find orphan ENSPs (zero STRING edges); rank their source genes by
         significance score descending; write combinatorial pairs among
         those orphans up to fallback_max_pairs tagged source=fallback.

    The ``source`` column lets downstream enrichment treat STRING pairs as
    known interactions and fallback pairs as novel hypotheses.

    Returns (n_genes_mapped, n_ensps, n_string_pairs, n_fallback_pairs).
    """
    from itertools import combinations

    import mygene

    mg = mygene.MyGeneInfo()

    # mygene returns bare (unversioned) query IDs; keep a reverse map
    # so we can look up the original gene_id for score retrieval.
    bare_to_orig: dict[str, str] = {
        _strip_ensembl_version(g): g for g in significant_genes
    }
    bare_genes = list(bare_to_orig.keys())

    results = mg.querymany(
        bare_genes,
        scopes="ensembl.gene",
        fields="ensembl.protein",
        species="human",
        verbose=False,
    )

    # bare ENSP → original gene_id (for significance-based orphan ranking)
    ensp_to_gene: dict[str, str] = {}
    n_mapped = 0

    for r in results:
        query_bare = r.get("query", "")
        ens = r.get("ensembl")
        if not ens:
            continue
        n_mapped += 1
        orig_gene = bare_to_orig.get(query_bare, query_bare)

        proteins: list[str] = []
        if isinstance(ens, list):
            for e in ens:
                p = e.get("protein")
                if isinstance(p, list):
                    proteins.extend(p)
                elif isinstance(p, str):
                    proteins.append(p)
        else:
            p = ens.get("protein")
            if isinstance(p, list):
                proteins.extend(p)
            elif isinstance(p, str):
                proteins.append(p)

        for ensp in proteins:
            # first gene to map this ENSP wins (avoids overwriting with a
            # less-significant gene if the same protein is shared)
            ensp_to_gene.setdefault(ensp, orig_gene)

    all_ensps = sorted(set(ensp_to_gene.keys()))

    # ---- STRING-backed pairs (no cap) --------------------------------
    string_pairs = _fetch_string_pairs(all_ensps, confidence=string_confidence)

    # Track which bare ENSPs appear in at least one STRING edge.
    # STRING IDs are formatted as "9606.ENSP..." — strip the taxon prefix.
    ensps_with_edges: set[str] = set()
    for sid_a, sid_b in string_pairs:
        ensps_with_edges.add(sid_a.split(".", 1)[-1])
        ensps_with_edges.add(sid_b.split(".", 1)[-1])

    # ---- Significance-ranked orphan fallback -------------------------
    orphan_ensps = [e for e in all_ensps if e not in ensps_with_edges]

    # Score = -log10(padj) × |log2FC|: captures both significance and
    # effect size so the highest-priority novel genes pair first.
    orphan_ensps.sort(
        key=lambda e: gene_scores.get(ensp_to_gene.get(e, ""), 0.0),
        reverse=True,
    )

    # ---- Write TSV ---------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_string_written = 0
    n_fallback_written = 0

    with out_path.open("w") as fh:
        fh.write("ensp_a\tensp_b\tsource\n")

        for sid_a, sid_b in sorted(string_pairs):
            fh.write(f"{sid_a}\t{sid_b}\tstring\n")
            n_string_written += 1

        for a, b in combinations(orphan_ensps, 2):
            if n_fallback_written >= fallback_max_pairs:
                break
            fh.write(f"9606.{a}\t9606.{b}\tfallback\n")
            n_fallback_written += 1

    return n_mapped, len(all_ensps), n_string_written, n_fallback_written


def node_deg(
    cfg: PreprocessingConfig,
    raw_counts_path: Path,
) -> NodeResult:
    """Differential expression analysis with PyDESeq2.

    Reads counts_raw.tsv (genes × samples), builds the sample → condition
    metadata table from cfg.samples, and runs:

        DeseqDataSet → dds.deseq2() → DeseqStats(contrast=[treated, reference])

    Significant genes are those with ``padj < cfg.padj_threshold`` AND
    ``|log2FoldChange| > cfg.lfc_threshold``.

    Outputs (all under cfg.out_dir/06_deg/):
        deseq2_full.tsv         every gene + log2FC, padj, etc.
        deseq2_significant.tsv  filtered to the significant set
        deg_gene_list.txt       just the gene IDs (one per line)
        deg_pairs.tsv           combinatorial ENSP pairs (if build_pairs_for_enrichment)
    """
    # If the user opted out at config time, skip immediately with a clear note.
    if not cfg.enable_deg:
        return NodeResult(
            name="deg",
            ok=True,
            message="skipped: cfg.enable_deg=False",
            metrics={"skipped": True},
        )

    # ---- sanity: enough conditions / replicates? ----
    cond_to_samples: dict[str, list[str]] = {}
    for s in cfg.samples:
        if s.condition is None:
            return NodeResult(
                name="deg",
                ok=False,
                message=f"sample '{s.name}' has no condition assigned",
            )
        cond_to_samples.setdefault(s.condition, []).append(s.name)

    if len(cond_to_samples) < 2:
        return NodeResult(
            name="deg",
            ok=False,
            message=f"need >=2 conditions for DEG, got {list(cond_to_samples.keys())}",
        )

    too_few = {c: n for c, n in cond_to_samples.items() if len(n) < 2}
    if too_few:
        return NodeResult(
            name="deg",
            ok=False,
            message=(
                "PyDESeq2 needs >=2 replicates per condition for stable variance "
                f"estimation. Conditions with <2 replicates: {too_few}"
            ),
        )

    if cfg.reference_condition not in cond_to_samples:
        return NodeResult(
            name="deg",
            ok=False,
            message=(
                f"reference_condition='{cfg.reference_condition}' not in "
                f"sample conditions {list(cond_to_samples)}"
            ),
        )
    treated = cfg.treated_condition
    if treated is None:
        non_ref = [c for c in cond_to_samples if c != cfg.reference_condition]
        treated = non_ref[0]
    if treated not in cond_to_samples:
        return NodeResult(
            name="deg",
            ok=False,
            message=(
                f"treated_condition='{treated}' not in sample conditions "
                f"{list(cond_to_samples)}"
            ),
        )

    # ---- run DESeq2 ----
    out_dir = cfg.out_dir / "06_deg"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Lazy heavy imports — only paid when this node actually runs.
        import pandas as pd
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats

        gene_ids, sample_names, matrix = _read_counts_raw(raw_counts_path)

        # PyDESeq2 wants samples × genes (rows × columns).
        counts_df = pd.DataFrame(matrix, index=gene_ids, columns=sample_names).T

        # Metadata: rows are samples, must contain a 'condition' column.
        cond_lookup = {s.name: s.condition for s in cfg.samples}
        metadata = pd.DataFrame(
            {"condition": [cond_lookup[s] for s in sample_names]},
            index=sample_names,
        )

        inference = DefaultInference(n_cpus=max(1, cfg.threads))
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design_factors="condition",
            refit_cooks=True,
            inference=inference,
            quiet=True,
        )
        dds.deseq2()

        ds = DeseqStats(
            dds,
            contrast=["condition", treated, cfg.reference_condition],
            inference=inference,
            quiet=True,
        )
        ds.summary()
        res_df: "pd.DataFrame" = ds.results_df.copy()
        res_df.index.name = "gene_id"

        # ---- write full results ----
        full_tsv = out_dir / "deseq2_full.tsv"
        res_df.to_csv(full_tsv, sep="\t")

        # ---- filter significant ----
        sig_mask = (
            res_df["padj"].notna()
            & (res_df["padj"] < cfg.padj_threshold)
            & (res_df["log2FoldChange"].abs() > cfg.lfc_threshold)
        )
        sig_df = res_df[sig_mask].sort_values("padj")
        sig_tsv = out_dir / "deseq2_significant.tsv"
        sig_df.to_csv(sig_tsv, sep="\t")

        gene_list_txt = out_dir / "deg_gene_list.txt"
        gene_list_txt.write_text("\n".join(sig_df.index.tolist()) + "\n")

        outputs = {
            "deseq2_full": str(full_tsv),
            "deseq2_significant": str(sig_tsv),
            "deg_gene_list": str(gene_list_txt),
        }
        metrics = {
            "n_genes_tested": int(res_df.shape[0]),
            "n_significant": int(sig_df.shape[0]),
            "padj_threshold": cfg.padj_threshold,
            "lfc_threshold": cfg.lfc_threshold,
            "treated": treated,
            "reference": cfg.reference_condition,
        }
        msg_parts = [
            f"DESeq2 done: {sig_df.shape[0]} sig genes "
            f"(of {res_df.shape[0]} tested) "
            f"@ padj<{cfg.padj_threshold}, |log2FC|>{cfg.lfc_threshold}, "
            f"contrast {treated} vs {cfg.reference_condition}"
        ]

        # ---- optional: gene → ENSP pairs for enrichment ----
        if cfg.build_pairs_for_enrichment and sig_df.shape[0] > 0:
            try:
                # Score = -log10(padj) × |log2FC|: ranks genes by combined
                # statistical significance and effect size so the orphan
                # fallback prioritises the most biologically relevant hits.
                gene_scores = {
                    gene: -math.log10(float(row["padj"]) + 1e-300)
                    * abs(float(row["log2FoldChange"]))
                    for gene, row in sig_df.iterrows()
                }
                pairs_tsv = out_dir / "deg_pairs.tsv"
                n_mapped, n_ensps, n_string, n_fallback = _build_deg_pairs_tsv(
                    sig_df.index.tolist(),
                    gene_scores,
                    pairs_tsv,
                    string_confidence=cfg.string_confidence_threshold,
                    fallback_max_pairs=cfg.pairs_fallback_max,
                )
                outputs["deg_pairs"] = str(pairs_tsv)
                metrics.update(
                    {
                        "deg_pairs_n_genes_mapped": n_mapped,
                        "deg_pairs_n_ensps": n_ensps,
                        "deg_pairs_n_string": n_string,
                        "deg_pairs_n_fallback": n_fallback,
                    }
                )
                msg_parts.append(
                    f"; mapped {n_mapped} genes to {n_ensps} ENSPs"
                    f" → {n_string} STRING pairs + {n_fallback} fallback pairs"
                )
            except Exception as exc:
                # Don't fail the whole DEG node just because the pair builder hit
                # a network/mygene issue — surface it in the message instead.
                msg_parts.append(
                    f"; (pair builder skipped: {type(exc).__name__}: {exc})"
                )

    except Exception as exc:
        return NodeResult(
            name="deg",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    return NodeResult(
        name="deg",
        ok=True,
        message="".join(msg_parts),
        outputs=outputs,
        metrics=metrics,
    )
