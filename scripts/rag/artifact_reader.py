"""Scan a pipeline run directory and read ALL available artifacts.

This replaces the need to manually pass file paths to the RAG chat.
Call ``scan_run_dir(run_dir)`` and get back a single dict with every
piece of information the pipeline produced.

Artifact sources read
---------------------
  runs/<run>/preprocessing_run.json  → QC gate decisions, config
  runs/<run>/01_qc/<sample>/         → FastQC reports (from zip)
  runs/<run>/03_bam/<sample>/        → STAR *Log.final.out
  runs/<run>/04_counts/<sample>/     → featureCounts *.summary
  runs/<run>/05_norm/                → normalisation method list
  runs/<run>/06_deg/                 → deseq2_full.tsv, deseq2_significant.tsv
  runs/<run_enrichment>/             → inference.json, test_metrics.json
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# STAR log parser
# ---------------------------------------------------------------------------


def _parse_star_log(log_path: Path) -> dict[str, Any]:
    """Parse STAR *Log.final.out and return key alignment stats.

    Uses substring matching on the label (left of '|') so minor whitespace
    variations in STAR output don't cause missed fields.
    """
    stats: dict[str, Any] = {}
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
        ("Average mapped length", "avg_mapped_length", False),
    ]
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            if "|" not in line:
                continue
            label, _, val = line.partition("|")
            label = label.strip()
            val = val.strip().rstrip("%")
            for star_label, key, _is_pct in _FIELD_MAP:
                if star_label in label:
                    try:
                        stats[key] = int(val) if "." not in val else float(val)
                    except ValueError:
                        print("No labels found")
                    break
    except Exception:
        pass
    pct = stats.get("pct_uniquely_mapped")
    stats["mapping_rate_ok"] = pct is not None and pct >= 70.0
    return stats


# ---------------------------------------------------------------------------
# featureCounts summary parser
# ---------------------------------------------------------------------------


def _parse_fc_summary(summary_path: Path) -> dict[str, Any]:
    """Parse featureCounts *.summary file and return assignment stats."""
    stats: dict[str, Any] = {}
    try:
        lines = summary_path.read_text().splitlines()
        total = 0
        assigned = 0
        for line in lines[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0]
            try:
                count = int(parts[1])
            except ValueError:
                continue
            total += count
            if status == "Assigned":
                assigned = count
            stats[f"fc_{status.lower()}"] = count
        if total > 0:
            stats["pct_fc_assigned"] = round(100 * assigned / total, 1)
        stats["fc_total"] = total
    except Exception:
        print("Could not parse feature count summary file please check the file again")
    return stats


# ---------------------------------------------------------------------------
# FastQC report parser (mirrors the one in nodes.py, standalone copy)
# ---------------------------------------------------------------------------


def _parse_fastqc_zip(zip_path: Path) -> dict[str, Any]:
    """Extract key QC metrics from a FastQC zip file."""
    result: dict[str, Any] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            data_files = [n for n in zf.namelist() if n.endswith("fastqc_data.txt")]
            if not data_files:
                return result
            content = zf.read(data_files[0]).decode("utf-8", errors="replace")
    except Exception:
        return result

    # Module statuses
    modules: dict[str, str] = {}
    for line in content.splitlines():
        if line.startswith(">>") and not line.startswith(">>END"):
            parts = line[2:].split("\t")
            if len(parts) == 2:
                modules[parts[0]] = parts[1].strip()
    result["modules"] = modules

    m = re.search(r"Total Sequences\s+(\d+)", content)
    if m:
        result["total_sequences"] = int(m.group(1))
    m = re.search(r"%GC\s+(\d+)", content)
    if m:
        result["pct_gc"] = float(m.group(1))
    m = re.search(r"#Total Deduplicated Percentage\s+([\d.]+)", content)
    if m:
        result["pct_duplicates"] = round(100 - float(m.group(1)), 1)
    return result


# ---------------------------------------------------------------------------
# preprocessing_run.json reader
# ---------------------------------------------------------------------------


def _read_preprocessing_json(run_dir: Path) -> dict[str, Any]:
    """Read preprocessing_run.json if present (has QC gate decisions)."""
    p = run_dir / "preprocessing_run.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# normalisation dir scanner
# ---------------------------------------------------------------------------


def _read_norm_info(run_dir: Path) -> dict[str, Any]:
    norm_dir = run_dir / "05_norm"
    if not norm_dir.exists():
        return {}
    methods = [p.stem for p in norm_dir.glob("*.tsv")]
    # Read first few lines of each norm file to get sample names
    samples: list[str] = []
    for tsv in sorted(norm_dir.glob("*.tsv"))[:1]:
        try:
            with tsv.open() as fh:
                header = fh.readline().rstrip("\n").split("\t")
            samples = header[1:]  # first col is gene_id
        except Exception:
            pass
    return {"norm_methods": methods, "norm_samples": samples}


# ---------------------------------------------------------------------------
# DEG reader
# ---------------------------------------------------------------------------


def _read_deg_files(run_dir: Path) -> dict[str, Any]:
    """Read DEG tables from the run directory."""
    deg_dir = run_dir / "06_deg"
    result: dict[str, Any] = {}

    full = deg_dir / "deseq2_full.tsv"
    if full.exists():
        result["deg_full_path"] = str(full)
        try:
            with full.open() as fh:
                result["n_genes_tested"] = sum(1 for _ in fh) - 1
        except Exception:
            pass

    sig = deg_dir / "deseq2_significant.tsv"
    if sig.exists():
        result["deg_sig_path"] = str(sig)
        try:
            with sig.open() as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            result["n_deg_significant"] = len(rows)
            result["deg_sample_rows"] = rows[:5]  # first 5 for display
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# enrichment artifacts reader
# ---------------------------------------------------------------------------


def _read_enrichment_artifacts(run_dir: Path) -> dict[str, Any]:
    """Find inference.json and test_metrics.json in sibling enrichment dirs."""
    result: dict[str, Any] = {}
    parent = run_dir.parent

    for candidate in sorted(
        parent.glob("*enrichment*"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        inf = candidate / "inference.json"
        if inf.exists() and "inference_path" not in result:
            result["inference_path"] = str(inf)
            try:
                data = json.loads(inf.read_text())
                result["inference_n_predicted"] = data.get("n_predicted")
                result["inference_n_skipped"] = data.get("n_skipped")
            except Exception:
                pass

        metrics = candidate / "test_metrics.json"
        if metrics.exists() and "gnn_f1" not in result:
            try:
                m = json.loads(metrics.read_text())
                all_m = m.get("all") or {}
                result["gnn_f1"] = all_m.get("f1")
                result["gnn_precision"] = all_m.get("p")
                result["gnn_recall"] = all_m.get("r")
                result["gnn_n_test"] = all_m.get("n")
                result["test_metrics_path"] = str(metrics)
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def scan_run_dir(run_dir: str | Path) -> dict[str, Any]:
    """Scan a pipeline run directory and return all available artifacts.

    Returns a comprehensive dict:
    {
        "run_dir":          str,
        "config":           dict,            # from preprocessing_run.json
        "qc_gate_decisions": dict,           # sample → {decision, trim_steps}
        "failed_samples":   list[str],
        "samples": {
            <name>: {
                "fastqc":   dict,            # module statuses + metrics
                "star":     dict,            # alignment stats
                "fc":       dict,            # featureCounts stats
            }
        },
        "norm_methods":     list[str],
        "norm_samples":     list[str],
        "n_genes_tested":   int | None,
        "n_deg_significant": int | None,
        "deg_full_path":    str | None,
        "deg_sig_path":     str | None,
        "inference_path":   str | None,
        "inference_n_predicted": int | None,
        "inference_n_skipped":   int | None,
        "gnn_f1":           float | None,
        "gnn_precision":    float | None,
        "gnn_recall":       float | None,
    }
    """
    run_dir = Path(run_dir)
    arts: dict[str, Any] = {"run_dir": str(run_dir)}

    # preprocessing_run.json — config + QC gate decisions + failed samples
    pre = _read_preprocessing_json(run_dir)
    arts["config"] = pre.get("config") or {}
    arts["failed_samples"] = pre.get("failed_samples") or []
    arts["qc_gate_decisions"] = {}  # filled per-sample below if not in json

    # Per-sample artifacts: FastQC, STAR, featureCounts
    samples: dict[str, dict] = {}
    qc_dir = run_dir / "01_qc"
    bam_dir = run_dir / "03_bam"
    counts_dir = run_dir / "04_counts"

    # Discover sample names from the QC dir (most reliably populated)
    sample_names: list[str] = []
    if qc_dir.exists():
        sample_names = [d.name for d in sorted(qc_dir.iterdir()) if d.is_dir()]

    for sname in sample_names:
        entry: dict[str, Any] = {}

        # FastQC
        sq = qc_dir / sname
        if sq.exists():
            zips = list(sq.glob("*_fastqc.zip"))
            if zips:
                entry["fastqc"] = _parse_fastqc_zip(zips[0])

        # STAR log
        sb = bam_dir / sname
        if sb.exists():
            logs = list(sb.glob("*Log.final.out"))
            if logs:
                entry["star"] = _parse_star_log(logs[0])

        # featureCounts summary
        sc = counts_dir / sname
        if sc.exists():
            summaries = list(sc.glob("*.summary"))
            if summaries:
                entry["fc"] = _parse_fc_summary(summaries[0])

        samples[sname] = entry

    arts["samples"] = samples

    # Normalisation info
    arts.update(_read_norm_info(run_dir))

    # DEG files
    arts.update(_read_deg_files(run_dir))

    # Enrichment artifacts (inference + GNN metrics)
    arts.update(_read_enrichment_artifacts(run_dir))

    return arts


# ---------------------------------------------------------------------------
# format for LLM prompt
# ---------------------------------------------------------------------------


def format_artifacts_for_prompt(arts: dict[str, Any]) -> str:
    """Format the full artifact scan into a concise text block for the LLM.

    Covers: config, per-sample alignment + QC stats, DEG summary,
    GNN metrics. Kept under ~1500 chars to fit small-context models.
    """
    if not arts:
        return "(no pipeline artifacts available)"

    lines: list[str] = ["=== Full pipeline run report ==="]

    cfg = arts.get("config") or {}
    if cfg.get("genome_dir"):
        lines.append(f"Reference genome  : {cfg['genome_dir']}")
    if cfg.get("gtf_file"):
        lines.append(f"Annotation (GTF)  : {cfg['gtf_file']}")
    if cfg.get("normalizations"):
        lines.append(f"Normalisation     : {', '.join(cfg['normalizations'])}")

    samples = arts.get("samples") or {}
    if samples:
        lines.append(f"\nSamples processed : {len(samples)}")
        failed = arts.get("failed_samples") or []
        if failed:
            lines.append(f"Failed samples    : {', '.join(failed)}")

        lines.append("\nPer-sample alignment stats:")
        for sname, s in samples.items():
            star = s.get("star") or {}
            fc = s.get("fc") or {}
            fqc = s.get("fastqc") or {}
            mods = fqc.get("modules") or {}
            ada = mods.get("Adapter Content", "?")
            q = mods.get("Per base sequence quality", "?")
            pct_map = star.get("pct_uniquely_mapped")
            pct_multi = star.get("pct_multi_mapped")
            pct_short = star.get("pct_unmapped_tooshort")
            mismatch = star.get("mismatch_rate_pct")
            n_input = star.get("n_input_reads")
            pct_asgn = fc.get("pct_fc_assigned", "?")
            map_ok = star.get("mapping_rate_ok")
            flag = " [LOW MAPPING]" if map_ok is False else ""
            parts = [f"  {sname}"]
            if n_input is not None:
                parts.append(f"input={n_input:,}")
            parts.append(
                f"uniquely_mapped={pct_map if pct_map is not None else '?'}%{flag}"
            )
            if pct_multi is not None:
                parts.append(f"multi={pct_multi}%")
            if pct_short is not None:
                parts.append(f"too_short={pct_short}%")
            if mismatch is not None:
                parts.append(f"mismatch={mismatch}%")
            parts.append(f"fc_assigned={pct_asgn}%")
            parts.append(f"adapter={ada}  quality={q}")
            lines.append("  " + "  ".join(parts))

    if arts.get("n_genes_tested") is not None:
        lines.append(f"\nGenes tested (DESeq2): {arts['n_genes_tested']}")
    if arts.get("n_deg_significant") is not None:
        lines.append(f"Significant DEGs     : {arts['n_deg_significant']}")

    if arts.get("gnn_f1") is not None:
        lines.append(
            f"\nGNN PPI model  F1={arts['gnn_f1']:.3f}  "
            f"P={arts.get('gnn_precision', '?'):.3f}  "
            f"R={arts.get('gnn_recall', '?'):.3f}"
        )
    if arts.get("inference_n_predicted") is not None:
        lines.append(f"PPI predicted  : {arts['inference_n_predicted']} pairs")

    return "\n".join(lines)
