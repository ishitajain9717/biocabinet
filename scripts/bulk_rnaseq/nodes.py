"""Pipeline nodes: each function runs one tool and returns a NodeResult."""
from __future__ import annotations

import subprocess
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


# ---------- nodes ----------

def node_fastqc(cfg: PreprocessingConfig, sample: Sample) -> NodeResult:
    out_dir = cfg.out_dir / "01_qc" / sample.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["fastqc", "-o", str(out_dir), str(sample.r1)]
    if sample.r2 is not None:
        cmd.append(str(sample.r2))

    _, err, rc = _run(cmd)
    return NodeResult(
        name="fastqc",
        ok=(rc == 0),
        message="ok" if rc == 0 else _tail(err),
        outputs={"qc_dir": str(out_dir)},
        metrics={"returncode": rc},
    )


def node_trim(cfg: PreprocessingConfig, sample: Sample) -> NodeResult:
    if cfg.skip_trim:
        outs: dict[str, str] = {"r1": str(sample.r1)}
        if sample.r2 is not None:
            outs["r2"] = str(sample.r2)
        return NodeResult(name="trim", ok=True, message="skipped", outputs=outs)

    if cfg.trimmomatic_jar is None:
        return NodeResult(
            name="trim",
            ok=False,
            message="trimmomatic_jar is required when skip_trim=False",
        )

    trim_dir = cfg.out_dir / "02_trim" / sample.name
    trim_dir.mkdir(parents=True, exist_ok=True)

    if sample.r2 is not None:
        r1_out = trim_dir / f"{sample.name}.R1_trimmed.fastq.gz"
        r2_out = trim_dir / f"{sample.name}.R2_trimmed.fastq.gz"
        r1_unp = trim_dir / f"{sample.name}.R1_unpaired.fastq.gz"
        r2_unp = trim_dir / f"{sample.name}.R2_unpaired.fastq.gz"
        cmd = [
            "java", "-jar", str(cfg.trimmomatic_jar),
            "PE", "-threads", str(cfg.threads),
            str(sample.r1), str(sample.r2),
            str(r1_out), str(r1_unp),
            str(r2_out), str(r2_unp),
        ]
        _, err, rc = _run(cmd)
        return NodeResult(
            name="trim",
            ok=(rc == 0 and r1_out.exists() and r2_out.exists()),
            message="ok" if rc == 0 else _tail(err),
            outputs={"r1": str(r1_out), "r2": str(r2_out)},
            metrics={"returncode": rc},
        )

    r1_out = trim_dir / f"{sample.name}.R1_trimmed.fastq.gz"
    cmd = [
        "java", "-jar", str(cfg.trimmomatic_jar),
        "SE", "-threads", str(cfg.threads),
        str(sample.r1), str(r1_out),
    ]
    _, err, rc = _run(cmd)
    return NodeResult(
        name="trim",
        ok=(rc == 0 and r1_out.exists()),
        message="ok" if rc == 0 else _tail(err),
        outputs={"r1": str(r1_out)},
        metrics={"returncode": rc},
    )


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
    prefix = bam_dir / f"{sample.name}_"

    cmd = [
        "STAR",
        "--runThreadN", str(cfg.threads),
        "--genomeDir", str(cfg.genome_dir),
        "--readFilesIn", str(r1),
    ]
    if r2 is not None:
        cmd.append(str(r2))
    if str(r1).endswith(".gz"):
        cmd += ["--readFilesCommand", "gunzip", "-c"]
    cmd += [
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFileNamePrefix", str(prefix),
    ]

    _, err, rc = _run(cmd)
    bam_path = Path(str(prefix) + "Aligned.sortedByCoord.out.bam")
    if rc != 0 or not bam_path.exists():
        return NodeResult(
            name="align",
            ok=False,
            message=_tail(err) or "STAR did not produce expected BAM",
            metrics={"returncode": rc},
        )

    _, err_idx, rc_idx = _run(["samtools", "index", str(bam_path)])
    if rc_idx != 0:
        return NodeResult(
            name="align",
            ok=False,
            message=f"samtools index failed: {_tail(err_idx)}",
            outputs={"bam": str(bam_path)},
            metrics={"returncode": rc_idx},
        )

    _, err_qc, rc_qc = _run(["samtools", "quickcheck", str(bam_path)])
    return NodeResult(
        name="align",
        ok=(rc_qc == 0),
        message="ok" if rc_qc == 0 else f"quickcheck failed: {_tail(err_qc)}",
        outputs={"bam": str(bam_path)},
        metrics={"returncode": rc_qc},
    )


def node_featurecounts(
    cfg: PreprocessingConfig, sample: Sample, bam_path: Path
) -> NodeResult:
    counts_dir = cfg.out_dir / "04_counts" / sample.name
    counts_dir.mkdir(parents=True, exist_ok=True)
    out_file = counts_dir / "counts.txt"

    strand_flag = _STRAND_MAP.get(cfg.strand, "0")
    cmd = [
        "featureCounts",
        "-T", str(cfg.threads),
        "-a", str(cfg.gtf),
        "-F", "GTF",
        "-t", "exon",
        "-g", "gene_id",
        "-s", strand_flag,
        "-o", str(out_file),
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
        outputs={"counts_txt": str(out_file)},
        metrics={"returncode": rc, "n_lines": n_lines},
    )


# ---------- normalization ----------

def _parse_counts(counts_txt: Path) -> tuple[list[str], list[int], list[int]]:
    """Parse a featureCounts output file.

    Returns (gene_ids, lengths, counts).
    Skips the leading comment line (starts with '#') and the column header row.
    Column 5 is Length; the last column is the count for the single BAM.
    """
    gene_ids: list[str] = []
    lengths: list[int] = []
    counts: list[int] = []
    with counts_txt.open() as fh:
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
    return [
        c * 1e9 / (l * lib_size) if l > 0 else 0.0
        for c, l in zip(counts, lengths)
    ]


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

    count_results: list of (sample_name, path_to_counts_txt)
    """
    norm_dir = cfg.out_dir / "05_normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    all_gene_ids: list[str] = []
    all_lengths: list[int] = []
    sample_counts: dict[str, list[int]] = {}

    for sname, counts_txt in count_results:
        gene_ids, lengths, counts = _parse_counts(counts_txt)
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
        tpm_by_sample = {s: _compute_tpm(sample_counts[s], all_lengths) for s in sample_names}
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
                [all_gene_ids[i]] + [round(fpkm_by_sample[s][i], 6) for s in sample_names]
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
    gene_ids:    list[str] = []
    matrix:      list[list[int]] = []
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


def _build_deg_pairs_tsv(
    significant_genes: list[str],
    out_path:          Path,
    max_pairs:         int = 5000,
) -> tuple[int, int, int]:
    """Map ENSG → ENSP via mygene, build all combinations, cap at max_pairs.

    Returns (n_genes_mapped, n_total_ensps, n_pairs_written).
    """
    from itertools import combinations
    import mygene
    mg = mygene.MyGeneInfo()

    bare_genes = [_strip_ensembl_version(g) for g in significant_genes]
    results = mg.querymany(
        bare_genes, scopes="ensembl.gene", fields="ensembl.protein",
        species="human", verbose=False,
    )

    all_ensps: list[str] = []
    n_mapped = 0
    for r in results:
        ens = r.get("ensembl")
        if not ens:
            continue
        n_mapped += 1
        if isinstance(ens, list):
            for e in ens:
                p = e.get("protein")
                if isinstance(p, list):
                    all_ensps.extend(p)
                elif isinstance(p, str):
                    all_ensps.append(p)
        else:
            p = ens.get("protein")
            if isinstance(p, list):
                all_ensps.extend(p)
            elif isinstance(p, str):
                all_ensps.append(p)

    # dedup + sort for deterministic output
    all_ensps = sorted(set(all_ensps))

    pairs_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("ensp_a\tensp_b\n")
        for a, b in combinations(all_ensps, 2):
            if pairs_written >= max_pairs:
                break
            fh.write(f"9606.{a}\t9606.{b}\n")
            pairs_written += 1

    return n_mapped, len(all_ensps), pairs_written


def node_deg(
    cfg:          PreprocessingConfig,
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
        from pydeseq2.dds  import DeseqDataSet
        from pydeseq2.ds   import DeseqStats
        from pydeseq2.default_inference import DefaultInference

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
            "deseq2_full":        str(full_tsv),
            "deseq2_significant": str(sig_tsv),
            "deg_gene_list":      str(gene_list_txt),
        }
        metrics = {
            "n_genes_tested":    int(res_df.shape[0]),
            "n_significant":     int(sig_df.shape[0]),
            "padj_threshold":    cfg.padj_threshold,
            "lfc_threshold":     cfg.lfc_threshold,
            "treated":           treated,
            "reference":         cfg.reference_condition,
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
                pairs_tsv = out_dir / "deg_pairs.tsv"
                n_mapped, n_ensps, n_pairs = _build_deg_pairs_tsv(
                    sig_df.index.tolist(), pairs_tsv,
                )
                outputs["deg_pairs"] = str(pairs_tsv)
                metrics.update({
                    "deg_pairs_n_genes_mapped": n_mapped,
                    "deg_pairs_n_ensps":        n_ensps,
                    "deg_pairs_n_written":      n_pairs,
                })
                msg_parts.append(
                    f"; mapped {n_mapped} genes to {n_ensps} ENSPs → {n_pairs} candidate pairs"
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
