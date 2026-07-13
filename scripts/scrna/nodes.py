"""Pipeline nodes for the scRNA-seq workflow.

Each function runs one Scanpy step and returns a NodeResult plus
(when applicable) the path to the AnnData snapshot it wrote.

Output layout under cfg.out_dir/:
    00_load/01_loaded.h5ad
    01_qc/...
    02_filter/...
    03_normalize/...
    04_pca/...
    05_cluster/...
    06_markers/...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import scanpy as sc

from scripts.common.node_result import NodeResult
from scripts.scrna.config import ScrnaConfig

# ---------- internal helpers ----------


def _save_adata(adata, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)


# ---------- node 1: load_data ----------


def node_load_data(cfg: ScrnaConfig) -> tuple[NodeResult, Optional[Path]]:
    """Read the user's input into AnnData and save 00_load/01_loaded.h5ad.

    Supports three input kinds:
      - "pbmc3k" : downloads Scanpy's built-in 3k PBMC dataset (cached after first run)
      - "h5ad"   : reads cfg.input_path as an .h5ad file
      - "10x"    : reads cfg.input_path as a 10x Genomics output folder
                   (must contain matrix.mtx + barcodes.tsv + features.tsv)
    """
    out_path = cfg.out_dir / "00_load" / "01_loaded.h5ad"

    try:
        if cfg.input_kind == "pbmc3k":
            adata = sc.datasets.pbmc3k()
        elif cfg.input_kind == "h5ad":
            if cfg.input_path is None:
                raise ValueError("input_kind='h5ad' but input_path is None")
            adata = sc.read_h5ad(cfg.input_path)
        elif cfg.input_kind == "10x":
            if cfg.input_path is None:
                raise ValueError("input_kind='10x' but input_path is None")
            adata = sc.read_10x_mtx(cfg.input_path)
        else:
            raise ValueError(f"Unknown input_kind: {cfg.input_kind}")

        # Many real datasets have duplicate gene symbols in var_names;
        # downstream Scanpy ops crash on duplicates so we de-dup up front.
        adata.var_names_make_unique()

        _save_adata(adata, out_path)
    except Exception as exc:
        return (
            NodeResult(
                name="load_data",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="load_data",
            ok=True,
            message=(
                f"loaded {adata.n_obs} cells x "
                f"{adata.n_vars} genes ({cfg.input_kind})"
            ),
            outputs={"adata": str(out_path)},
            metrics={"n_cells": adata.n_obs, "n_genes": adata.n_vars},
        ),
        out_path,
    )


# ---------- node 2: qc ----------


def _detect_mt_prefix(adata) -> str:
    """Pick 'MT-' (human) or 'mt-' (mouse) based on which matches more genes."""
    n_human = int(adata.var_names.str.startswith("MT-").sum())
    n_mouse = int(adata.var_names.str.startswith("mt-").sum())
    return "MT-" if n_human >= n_mouse else "mt-"


def node_qc(cfg: ScrnaConfig, adata_in: Path) -> tuple[NodeResult, Optional[Path]]:
    """Compute per-cell and per-gene QC metrics; write violin/scatter plots.

    Does NOT filter — that's node_filter's job. We just calculate so the
    user (and the filter node) can see the distributions.
    """
    out_dir = cfg.out_dir / "01_qc"
    out_h5ad = out_dir / "02_qc.h5ad"

    try:
        adata = sc.read_h5ad(adata_in)

        # Flag mitochondrial and ribosomal genes by name prefix.
        mt_prefix = _detect_mt_prefix(adata)
        adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
        adata.var["ribo"] = adata.var_names.str.startswith(("RPL", "RPS", "Rpl", "Rps"))

        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt", "ribo"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

        # Save plots with Scanpy's built-ins. Scanpy auto-prepends the function
        # name to `save`, so save="_qc.png" produces violin_qc.png, etc.
        out_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(out_dir)

        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            multi_panel=True,
            show=False,
            save="_qc.png",
        )
        sc.pl.scatter(
            adata, x="total_counts", y="pct_counts_mt", show=False, save="_mt.png"
        )
        sc.pl.scatter(
            adata,
            x="total_counts",
            y="n_genes_by_counts",
            show=False,
            save="_genes.png",
        )

        _save_adata(adata, out_h5ad)

    except Exception as exc:
        return (
            NodeResult(
                name="qc",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="qc",
            ok=True,
            message=(
                f"QC done: median "
                f"{adata.obs['n_genes_by_counts'].median():.0f} genes/cell, "
                f"median {adata.obs['pct_counts_mt'].median():.2f}% mt"
            ),
            outputs={"adata": str(out_h5ad), "qc_dir": str(out_dir)},
            metrics={
                "median_n_genes": float(adata.obs["n_genes_by_counts"].median()),
                "median_pct_mt": float(adata.obs["pct_counts_mt"].median()),
                "median_total_counts": float(adata.obs["total_counts"].median()),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
            },
        ),
        out_h5ad,
    )


# ---------- node 3: filter ----------


def node_filter(cfg: ScrnaConfig, adata_in: Path) -> tuple[NodeResult, Optional[Path]]:
    """Drop low-quality cells and rarely-expressed genes using cfg thresholds.

    Order: genes first (smaller matrix → faster cell checks), then cells.
    Requires node_qc to have been run first (needs pct_counts_mt in obs).
    """
    out_dir = cfg.out_dir / "02_filter"
    out_path = out_dir / "03_filtered.h5ad"

    try:
        adata = sc.read_h5ad(adata_in)
        n_in_cells, n_in_genes = adata.n_obs, adata.n_vars

        if "pct_counts_mt" not in adata.obs.columns:
            raise ValueError(
                "pct_counts_mt not in adata.obs — node_qc must run before node_filter"
            )

        # Step 1: genes detected in < min_cells_per_gene cells.
        sc.pp.filter_genes(adata, min_cells=cfg.min_cells_per_gene)

        # Step 2: cells with < min_genes_per_cell genes detected.
        sc.pp.filter_cells(adata, min_genes=cfg.min_genes_per_cell)

        # Step 3: cells with too-high mitochondrial fraction.
        # Boolean indexing returns a view; force .copy() so downstream
        # writes don't trigger ImplicitModificationWarning or fail.
        adata = adata[adata.obs["pct_counts_mt"] < cfg.max_pct_mt].copy()

        if adata.n_obs == 0:
            raise ValueError(
                f"All cells filtered out — relax thresholds "
                f"(min_genes_per_cell={cfg.min_genes_per_cell}, "
                f"max_pct_mt={cfg.max_pct_mt})"
            )

        _save_adata(adata, out_path)

    except Exception as exc:
        return (
            NodeResult(
                name="filter",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="filter",
            ok=True,
            message=(
                f"kept {adata.n_obs}/{n_in_cells} cells, "
                f"{adata.n_vars}/{n_in_genes} genes"
            ),
            outputs={"adata": str(out_path)},
            metrics={
                "n_cells_in": n_in_cells,
                "n_cells_kept": int(adata.n_obs),
                "n_genes_in": n_in_genes,
                "n_genes_kept": int(adata.n_vars),
                "min_genes_per_cell": cfg.min_genes_per_cell,
                "min_cells_per_gene": cfg.min_cells_per_gene,
                "max_pct_mt": cfg.max_pct_mt,
            },
        ),
        out_path,
    )


# ---------- node 4: normalize ----------


def node_normalize(
    cfg: ScrnaConfig, adata_in: Path
) -> tuple[NodeResult, Optional[Path]]:
    """normalize_total → log1p → highly_variable_genes (flag, don't subset).

    The HVG flag goes into adata.var['highly_variable']; node_pca later
    sets use_highly_variable=True so PCA only uses those genes.

    Raw counts are stashed in adata.raw before normalization so the
    marker-gene node can fall back to them later (Scanpy convention).
    """
    out_dir = cfg.out_dir / "03_normalize"
    out_path = out_dir / "04_normalized.h5ad"

    try:
        adata = sc.read_h5ad(adata_in)

        sc.pp.normalize_total(adata, target_sum=cfg.target_sum)
        sc.pp.log1p(adata)

        # Stash a snapshot of the LOG-NORMALIZED data in adata.raw.
        # node_markers / rank_genes_groups will use adata.raw.X for the
        # Wilcoxon test — NOT the z-scored values that sc.pp.scale writes
        # into adata.X later in node_pca.
        adata.raw = adata

        sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_hvg)

        n_hvg = int(adata.var["highly_variable"].sum())

        out_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(out_dir)
        sc.pl.highly_variable_genes(adata, show=False, save="_hvg.png")

        _save_adata(adata, out_path)

    except Exception as exc:
        return (
            NodeResult(
                name="normalize",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="normalize",
            ok=True,
            message=f"normalized; flagged {n_hvg} highly-variable genes",
            outputs={"adata": str(out_path), "norm_dir": str(out_dir)},
            metrics={
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_hvg": n_hvg,
                "target_sum": cfg.target_sum,
            },
        ),
        out_path,
    )


# ---------- node 5: pca ----------


def node_pca(cfg: ScrnaConfig, adata_in: Path) -> tuple[NodeResult, Optional[Path]]:
    """Scale (z-score, optional clip) then PCA on highly-variable genes.

    Requires node_normalize to have run first (needs adata.var['highly_variable']
    and adata.raw — the latter so node_markers can fall back to raw counts later,
    because sc.pp.scale overwrites adata.X with z-scored values).
    """
    out_dir = cfg.out_dir / "04_pca"
    out_path = out_dir / "05_pca.h5ad"

    try:
        adata = sc.read_h5ad(adata_in)

        if "highly_variable" not in adata.var.columns:
            raise ValueError(
                "adata.var['highly_variable'] missing — "
                "node_normalize must run before node_pca"
            )

        # Z-score each gene; clipping is user-configurable via cfg.
        if cfg.scale_clip:
            sc.pp.scale(adata, max_value=cfg.scale_max_value)
        else:
            sc.pp.scale(adata)

        sc.tl.pca(adata, n_comps=cfg.n_pcs, use_highly_variable=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(out_dir)
        sc.pl.pca_variance_ratio(
            adata, n_pcs=cfg.n_pcs, log=True, show=False, save="_pca.png"
        )

        _save_adata(adata, out_path)

    except Exception as exc:
        return (
            NodeResult(
                name="pca",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    var_ratio = adata.uns["pca"]["variance_ratio"]
    cum_var_top10 = float(var_ratio[:10].sum())

    return (
        NodeResult(
            name="pca",
            ok=True,
            message=(
                f"PCA computed: {cfg.n_pcs} PCs, "
                f"top 10 explain {cum_var_top10:.1%} variance"
            ),
            outputs={"adata": str(out_path), "pca_dir": str(out_dir)},
            metrics={
                "n_pcs": cfg.n_pcs,
                "cum_variance_top10": cum_var_top10,
                "scale_clip": cfg.scale_clip,
                "scale_max_value": cfg.scale_max_value if cfg.scale_clip else None,
            },
        ),
        out_path,
    )


# ---------- node 6: cluster ----------


def node_cluster(cfg: ScrnaConfig, adata_in: Path) -> tuple[NodeResult, Optional[Path]]:
    """Build kNN graph in PCA space, run Leiden clustering, compute UMAP.

    Requires node_pca to have run first (needs adata.obsm['X_pca']).
    Order: neighbors → leiden → umap, so the UMAP plot can be coloured
    by cluster on the same call.
    """
    out_dir = cfg.out_dir / "05_cluster"
    out_path = out_dir / "06_clustered.h5ad"

    try:
        adata = sc.read_h5ad(adata_in)

        if "X_pca" not in adata.obsm:
            raise ValueError(
                "adata.obsm['X_pca'] missing — node_pca must run before node_cluster"
            )

        sc.pp.neighbors(adata, n_neighbors=cfg.n_neighbors, n_pcs=cfg.n_pcs)
        sc.tl.leiden(adata, resolution=cfg.leiden_resolution)
        sc.tl.umap(adata)

        n_clusters = int(adata.obs["leiden"].nunique())

        out_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(out_dir)
        # Headline plot — clusters on UMAP.
        sc.pl.umap(adata, color="leiden", show=False, save="_clusters.png")
        # QC-coloured UMAP — spots clusters that are just low-quality cells.
        sc.pl.umap(
            adata,
            color=["total_counts", "pct_counts_mt"],
            show=False,
            save="_qc.png",
        )

        _save_adata(adata, out_path)

    except Exception as exc:
        return (
            NodeResult(
                name="cluster",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="cluster",
            ok=True,
            message=(
                f"clustered into {n_clusters} clusters "
                f"(leiden res={cfg.leiden_resolution})"
            ),
            outputs={"adata": str(out_path), "cluster_dir": str(out_dir)},
            metrics={
                "n_clusters": n_clusters,
                "n_neighbors": cfg.n_neighbors,
                "leiden_resolution": cfg.leiden_resolution,
                "n_pcs_used": cfg.n_pcs,
            },
        ),
        out_path,
    )


# ---------- node 7: markers ----------


def node_markers(cfg: ScrnaConfig, adata_in: Path) -> tuple[NodeResult, Optional[Path]]:
    """Rank genes per Leiden cluster (Wilcoxon) and write a tidy markers.csv.

    Requires:
      - node_cluster to have run first (needs adata.obs['leiden'])
      - node_normalize's adata.raw to hold log-normalized data
        (set after log1p, so the Wilcoxon test sees normalised values)

    markers.csv columns: cluster, gene, score, logfoldchange, pval, pval_adj
    This file is the natural input to the downstream enrichment pipeline.
    """
    import pandas as pd

    out_dir = cfg.out_dir / "06_markers"
    out_h5ad = out_dir / "07_markers.h5ad"
    out_csv = out_dir / "markers.csv"

    try:
        adata = sc.read_h5ad(adata_in)

        if "leiden" not in adata.obs.columns:
            raise ValueError(
                "adata.obs['leiden'] missing — "
                "node_cluster must run before node_markers"
            )

        sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

        # Build tidy DataFrame from Scanpy's structured-array result.
        result = adata.uns["rank_genes_groups"]
        cluster_names = result["names"].dtype.names

        rows = []
        for c in cluster_names:
            n_take = min(cfg.n_top_markers, len(result["names"][c]))
            for i in range(n_take):
                rows.append(
                    {
                        "cluster": c,
                        "gene": result["names"][c][i],
                        "score": float(result["scores"][c][i]),
                        "logfoldchange": float(result["logfoldchanges"][c][i]),
                        "pval": float(result["pvals"][c][i]),
                        "pval_adj": float(result["pvals_adj"][c][i]),
                    }
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        sc.settings.figdir = str(out_dir)
        sc.pl.rank_genes_groups(
            adata, n_genes=10, sharey=False, show=False, save="_markers.png"
        )
        sc.pl.rank_genes_groups_dotplot(
            adata, n_genes=5, show=False, save="_dotplot.png"
        )

        _save_adata(adata, out_h5ad)

    except Exception as exc:
        return (
            NodeResult(
                name="markers",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="markers",
            ok=True,
            message=(
                f"top {cfg.n_top_markers} markers per cluster written "
                f"({len(rows)} total)"
            ),
            outputs={
                "adata": str(out_h5ad),
                "markers_csv": str(out_csv),
                "markers_dir": str(out_dir),
            },
            metrics={
                "n_clusters": len(cluster_names),
                "n_top_markers": cfg.n_top_markers,
                "n_marker_rows": len(rows),
            },
        ),
        out_h5ad,
    )


# ---------- node 8: benchmark ----------


def node_benchmark(
    cfg: ScrnaConfig,
    adata_in: Path,
    node_history: list,
) -> tuple[NodeResult, Optional[Path]]:
    """Compute clustering quality metrics and a per-node timing report.

    Clustering metrics
    ------------------
    Silhouette score (PCA space) — always computed; no ground truth needed.

    ARI / NMI — computed when ground truth labels are available:
      - ``input_kind == 'pbmc3k'``: auto-loads ``sc.datasets.pbmc3k_processed()``
        and aligns by obs_names (barcodes). Reference labels: ``obs['louvain']``.
      - ``cfg.true_labels_col``: reads that column from the current adata.

    Timing report
    -------------
    Reads ``wall_time_s`` from each NodeResult in node_history (set by _timed
    in graph_nodes.py) and writes a sorted table.

    Writes
    ------
    09_benchmark/benchmark_report.json
    09_benchmark/benchmark_report.txt  (human-readable)
    """
    import json

    import numpy as np
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    out_dir = cfg.out_dir / "09_benchmark"
    out_json = out_dir / "benchmark_report.json"
    out_txt = out_dir / "benchmark_report.txt"

    try:
        adata = sc.read_h5ad(adata_in)

        if "leiden" not in adata.obs.columns:
            raise ValueError(
                "adata.obs['leiden'] missing — "
                "node_cluster must run before node_benchmark"
            )
        if "X_pca" not in adata.obsm:
            raise ValueError(
                "adata.obsm['X_pca'] missing — "
                "node_pca must run before node_benchmark"
            )

        pred_labels = adata.obs["leiden"].values

        # ── silhouette (always) ──────────────────────────────────────────────
        sil = float(
            silhouette_score(adata.obsm["X_pca"], pred_labels, metric="euclidean")
        )

        # ── ground truth labels ──────────────────────────────────────────────
        ari: float | None = None
        nmi: float | None = None
        n_ref_clusters: int | None = None
        cells_matched: int | None = None
        ref_source: str = "none"
        ref_labels: np.ndarray | None = None

        if cfg.input_kind == "pbmc3k":
            try:
                ref_adata = sc.datasets.pbmc3k_processed()
                common = adata.obs_names.intersection(ref_adata.obs_names)
                cells_matched = len(common)
                if cells_matched < 100:
                    ref_source = (
                        f"pbmc3k_processed (only {cells_matched} "
                        "cells matched — skipping ARI/NMI)"
                    )
                else:
                    ref_labels = ref_adata[common].obs["louvain"].values
                    pred_sub = adata[common].obs["leiden"].values
                    ari = float(adjusted_rand_score(ref_labels, pred_sub))
                    nmi = float(normalized_mutual_info_score(ref_labels, pred_sub))
                    n_ref_clusters = int(ref_adata.obs["louvain"].nunique())
                    ref_source = (
                        f"pbmc3k_processed louvain " f"({cells_matched} cells matched)"
                    )
            except Exception as exc:
                ref_source = f"pbmc3k_processed load failed: {exc}"

        elif cfg.true_labels_col:
            col = cfg.true_labels_col
            if col in adata.obs.columns:
                ref_labels = adata.obs[col].values
                ari = float(adjusted_rand_score(ref_labels, pred_labels))
                nmi = float(normalized_mutual_info_score(ref_labels, pred_labels))
                n_ref_clusters = int(adata.obs[col].nunique())
                cells_matched = len(adata)
                ref_source = f"adata.obs['{col}'] ({n_ref_clusters} classes)"
            else:
                ref_source = f"column '{col}' not found in adata.obs"

        # ── timing from node_history ─────────────────────────────────────────
        timing: dict[str, float] = {}
        for r in node_history:
            wt = r.metrics.get("wall_time_s")
            if wt is not None:
                timing[r.name] = wt
        total_s = sum(timing.values())

        # ── assemble report ──────────────────────────────────────────────────
        clustering_section: dict = {
            "silhouette_pca": round(sil, 4),
            "n_clusters_found": int(adata.obs["leiden"].nunique()),
            "ari": round(ari, 4) if ari is not None else None,
            "nmi": round(nmi, 4) if nmi is not None else None,
            "n_ref_clusters": n_ref_clusters,
            "cells_matched": cells_matched,
            "reference_source": ref_source,
        }
        report = {
            "clustering": clustering_section,
            "timing_s": {k: v for k, v in sorted(timing.items(), key=lambda x: -x[1])},
            "total_time_s": round(total_s, 3),
        }

        # ── human-readable text ──────────────────────────────────────────────
        lines: list[str] = [
            "=" * 56,
            "  scRNA-seq Benchmark Report",
            "=" * 56,
            "",
            "Clustering quality",
            "-" * 30,
            f"  Leiden clusters found : {clustering_section['n_clusters_found']}",
            f"  Silhouette (PCA)      : {sil:.4f}  "
            f"{'★ good' if sil > 0.25 else '△ moderate' if sil > 0.1 else '✗ poor'}",
        ]
        if ari is not None:
            lines += [
                f"  ARI vs reference      : {ari:.4f}  "
                f"{'★ good' if ari > 0.7 else '△ moderate' if ari > 0.4 else '✗ poor'}",
                f"  NMI vs reference      : {nmi:.4f}",
                f"  Reference             : {ref_source}",
            ]
        else:
            lines.append(f"  Reference             : {ref_source}")

        lines += [
            "",
            "Per-node wall-clock time",
            "-" * 30,
        ]
        for node_name, secs in sorted(timing.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(secs / max(timing.values()) * 20))
            lines.append(f"  {node_name:<14} {bar:<22} {secs:>6.1f}s")
        lines += [
            "-" * 30,
            f"  {'TOTAL':<14} {'':22} {total_s:>6.1f}s",
            "=" * 56,
        ]
        report_txt = "\n".join(lines)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2))
        out_txt.write_text(report_txt)
        print(f"\n{report_txt}\n", flush=True)

    except Exception as exc:
        return (
            NodeResult(
                name="benchmark",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="benchmark",
            ok=True,
            message=(
                f"silhouette={sil:.3f}"
                + (f"  ARI={ari:.3f}  NMI={nmi:.3f}" if ari is not None else "")
                + f"  total_time={total_s:.1f}s"
            ),
            outputs={
                "benchmark_json": str(out_json),
                "benchmark_txt": str(out_txt),
                "benchmark_dir": str(out_dir),
            },
            metrics={
                "silhouette_pca": sil,
                "ari": ari,
                "nmi": nmi,
                "total_time_s": total_s,
            },
        ),
        None,
    )  # benchmark does not write a new adata


# ---------- node 8: trajectory (PAGA) ----------


def node_trajectory(
    cfg: ScrnaConfig, adata_in: Path
) -> tuple[NodeResult, Optional[Path]]:
    """Run PAGA trajectory inference on the Leiden clustering.

    Steps
    -----
    1. sc.tl.paga        — build a cluster-level connectivity graph from the
                           kNN graph computed in node_cluster.
    2. sc.pl.paga        — save a connectivity plot.
    3. sc.tl.umap(init_pos='paga')
                         — re-embed with PAGA-initialised positions; the result
                           is a more topology-faithful 2-D layout than the
                           default random init.
    4. Export paga_graph.json
                         — flat dict with cluster labels and the upper-triangle
                           of the PAGA adjacency matrix, ready for Palantir v2.

    Requires
    --------
    - adata.obs['leiden']           (node_cluster)
    - adata.obsp['connectivities']  (sc.pp.neighbors inside node_cluster)
    """
    import json

    import numpy as np

    out_dir = cfg.out_dir / "07_trajectory"
    out_h5ad = out_dir / "08_trajectory.h5ad"
    out_json = out_dir / "paga_graph.json"

    try:
        adata = sc.read_h5ad(adata_in)

        for req in ("leiden",):
            if req not in adata.obs.columns:
                raise ValueError(
                    f"adata.obs['{req}'] missing — "
                    "node_cluster must run before node_trajectory"
                )
        if "connectivities" not in adata.obsp:
            raise ValueError(
                "adata.obsp['connectivities'] missing — "
                "node_cluster must run before node_trajectory"
            )

        # 1. PAGA — cluster-level graph
        sc.tl.paga(adata, groups="leiden")

        out_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(out_dir)

        # 2. Connectivity plot
        sc.pl.paga(
            adata,
            plot=True,
            show=False,
            save="_connectivity.png",
            title="PAGA cluster connectivity",
        )

        # 3. PAGA-initialised UMAP — overwrites the cluster UMAP for a
        #    topology-faithful embedding; original umap coords are backed up.
        if "X_umap" in adata.obsm:
            adata.obsm["X_umap_cluster"] = adata.obsm["X_umap"].copy()

        sc.tl.umap(adata, init_pos="paga")
        sc.pl.umap(
            adata,
            color="leiden",
            show=False,
            save="_paga_umap.png",
            title="UMAP (PAGA init)",
        )

        # 4. Export graph topology as JSON for downstream tools (Palantir v2)
        clusters = list(adata.obs["leiden"].cat.categories)
        conn = adata.uns["paga"]["connectivities"]
        # scipy sparse → dense upper triangle
        conn_dense = conn.toarray() if hasattr(conn, "toarray") else np.array(conn)
        edges = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                w = float(conn_dense[i, j])
                if w > 0:
                    edges.append(
                        {
                            "source": clusters[i],
                            "target": clusters[j],
                            "weight": round(w, 4),
                        }
                    )

        paga_export = {
            "clusters": clusters,
            "n_clusters": len(clusters),
            "edges": edges,
            "n_edges": len(edges),
            "root_cluster": None,  # user specifies root for Palantir in v2
        }
        out_json.write_text(json.dumps(paga_export, indent=2))

        _save_adata(adata, out_h5ad)

    except Exception as exc:
        return (
            NodeResult(
                name="trajectory",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="trajectory",
            ok=True,
            message=(
                f"PAGA graph: {len(clusters)} clusters, {len(edges)} edges; "
                "UMAP re-embedded with PAGA init"
            ),
            outputs={
                "adata": str(out_h5ad),
                "paga_json": str(out_json),
                "traj_dir": str(out_dir),
            },
            metrics={
                "n_clusters": len(clusters),
                "n_paga_edges": len(edges),
                "paga_umap_reinit": True,
            },
        ),
        out_h5ad,
    )


# ---------- node 9: palantir ----------


def node_palantir(
    cfg: ScrnaConfig,
    adata_in: Path,
    root_cluster: str,
) -> tuple[NodeResult, Optional[Path]]:
    """Run Palantir pseudotime from the PAGA-initialised AnnData.

    The root *cell* is chosen automatically as the PCA-centroid-nearest cell
    in ``root_cluster`` — the biologist only needs to name the cluster.

    Requires
    --------
    - adata.obs['leiden']          (node_cluster)
    - adata.obsm['X_pca']         (node_pca)
    - adata.uns['paga']            (node_trajectory)

    Writes
    ------
    - 08_palantir/pseudotime.csv
    - 08_palantir/branch_probs.csv
    - 08_palantir/palantir_summary.json
    - 08_palantir/09_palantir.h5ad   (obs enriched with pseudotime + branch probs)
    """
    import json

    import numpy as np

    out_dir = cfg.out_dir / "08_palantir"
    out_h5ad = out_dir / "09_palantir.h5ad"
    out_pt = out_dir / "pseudotime.csv"
    out_bp = out_dir / "branch_probs.csv"
    out_json = out_dir / "palantir_summary.json"

    try:
        import palantir  # optional; pip install palantir
    except ImportError:
        return (
            NodeResult(
                name="palantir",
                ok=False,
                message="palantir package not installed — pip install palantir",
            ),
            None,
        )

    try:
        adata = sc.read_h5ad(adata_in)

        if "leiden" not in adata.obs.columns:
            raise ValueError(
                "adata.obs['leiden'] missing — node_cluster must run first"
            )
        if "X_pca" not in adata.obsm:
            raise ValueError("adata.obsm['X_pca'] missing — node_pca must run first")
        if "paga" not in adata.uns:
            raise ValueError(
                "adata.uns['paga'] missing — node_trajectory must run first"
            )

        # --- pick root cell: PCA-centroid-nearest cell in root_cluster ---
        cluster_mask = adata.obs["leiden"] == str(root_cluster)
        if not cluster_mask.any():
            available = sorted(adata.obs["leiden"].unique())
            raise ValueError(
                f"Root cluster '{root_cluster}' not found. " f"Available: {available}"
            )
        cluster_cells = adata.obs.index[cluster_mask]
        pca_sub = adata[cluster_cells].obsm["X_pca"]
        centroid = pca_sub.mean(axis=0)
        dists = np.linalg.norm(pca_sub - centroid, axis=1)
        root_cell = str(cluster_cells[int(dists.argmin())])

        print(
            f"  [palantir] root cluster={root_cluster}, "
            f"root cell={root_cell} (closest to cluster centroid in PCA space)",
            flush=True,
        )

        # --- run Palantir ---
        # Try AnnData-native API (≥1.3), fall back to DataFrame API.
        try:
            pr_res = palantir.core.run_palantir(
                adata,
                early_cell=root_cell,
                num_waypoints=cfg.n_waypoints,
                use_early_cell_as_start=True,
            )
            pseudotime = pr_res.pseudotime
            branch_probs = pr_res.branch_probs
        except TypeError:
            # Older palantir (<1.3): needs magic-imputed DataFrame from diffusion maps
            import pandas as pd

            dm_res = palantir.utils.run_diffusion_maps(
                pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names)
            )
            ms_data = palantir.utils.determine_multiscale_space(dm_res)
            pr_res = palantir.core.run_palantir(
                ms_data, root_cell, num_waypoints=cfg.n_waypoints
            )
            pseudotime = pr_res.pseudotime
            branch_probs = pr_res.branch_probs

        # --- write outputs ---
        out_dir.mkdir(parents=True, exist_ok=True)

        pseudotime.to_csv(out_pt, header=["pseudotime"])
        branch_probs.to_csv(out_bp)

        # Embed results in adata for downstream plotting
        adata.obs["palantir_pseudotime"] = pseudotime
        for col in branch_probs.columns:
            adata.obs[f"palantir_branch_{col}"] = branch_probs[col]

        sc.settings.figdir = str(out_dir)
        sc.pl.umap(
            adata,
            color="palantir_pseudotime",
            show=False,
            save="_pseudotime.png",
            color_map="RdYlBu_r",
            title="Palantir pseudotime",
        )

        n_branches = int(branch_probs.shape[1])
        pt_min = round(float(pseudotime.min()), 4)
        pt_max = round(float(pseudotime.max()), 4)

        summary_data = {
            "root_cluster": root_cluster,
            "root_cell": root_cell,
            "n_waypoints": cfg.n_waypoints,
            "n_branches": n_branches,
            "pseudotime_range": [pt_min, pt_max],
        }
        out_json.write_text(json.dumps(summary_data, indent=2))
        _save_adata(adata, out_h5ad)

    except Exception as exc:
        return (
            NodeResult(
                name="palantir",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="palantir",
            ok=True,
            message=(
                f"Palantir: root cluster={root_cluster}, {n_branches} branch(es), "
                f"pseudotime [{pt_min:.3f}, {pt_max:.3f}]"
            ),
            outputs={
                "adata": str(out_h5ad),
                "pseudotime_csv": str(out_pt),
                "branch_probs_csv": str(out_bp),
                "palantir_json": str(out_json),
                "palantir_dir": str(out_dir),
            },
            metrics={
                "root_cluster": root_cluster,
                "root_cell": root_cell,
                "n_waypoints": cfg.n_waypoints,
                "n_branches": n_branches,
            },
        ),
        out_h5ad,
    )
