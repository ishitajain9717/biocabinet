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
        return NodeResult(
            name="load_data",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        ), None

    return NodeResult(
        name="load_data",
        ok=True,
        message=f"loaded {adata.n_obs} cells x {adata.n_vars} genes ({cfg.input_kind})",
        outputs={"adata": str(out_path)},
        metrics={"n_cells": adata.n_obs, "n_genes": adata.n_vars},
    ), out_path


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
            jitter=0.4, multi_panel=True, show=False, save="_qc.png",
        )
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt",
                      show=False, save="_mt.png")
        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts",
                      show=False, save="_genes.png")

        _save_adata(adata, out_h5ad)

    except Exception as exc:
        return NodeResult(
            name="qc",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        ), None

    return NodeResult(
        name="qc",
        ok=True,
        message=(
            f"QC done: median {adata.obs['n_genes_by_counts'].median():.0f} genes/cell, "
            f"median {adata.obs['pct_counts_mt'].median():.2f}% mt"
        ),
        outputs={"adata": str(out_h5ad), "qc_dir": str(out_dir)},
        metrics={
            "median_n_genes":      float(adata.obs["n_genes_by_counts"].median()),
            "median_pct_mt":       float(adata.obs["pct_counts_mt"].median()),
            "median_total_counts": float(adata.obs["total_counts"].median()),
            "n_cells":             int(adata.n_obs),
            "n_genes":             int(adata.n_vars),
        },
    ), out_h5ad


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
        return NodeResult(
            name="filter",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        ), None

    return NodeResult(
        name="filter",
        ok=True,
        message=f"kept {adata.n_obs}/{n_in_cells} cells, {adata.n_vars}/{n_in_genes} genes",
        outputs={"adata": str(out_path)},
        metrics={
            "n_cells_in":         n_in_cells,
            "n_cells_kept":       int(adata.n_obs),
            "n_genes_in":         n_in_genes,
            "n_genes_kept":       int(adata.n_vars),
            "min_genes_per_cell": cfg.min_genes_per_cell,
            "min_cells_per_gene": cfg.min_cells_per_gene,
            "max_pct_mt":         cfg.max_pct_mt,
        },
    ), out_path


# ---------- node 4: normalize ----------

def node_normalize(cfg: ScrnaConfig, adata_in: Path) -> tuple[NodeResult, Optional[Path]]:
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
        return NodeResult(
            name="normalize",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        ), None

    return NodeResult(
        name="normalize",
        ok=True,
        message=f"normalized; flagged {n_hvg} highly-variable genes",
        outputs={"adata": str(out_path), "norm_dir": str(out_dir)},
        metrics={
            "n_cells":    int(adata.n_obs),
            "n_genes":    int(adata.n_vars),
            "n_hvg":      n_hvg,
            "target_sum": cfg.target_sum,
        },
    ), out_path


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
        return NodeResult(
            name="pca",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        ), None

    var_ratio = adata.uns["pca"]["variance_ratio"]
    cum_var_top10 = float(var_ratio[:10].sum())

    return NodeResult(
        name="pca",
        ok=True,
        message=f"PCA computed: {cfg.n_pcs} PCs, top 10 explain {cum_var_top10:.1%} variance",
        outputs={"adata": str(out_path), "pca_dir": str(out_dir)},
        metrics={
            "n_pcs":              cfg.n_pcs,
            "cum_variance_top10": cum_var_top10,
            "scale_clip":         cfg.scale_clip,
            "scale_max_value":    cfg.scale_max_value if cfg.scale_clip else None,
        },
    ), out_path


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
            show=False, save="_qc.png",
        )

        _save_adata(adata, out_path)

    except Exception as exc:
        return NodeResult(
            name="cluster",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        ), None

    return NodeResult(
        name="cluster",
        ok=True,
        message=f"clustered into {n_clusters} clusters (leiden res={cfg.leiden_resolution})",
        outputs={"adata": str(out_path), "cluster_dir": str(out_dir)},
        metrics={
            "n_clusters":        n_clusters,
            "n_neighbors":       cfg.n_neighbors,
            "leiden_resolution": cfg.leiden_resolution,
            "n_pcs_used":        cfg.n_pcs,
        },
    ), out_path


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
                "adata.obs['leiden'] missing — node_cluster must run before node_markers"
            )

        sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

        # Build tidy DataFrame from Scanpy's structured-array result.
        result = adata.uns["rank_genes_groups"]
        cluster_names = result["names"].dtype.names

        rows = []
        for c in cluster_names:
            n_take = min(cfg.n_top_markers, len(result["names"][c]))
            for i in range(n_take):
                rows.append({
                    "cluster":       c,
                    "gene":          result["names"][c][i],
                    "score":         float(result["scores"][c][i]),
                    "logfoldchange": float(result["logfoldchanges"][c][i]),
                    "pval":          float(result["pvals"][c][i]),
                    "pval_adj":      float(result["pvals_adj"][c][i]),
                })

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
        return NodeResult(
            name="markers",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        ), None

    return NodeResult(
        name="markers",
        ok=True,
        message=f"top {cfg.n_top_markers} markers per cluster written ({len(rows)} total)",
        outputs={
            "adata":       str(out_h5ad),
            "markers_csv": str(out_csv),
            "markers_dir": str(out_dir),
        },
        metrics={
            "n_clusters":     len(cluster_names),
            "n_top_markers":  cfg.n_top_markers,
            "n_marker_rows":  len(rows),
        },
    ), out_h5ad
