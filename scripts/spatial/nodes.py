"""Pipeline nodes for the imaging-based spatial workflow (Vizgen v1).

Each function runs one step and returns a NodeResult plus (when applicable)
the path to the AnnData snapshot it wrote.

Output layout under cfg.out_dir/:
    00_load/01_loaded.h5ad
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from scripts.common.node_result import NodeResult
from scripts.spatial.config import SpatialConfig

# ---------- internal helpers ----------


def _save_adata(adata, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)


# ---------- node 1: load_data (squidpy Vizgen reader) ----------


def node_load_vizgen(cfg: SpatialConfig) -> tuple[NodeResult, Optional[Path]]:
    """Read MERSCOPE/Vizgen cell tables into AnnData via squidpy.

    squidpy.read.vizgen() reads counts_file + meta_file and returns an
    AnnData where:
      - X                       = cell x gene counts (real genes only)
      - obsm['spatial']         = cell centroids in microns (the spatial key)
      - obsm['blank_genes']     = the Blank-* negative-control barcodes
      - obs                     = per-cell metadata (volume, fov, centroids)

    Saves 00_load/01_loaded.h5ad and returns (NodeResult, path).
    """
    out_path = cfg.out_dir / "00_load" / "01_loaded.h5ad"

    try:
        # Lazy import: squidpy pulls in dask/spatialdata which are heavy and
        # have import-time syscalls; keep this module importable without them.
        import squidpy as sq

        adata = sq.read.vizgen(
            path=str(cfg.input_dir),
            counts_file=cfg.counts_file,
            meta_file=cfg.meta_file,
            transformation_file=cfg.transformation_file,
            library_id=cfg.library_id,
        )

        # Duplicate gene symbols break downstream scanpy ops — de-dup up front.
        adata.var_names_make_unique()

        # --- validate the things the rest of the pipeline depends on ---
        if adata.n_obs == 0:
            raise ValueError("loaded AnnData has 0 cells")
        if adata.n_vars == 0:
            raise ValueError("loaded AnnData has 0 genes")
        if "spatial" not in adata.obsm:
            raise ValueError("obsm['spatial'] missing — metadata file lacks centroids")

        n_blank = (
            int(adata.obsm["blank_genes"].shape[1])
            if "blank_genes" in adata.obsm
            else 0
        )
        has_volume = "volume" in adata.obs.columns
        n_fov = int(adata.obs["fov"].nunique()) if "fov" in adata.obs.columns else 0

        _save_adata(adata, out_path)

    except Exception as exc:
        return (
            NodeResult(
                name="load_vizgen",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        NodeResult(
            name="load_vizgen",
            ok=True,
            message=(
                f"loaded {adata.n_obs} cells x {adata.n_vars} genes"
                f" ({n_blank} blank barcodes, {n_fov} FOVs)"
            ),
            outputs={"adata": str(out_path)},
            metrics={
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_blank_genes": n_blank,
                "n_fov": n_fov,
                "has_volume": has_volume,
                "has_spatial": True,
            },
        ),
        out_path,
    )


# ---------- node 2: spatial QC (blank-FDR, per-cell, per-FOV) ----------


def _blank_fdr(adata) -> tuple[Optional[float], Optional[float], int]:
    """MERSCOPE misidentification rate from the Blank-* negative controls.

    The Blank barcodes are codewords with no targeting probe, so any counts
    on them are false detections. Comparing the mean counts-per-blank against
    the mean counts-per-real-gene estimates the per-gene false-detection rate.

    Returns (misid_rate, frac_blank_counts, n_blank_barcodes).
    """
    import numpy as np

    if "blank_genes" not in adata.obsm:
        return None, None, 0
    blanks = np.asarray(adata.obsm["blank_genes"])
    n_blank = int(blanks.shape[1])
    if n_blank == 0:
        return None, None, 0
    total_blank = float(blanks.sum())
    total_real = float(np.asarray(adata.X.sum()))
    mean_per_blank = total_blank / n_blank
    mean_per_real = total_real / max(adata.n_vars, 1)
    misid = mean_per_blank / (mean_per_real + 1e-9)
    frac_blank = total_blank / (total_blank + total_real + 1e-9)
    return float(misid), float(frac_blank), n_blank


def node_spatial_qc(
    cfg: SpatialConfig,
    adata_in: Path,
    fov_key: str = "fov",
) -> tuple[NodeResult, Optional[Path]]:
    """Compute QC metrics, filter low-quality cells, flag outlier FOVs.

    Cell-level filtering (counts / genes / volume) is applied because it is
    standard and safe. FOV-level outliers are *flagged* (obs['fov_flagged'])
    but not dropped — losing a whole imaging tile is a bigger call left to the
    user / downstream agentic step.

    Writes 01_qc/qc.h5ad with per-cell QC columns and per-FOV summary in
    adata.uns['fov_qc'].
    """
    out_path = cfg.out_dir / "01_qc" / "qc.h5ad"

    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scanpy as sc

        adata = ad.read_h5ad(adata_in)
        n_before = int(adata.n_obs)

        # --- dataset-level: blank false-detection rate ---
        misid, frac_blank, n_blank = _blank_fdr(adata)

        # --- per-cell QC metrics ---
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        total_counts = adata.obs["total_counts"].values
        n_genes = adata.obs["n_genes_by_counts"].values

        keep = (total_counts >= cfg.min_counts_per_cell) & (
            n_genes >= cfg.min_genes_per_cell
        )

        # --- volume outliers (percentile-based; robust to unit choice) ---
        vol_low = vol_high = None
        if "volume" in adata.obs.columns:
            vol = adata.obs["volume"].values.astype(float)
            vol_low = float(np.percentile(vol, cfg.volume_pct_low))
            vol_high = float(np.percentile(vol, cfg.volume_pct_high))
            keep &= (vol >= vol_low) & (vol <= vol_high)

        adata.obs["qc_pass"] = keep

        # --- per-FOV summary + outlier flagging (non-destructive) ---
        flagged_fovs: list[str] = []
        if fov_key in adata.obs.columns:
            df = pd.DataFrame(
                {
                    "fov": adata.obs[fov_key].astype(str).values,
                    "total_counts": total_counts,
                    "n_genes": n_genes,
                }
            )
            fov_stats = df.groupby("fov").agg(
                n_cells=("total_counts", "size"),
                median_counts=("total_counts", "median"),
                median_genes=("n_genes", "median"),
            )
            global_median = float(df["total_counts"].median())
            cutoff = cfg.fov_min_median_ratio * global_median
            fov_stats["flagged"] = fov_stats["median_counts"] < cutoff
            flagged_fovs = fov_stats.index[fov_stats["flagged"]].tolist()
            adata.obs["fov_flagged"] = adata.obs[fov_key].astype(str).isin(flagged_fovs)
            adata.uns["fov_qc"] = fov_stats.reset_index().to_dict("list")

        # --- apply cell-level filter ---
        adata = adata[keep].copy()
        n_after = int(adata.n_obs)

        _save_adata(adata, out_path)

    except Exception as exc:
        return (
            NodeResult(
                name="spatial_qc",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    dropped = n_before - n_after
    msg = (
        f"kept {n_after}/{n_before} cells "
        f"(dropped {dropped}, {dropped / max(n_before, 1):.1%})"
    )
    if misid is not None:
        warn = " [HIGH]" if misid > cfg.max_blank_fdr else ""
        msg += f" | blank misID rate {misid:.3f}{warn}"
    if flagged_fovs:
        msg += f" | {len(flagged_fovs)} low-signal FOV(s) flagged"

    metrics = {
        "n_cells_before": n_before,
        "n_cells_after": n_after,
        "n_dropped": dropped,
        "frac_dropped": round(dropped / max(n_before, 1), 4),
        "blank_misid_rate": round(misid, 4) if misid is not None else None,
        "frac_blank_counts": (round(frac_blank, 4) if frac_blank is not None else None),
        "n_blank_barcodes": n_blank,
        "blank_fdr_high": (
            bool(misid > cfg.max_blank_fdr) if misid is not None else None
        ),
        "n_fov_flagged": len(flagged_fovs),
        "flagged_fovs": flagged_fovs,
        "volume_low_cut": vol_low,
        "volume_high_cut": vol_high,
    }

    return (
        NodeResult(
            name="spatial_qc",
            ok=True,
            message=msg,
            outputs={"adata": str(out_path)},
            metrics=metrics,
        ),
        out_path,
    )


# ---------- clustering helper (shared by cluster + fov-bias nodes) ----------


def cluster_adata(
    adata,
    use_rep: str = "X_pca",
    n_pcs: int = 50,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    cluster_key: str = "leiden",
    recompute_pca: bool = True,
) -> None:
    """PCA → neighbors → Leiden, in place. Writes obs[cluster_key]."""
    import scanpy as sc

    if recompute_pca or "X_pca" not in adata.obsm:
        n_comps = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
        sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=cluster_key,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )


# ---------- FOV-bias evidence computation (deterministic) ----------


def _batch_correct_fov(adata, fov_key: str, out_key: str = "X_pca_harmony"):
    """Batch-correct the PCA embedding on FOV, writing obsm[out_key].

    Calls harmonypy directly (scanpy's wrapper mis-transposes Z_corr under
    harmonypy>=2.0), with an orientation guard. Falls back to simple per-FOV
    mean-centering in PCA space if Harmony is unavailable or fails.
    """
    import numpy as np

    emb = np.asarray(adata.obsm["X_pca"])
    try:
        import harmonypy

        ho = harmonypy.run_harmony(emb, adata.obs, [fov_key])
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != adata.n_obs:  # harmonypy returns (n_pcs, n_cells)
            Z = Z.T
        if Z.shape != emb.shape:
            raise ValueError(f"unexpected Harmony output shape {Z.shape}")
        adata.obsm[out_key] = Z
        return "harmony"
    except Exception:
        # Fallback: remove the additive per-FOV offset in PCA space.
        corrected = emb.copy()
        fovs = adata.obs[fov_key].astype(str).values
        for f in np.unique(fovs):
            m = fovs == f
            corrected[m] -= corrected[m].mean(axis=0, keepdims=True)
        corrected += emb.mean(axis=0, keepdims=True)
        adata.obsm[out_key] = corrected
        return "fov_centering"


def _inverse_simpson_ilisi(emb, fov_codes, k: int = 30):
    """Per-cell effective number of FOVs among k expression-space neighbors.

    iLISI ≈ 1 → all neighbors from one FOV (segregated / biased).
    iLISI ≈ n_fov → perfectly mixed.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    k = min(k, emb.shape[0])
    nn = NearestNeighbors(n_neighbors=k).fit(emb)
    _, idx = nn.kneighbors(emb)
    out = np.empty(emb.shape[0], dtype=float)
    for i, row in enumerate(idx):
        _, counts = np.unique(fov_codes[row], return_counts=True)
        p = counts / counts.sum()
        out[i] = 1.0 / float(np.sum(p * p))
    return out


def _cramers_v(clusters, fovs) -> float:
    """Bias-corrected Cramér's V for cluster↔FOV association (0..1)."""
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency

    tab = pd.crosstab(pd.Series(clusters), pd.Series(fovs)).values
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rr = r - (r - 1) ** 2 / (n - 1)
    kk = k - (k - 1) ** 2 / (n - 1)
    denom = min(kk - 1, rr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def compute_fov_bias_evidence(
    adata,
    cluster_key: str = "leiden",
    fov_key: str = "fov",
    embed_key: str = "X_pca",
    purity_thresh: float = 0.70,
    sim_threshold: float = 0.85,
    max_pairs: int = 6,
) -> dict:
    """Assemble the deterministic evidence packet for FOV-bias adjudication.

    The robust signals are GLOBAL: do expression-space neighbors mix across
    FOVs (iLISI), how strongly is cluster identity explained by FOV
    (Cramér's V), and what fraction of cells sit in FOV-pure clusters. Per-
    pair details (near-identical profiles split by FOV) are kept as
    supplementary fingerprints for the "one type split in two" case.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    emb = adata.obsm[embed_key]
    clusters = adata.obs[cluster_key].astype(str).values
    fovs = adata.obs[fov_key].astype(str).values
    uniq = sorted(set(clusters))
    n_fov = len(set(fovs))
    n_cells = len(clusters)

    # --- global mixing + association ---
    fov_codes = pd.Categorical(fovs).codes
    ilisi = _inverse_simpson_ilisi(emb, fov_codes)
    median_ilisi = float(np.median(ilisi))
    frac_low = float(np.mean(ilisi < 1.5))
    cramers_v = _cramers_v(clusters, fovs)

    X = adata.X

    def _mean_expr(mask):
        sub = X[mask]
        m = sub.mean(0)
        return np.asarray(m).ravel()

    def _median_libsize(mask):
        sub = X[mask]
        tot = np.asarray(sub.sum(1)).ravel() if sp.issparse(sub) else sub.sum(1)
        return float(np.median(tot))

    def _dom_fov(mask):
        vals, counts = np.unique(fovs[mask], return_counts=True)
        j = int(counts.argmax())
        return str(vals[j]), float(counts[j] / counts.sum())

    centroids = {c: emb[clusters == c].mean(0) for c in uniq}
    profiles = {c: _mean_expr(clusters == c) for c in uniq}
    libsizes = {c: _median_libsize(clusters == c) for c in uniq}

    # --- FOV-pure clusters (the batch fingerprint) ---
    fov_pure = []
    n_pure_cells = 0
    for c in uniq:
        mask = clusters == c
        dom_fov, frac = _dom_fov(mask)
        size = int(mask.sum())
        if frac >= purity_thresh:
            n_pure_cells += size
            fov_pure.append(
                {
                    "cluster": c,
                    "dominant_fov": dom_fov,
                    "dominant_frac": round(frac, 3),
                    "n_cells": size,
                }
            )
    frac_cells_fov_pure = float(n_pure_cells / n_cells) if n_cells else 0.0

    # --- supplementary: near-identical profiles split across FOVs ---
    def _cos(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    pairs = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            ca, cb = uniq[i], uniq[j]
            sim = _cos(centroids[ca], centroids[cb])
            if sim < sim_threshold:
                continue
            pa, pb = profiles[ca], profiles[cb]
            corr = float(np.corrcoef(pa, pb)[0, 1])
            la, lb = libsizes[ca], libsizes[cb]
            ratio = (max(la, lb) / min(la, lb)) if min(la, lb) > 0 else 1.0
            a_fov, a_frac = _dom_fov(clusters == ca)
            b_fov, b_frac = _dom_fov(clusters == cb)
            fov_disjoint = a_fov != b_fov and a_frac > 0.5 and b_frac > 0.5
            lfc = np.log2((pa + 1e-6) / (pb + 1e-6))
            n_de = int(np.sum(np.abs(lfc) > 1.0))
            de_type = "magnitude_shift" if (corr > 0.95 and ratio > 1.3) else "markers"
            pairs.append(
                {
                    "cluster_a": ca,
                    "cluster_b": cb,
                    "centroid_similarity": round(sim, 3),
                    "profile_corr": round(corr, 3),
                    "libsize_ratio": round(ratio, 3),
                    "a_dominant_fov": a_fov,
                    "a_dominant_frac": round(a_frac, 3),
                    "b_dominant_fov": b_fov,
                    "b_dominant_frac": round(b_frac, 3),
                    "fov_disjoint": bool(fov_disjoint),
                    "de_type": de_type,
                    "n_de_genes": n_de,
                }
            )

    pairs.sort(key=lambda p: -p["centroid_similarity"])
    return {
        "n_clusters": len(uniq),
        "n_fov": n_fov,
        "median_ilisi": median_ilisi,
        "frac_low_ilisi": frac_low,
        "cramers_v": round(cramers_v, 3),
        "frac_cells_fov_pure": round(frac_cells_fov_pure, 3),
        "n_fov_pure_clusters": len(fov_pure),
        "fov_pure_clusters": fov_pure,
        "suspicious_pairs": pairs[:max_pairs],
    }


# ---------- node: FOV-bias check (diagnose → adjudicate → correct loop) ----


def node_fov_bias_check(
    cfg: SpatialConfig,
    adata_in: Path,
    cluster_key: str = "leiden",
    fov_key: str = "fov",
) -> tuple[NodeResult, Optional[Path]]:
    """Diagnose FOV-driven cluster splitting; if confirmed, batch-correct on
    FOV and re-cluster, then verify the split actually collapsed.

    Expects an AnnData that already has obs[cluster_key], obs[fov_key], and
    obsm['X_pca'] (i.e. has been clustered upstream).
    """
    out_path = cfg.out_dir / "07_fov_bias" / "fov_checked.h5ad"

    try:
        import anndata as ad

        from scripts.spatial.graph_nodes import llm_fov_bias_adjudicate

        adata = ad.read_h5ad(adata_in)

        if cluster_key not in adata.obs or fov_key not in adata.obs:
            raise ValueError(
                f"need obs['{cluster_key}'] and obs['{fov_key}']"
                " — run clustering first"
            )

        before = compute_fov_bias_evidence(
            adata, cluster_key=cluster_key, fov_key=fov_key
        )
        verdict = llm_fov_bias_adjudicate(before)
        print(
            f"  [fov-bias] {verdict['decision']} / {verdict['action']}"
            f" ({verdict['source']}): {verdict['reason']}",
            flush=True,
        )

        corrected = False
        correction_method = None
        after = None
        if verdict["action"] == "correct_and_recluster":
            # Batch-correct on FOV → re-cluster on the corrected embedding
            correction_method = _batch_correct_fov(adata, fov_key)
            cluster_adata(
                adata,
                use_rep="X_pca_harmony",
                recompute_pca=False,
                cluster_key=cluster_key,
            )
            corrected = True
            after = compute_fov_bias_evidence(
                adata,
                cluster_key=cluster_key,
                fov_key=fov_key,
                embed_key="X_pca_harmony",
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_path)

    except Exception as exc:
        return (
            NodeResult(
                name="fov_bias_check",
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    metrics = {
        "decision": verdict["decision"],
        "action": verdict["action"],
        "source": verdict["source"],
        "n_clusters_before": before["n_clusters"],
        "median_ilisi_before": round(before["median_ilisi"], 3),
        "n_suspicious_before": len(before["suspicious_pairs"]),
        "corrected": corrected,
        "correction_method": correction_method,
    }
    msg = f"{verdict['decision']} ({verdict['source']}): {verdict['reason']}"
    if corrected and after is not None:
        metrics["n_clusters_after"] = after["n_clusters"]
        metrics["median_ilisi_after"] = round(after["median_ilisi"], 3)
        metrics["n_suspicious_after"] = len(after["suspicious_pairs"])
        collapsed = before["n_clusters"] - after["n_clusters"]
        msg += (
            f" | corrected: {before['n_clusters']}→{after['n_clusters']} "
            f"clusters ({collapsed} collapsed), "
            f"iLISI {before['median_ilisi']:.2f}→{after['median_ilisi']:.2f}"
        )

    return (
        NodeResult(
            name="fov_bias_check",
            ok=True,
            message=msg,
            outputs={"adata": str(out_path)},
            metrics=metrics,
        ),
        out_path,
    )
