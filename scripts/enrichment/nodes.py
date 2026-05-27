"""Pipeline nodes for the enrichment (GNN-PPI) workflow.

Each function runs one stage and returns a NodeResult. They wrap the
underlying training / evaluation / inference functions in
``gnn_train.py``, ``gnn_test.py`` and ``inference.py`` and translate
their outputs into the structured NodeResult format the LangGraph layer
expects.

Each node rebuilds the GNNDataset from disk (cheap: ~3 sec for SHS27k)
so the LangGraph state never has to hold a PyG Data object. This is the
same disk-passing pattern the scRNA-seq pipeline uses for AnnData.
"""

from __future__ import annotations

from pathlib import Path

from scripts.common.node_result import NodeResult
from scripts.enrichment.config import EnrichmentConfig

# ---------- node 1: load_data ----------


def node_load_data(cfg: EnrichmentConfig) -> NodeResult:
    """Build the GNNDataset once to validate inputs and return summary stats.

    We don't persist the dataset between nodes; each subsequent node rebuilds
    it from cfg paths. The point of this node is to fail fast if any of the
    PPI/ESM/pathway files is missing or malformed, before kicking off training.
    """
    try:
        # lazy import: heavy torch/torch_geometric load
        from scripts.enrichment.gnn_data import GNNDataset

        dataset = GNNDataset(
            ppi_path=cfg.ppi_path,
            esm_emb_path=cfg.esm_path,
            pathway_emb_path=cfg.pathway_path,
            verbose=False,
        )
        n_proteins = dataset.num_nodes
        n_edges_uni = dataset.num_unique_edges
        n_components = dataset.count_components()
        esm_dim = dataset.esm_dim
        pathway_dim = dataset.pathway_x.shape[1] if dataset.pathway_x is not None else 0

        if dataset.pathway_x is not None and pathway_dim > 0:
            row_norms = dataset.pathway_x.norm(dim=1)
            pathway_coverage = float((row_norms > 0).float().mean().item())
        else:
            pathway_coverage = 0.0

    except Exception as exc:
        return NodeResult(
            name="load_data",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    return NodeResult(
        name="load_data",
        ok=True,
        message=(
            f"loaded GNNDataset: {n_proteins} proteins, {n_edges_uni} edges, "
            f"{n_components} components, esm_dim={esm_dim}, pathway_dim={pathway_dim}, "
            f"pathway_coverage={pathway_coverage:.3f}"
        ),
        outputs={
            "ppi_path": str(cfg.ppi_path),
            "esm_path": str(cfg.esm_path),
            "pathway_path": str(cfg.pathway_path) if cfg.pathway_path else "",
        },
        metrics={
            "n_proteins": n_proteins,
            "n_edges": n_edges_uni,
            "n_components": n_components,
            "esm_dim": esm_dim,
            "pathway_dim": pathway_dim,
            "pathway_coverage": pathway_coverage,
        },
    )


# ---------- node 2: train ----------


def node_train(cfg: EnrichmentConfig) -> NodeResult:
    """Train GIN_PPI on the configured PPI dataset.

    Writes ``gnn_model_valid_best.ckpt`` + ``gnn_model_train_last.ckpt`` +
    ``training_history.json`` under ``cfg.out_dir``.
    """
    try:
        from scripts.enrichment.gnn_data import GNNDataset
        from scripts.enrichment.gnn_train import train as run_train

        dataset = GNNDataset(
            ppi_path=cfg.ppi_path,
            esm_emb_path=cfg.esm_path,
            pathway_emb_path=cfg.pathway_path,
            verbose=False,
        )

        result = run_train(
            dataset=dataset,
            out_dir=cfg.out_dir,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
            split_method=cfg.split_method,
            test_size=cfg.test_size,
            seed=cfg.seed,
            use_tensorboard=cfg.use_tensorboard,
            early_stop_patience=cfg.early_stop,
        )
    except Exception as exc:
        return NodeResult(
            name="train",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    return NodeResult(
        name="train",
        ok=True,
        message=(
            f"trained {cfg.epochs} epochs (best at {result.best_epoch}), "
            f"val_f1={result.best_f1:.4f}, n_params={result.n_params}"
        ),
        outputs={
            "best_ckpt": str(result.best_ckpt_path),
            "last_ckpt": str(result.last_ckpt_path),
            "history": str(result.history_path),
        },
        metrics={
            "best_val_f1": result.best_f1,
            "best_epoch": result.best_epoch,
            "n_train": result.n_train,
            "n_val": result.n_val,
            "n_params": result.n_params,
            "device": result.device,
        },
    )


# ---------- node 3: eval ----------


def node_eval(cfg: EnrichmentConfig, ckpt_path: Path) -> NodeResult:
    """Bucketed evaluation (test1/2/3) of the trained checkpoint."""
    try:
        from scripts.enrichment.gnn_test import evaluate_checkpoint

        out_path = cfg.out_dir / "test_metrics.json"
        summary = evaluate_checkpoint(
            ppi_path=cfg.ppi_path,
            esm_path=cfg.esm_path,
            pathway_path=cfg.pathway_path,
            ckpt_path=ckpt_path,
            out_path=out_path,
            batch_size=cfg.eval_batch_size,
            device=cfg.device,
            split_method=cfg.split_method,
            test_size=cfg.test_size,
            seed=cfg.seed,
        )
    except Exception as exc:
        return NodeResult(
            name="eval",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    buckets = summary["buckets"]
    all_metrics = buckets["all"]
    return NodeResult(
        name="eval",
        ok=True,
        message=(
            f"eval F1 by bucket — "
            f"all={all_metrics['f1']:.3f} (n={all_metrics['n_edges']}), "
            f"test1={buckets['test1']['f1']:.3f} (n={buckets['test1']['n_edges']}), "
            f"test2={buckets['test2']['f1']:.3f} (n={buckets['test2']['n_edges']}), "
            f"test3={buckets['test3']['f1']:.3f} (n={buckets['test3']['n_edges']})"
        ),
        outputs={"metrics_path": str(out_path)},
        metrics={
            "all_f1": all_metrics["f1"],
            "test1_f1": buckets["test1"]["f1"],
            "test2_f1": buckets["test2"]["f1"],
            "test3_f1": buckets["test3"]["f1"],
            "n_train": summary["n_train"],
            "n_val": summary["n_val"],
            "buckets": buckets,
        },
    )


# ---------- node 4: infer ----------


def node_infer(cfg: EnrichmentConfig, ckpt_path: Path) -> NodeResult:
    """Predict interaction types for novel candidate ENSP pairs.

    No-op (returns ok=True with an explanatory message) when ``cfg.pairs_path``
    is None — i.e. the user opted out at config time.
    """
    if cfg.pairs_path is None:
        return NodeResult(
            name="infer",
            ok=True,
            message="skipped: no candidate pairs file configured",
            metrics={"skipped": True},
        )

    try:
        from scripts.enrichment.inference import predict_pairs, read_pairs

        pairs = read_pairs(cfg.pairs_path)
        out_path = cfg.out_dir / "inference.json"
        summary = predict_pairs(
            ppi_path=cfg.ppi_path,
            esm_path=cfg.esm_path,
            pathway_path=cfg.pathway_path,
            ckpt_path=ckpt_path,
            pairs=pairs,
            out_path=out_path,
            device=cfg.device,
            threshold=cfg.inference_threshold,
        )
    except Exception as exc:
        return NodeResult(
            name="infer",
            ok=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    return NodeResult(
        name="infer",
        ok=True,
        message=(
            f"predicted {summary['n_predicted']}/{summary['n_pairs']} pairs "
            f"({summary['n_skipped']} skipped not in graph)"
        ),
        outputs={"inference_path": str(out_path)},
        metrics={
            "n_pairs_in": summary["n_pairs"],
            "n_predicted": summary["n_predicted"],
            "n_skipped": summary["n_skipped"],
            "threshold": summary["threshold"],
        },
    )
