"""End-to-end smoke test: bulk DEG  →  ENSP pairs  →  GNN inference  →  SHS27k ground truth.

Picks 30 well-studied human genes (cell cycle / DNA damage / apoptosis), most of
which are in SHS27k and therefore have ground-truth STRING interaction labels.

Pipeline tested:
    synthetic counts (real ENSGs)
        → node_deg                    (PyDESeq2 + mygene gene→ENSP pair builder)
        → predict_pairs               (GNN inference using BFS-trained checkpoint)
        → compare predicted_classes vs SHS27k mode labels for in-graph pairs

Output:
    runs/chain_smoke/
        counts_raw.tsv               6 samples × 30 genes
        06_deg/                      DESeq2 outputs + deg_pairs.tsv
        inference.json               per-pair GNN predictions
        evaluation.json              per-pair comparison vs ground truth

Ground-truth metric reported:
    Among DEG pairs that exist in SHS27k, what fraction of true STRING evidence
    channels did the GNN recover, and what fraction of its predicted channels
    are real?
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# 30 well-studied human genes — DNA damage, cell cycle, apoptosis
# ---------------------------------------------------------------------------

GENES: list[tuple[str, str]] = [
    # 10 will be upregulated in "treated"
    ("TP53",   "ENSG00000141510"),
    ("MDM2",   "ENSG00000135679"),
    ("CDKN1A", "ENSG00000124762"),
    ("BRCA1",  "ENSG00000012048"),
    ("BRCA2",  "ENSG00000139618"),
    ("ATM",    "ENSG00000149311"),
    ("ATR",    "ENSG00000175054"),
    ("CHEK1",  "ENSG00000149554"),
    ("CHEK2",  "ENSG00000183765"),
    ("RAD51",  "ENSG00000051180"),
    # 10 will be downregulated in "treated"
    ("CDK1",   "ENSG00000170312"),
    ("CDK2",   "ENSG00000123374"),
    ("CDK4",   "ENSG00000135446"),
    ("CDK6",   "ENSG00000105810"),
    ("CCND1",  "ENSG00000110092"),
    ("CCNE1",  "ENSG00000105173"),
    ("CCNB1",  "ENSG00000134057"),
    ("E2F1",   "ENSG00000101412"),
    ("MYC",    "ENSG00000136997"),
    ("RB1",    "ENSG00000139687"),
    # 10 neutral controls (NOT differentially expressed)
    ("ACTB",   "ENSG00000075624"),    # beta-actin (housekeeping)
    ("GAPDH",  "ENSG00000111640"),    # glyceraldehyde-3-phosphate dehydrogenase
    ("RPL13A", "ENSG00000142541"),
    ("HPRT1",  "ENSG00000165704"),
    ("PPIA",   "ENSG00000196262"),
    ("TBP",    "ENSG00000112592"),
    ("YWHAZ",  "ENSG00000164924"),
    ("B2M",    "ENSG00000166710"),
    ("HMBS",   "ENSG00000256269"),
    ("UBC",    "ENSG00000150991"),
]
N_DE_UP, N_DE_DOWN = 10, 10
SAMPLES_CTRL    = ["ctrl_1", "ctrl_2", "ctrl_3"]
SAMPLES_TREATED = ["treat_1", "treat_2", "treat_3"]
ALL_SAMPLES     = SAMPLES_CTRL + SAMPLES_TREATED


# ---------------------------------------------------------------------------
# step 1: build counts_raw.tsv
# ---------------------------------------------------------------------------

def build_counts(out_path: Path, seed: int = 42) -> None:
    random.seed(seed)
    de_up   = set(g for _, g in GENES[:N_DE_UP])
    de_down = set(g for _, g in GENES[N_DE_UP : N_DE_UP + N_DE_DOWN])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("\t".join(["gene_id", "length"] + ALL_SAMPLES) + "\n")
        for sym, ensg in GENES:
            length = random.randint(1000, 4000)
            cols: list[str] = [ensg, str(length)]
            for s in ALL_SAMPLES:
                base = max(0, int(random.gauss(400, 80)))
                if not s.startswith("treat_"):
                    cols.append(str(base))
                    continue
                if ensg in de_up:
                    cols.append(str(int(base * random.uniform(8, 14))))
                elif ensg in de_down:
                    cols.append(str(max(0, int(base * random.uniform(0.05, 0.15)))))
                else:
                    cols.append(str(base))
            fh.write("\t".join(cols) + "\n")
    print(f"[1] wrote {out_path}  ({len(GENES)} genes × {len(ALL_SAMPLES)} samples)")


# ---------------------------------------------------------------------------
# step 2: SHS27k ground-truth loader
# ---------------------------------------------------------------------------

def load_shs27k_labels(ppi_path: Path) -> dict[frozenset, set[str]]:
    """Build {frozenset({ensp_a, ensp_b}): set_of_evidence_channels}."""
    labels: dict[frozenset, set[str]] = defaultdict(set)
    with ppi_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            a, b, mode = row["item_id_a"], row["item_id_b"], row["mode"]
            labels[frozenset({a, b})].add(mode)
    print(f"[gt] loaded {len(labels)} unique SHS27k pairs across "
          f"{len(set(m for s in labels.values() for m in s))} channels")
    return dict(labels)


# ---------------------------------------------------------------------------
# step 3+4: run DEG → inference
# ---------------------------------------------------------------------------

def run_deg(out_dir: Path) -> Path:
    """Returns the path to deg_pairs.tsv."""
    from scripts.bulk_rnaseq.config import PreprocessingConfig, Sample
    from scripts.bulk_rnaseq.nodes  import node_deg

    cfg = PreprocessingConfig(
        samples=[
            Sample(name="ctrl_1",  r1=Path("/dev/null"), condition="control"),
            Sample(name="ctrl_2",  r1=Path("/dev/null"), condition="control"),
            Sample(name="ctrl_3",  r1=Path("/dev/null"), condition="control"),
            Sample(name="treat_1", r1=Path("/dev/null"), condition="treated"),
            Sample(name="treat_2", r1=Path("/dev/null"), condition="treated"),
            Sample(name="treat_3", r1=Path("/dev/null"), condition="treated"),
        ],
        gtf=Path("/dev/null"),
        genome_dir=Path("/dev/null"),
        out_dir=out_dir,
        threads=2,
        enable_deg=True,
        reference_condition="control",
        treated_condition="treated",
        padj_threshold=0.05,
        lfc_threshold=1.0,
        build_pairs_for_enrichment=True,
    )
    raw_counts = out_dir / "counts_raw.tsv"
    res = node_deg(cfg, raw_counts)
    print(f"[2] {res.message}")
    if not res.ok or "deg_pairs" not in (res.outputs or {}):
        raise RuntimeError(f"node_deg failed or did not produce deg_pairs.tsv: {res.message}")
    return Path(res.outputs["deg_pairs"])


def run_inference(
    deg_pairs:    Path,
    out_dir:      Path,
    ckpt_path:    Path,
    ppi_path:     Path,
    esm_path:     Path,
    pathway_path: Path,
) -> Path:
    from scripts.enrichment.inference import predict_pairs, read_pairs

    pairs = read_pairs(deg_pairs)
    print(f"[3] read {len(pairs)} candidate pairs from {deg_pairs.name}")

    out_path = out_dir / "inference.json"
    summary = predict_pairs(
        ppi_path=ppi_path,
        esm_path=esm_path,
        pathway_path=pathway_path,
        ckpt_path=ckpt_path,
        pairs=pairs,
        out_path=out_path,
        device="cpu",
        threshold=0.5,
    )
    print(f"[3] inference: predicted={summary['n_predicted']}, "
          f"skipped (not in graph)={summary['n_skipped']}")
    return out_path


# ---------------------------------------------------------------------------
# step 5: compare predicted vs ground truth
# ---------------------------------------------------------------------------

def evaluate(inference_json: Path, ground_truth: dict[frozenset, set[str]],
             out_path: Path) -> dict:
    inf = json.loads(inference_json.read_text())
    in_graph_pairs   = [r for r in inf["results"] if r.get("in_graph_a") and r.get("in_graph_b")]
    pairs_with_gt    = []
    pairs_without_gt = []

    tp = fp = fn = 0
    per_channel_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in in_graph_pairs:
        key = frozenset({r["ensp_a"], r["ensp_b"]})
        gt = ground_truth.get(key)
        predicted = set(r.get("predicted_classes") or [])
        rec = {
            "ensp_a":            r["ensp_a"],
            "ensp_b":            r["ensp_b"],
            "predicted_classes": sorted(predicted),
            "ground_truth":      sorted(gt) if gt else None,
        }
        if gt is None:
            pairs_without_gt.append(rec)
            continue

        rec_tp = predicted & gt
        rec_fp = predicted - gt
        rec_fn = gt - predicted
        rec["tp"] = sorted(rec_tp)
        rec["fp"] = sorted(rec_fp)
        rec["fn"] = sorted(rec_fn)
        tp += len(rec_tp)
        fp += len(rec_fp)
        fn += len(rec_fn)
        for ch in rec_tp: per_channel_stats[ch]["tp"] += 1
        for ch in rec_fp: per_channel_stats[ch]["fp"] += 1
        for ch in rec_fn: per_channel_stats[ch]["fn"] += 1

        pairs_with_gt.append(rec)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    summary = {
        "n_pairs_total":            len(inf["results"]),
        "n_pairs_in_graph":         len(in_graph_pairs),
        "n_pairs_with_ground_truth": len(pairs_with_gt),
        "n_pairs_no_ground_truth":   len(pairs_without_gt),
        "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": round(precision, 4),
        "micro_recall":    round(recall,    4),
        "micro_f1":        round(f1,        4),
        "per_channel": {
            ch: {
                **counts,
                "precision": round(counts["tp"] / (counts["tp"] + counts["fp"]), 4)
                              if (counts["tp"] + counts["fp"]) else 0.0,
                "recall":    round(counts["tp"] / (counts["tp"] + counts["fn"]), 4)
                              if (counts["tp"] + counts["fn"]) else 0.0,
            }
            for ch, counts in per_channel_stats.items()
        },
        "pairs_with_ground_truth":   pairs_with_gt,
        "pairs_no_ground_truth":     pairs_without_gt[:20],   # head only
    }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("runs/chain_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. build synthetic counts
    build_counts(out_dir / "counts_raw.tsv")

    # 2. run DEG node (writes deg_pairs.tsv via mygene)
    deg_pairs = run_deg(out_dir)

    # 3. run inference using the BFS-trained checkpoint
    inf_json = run_inference(
        deg_pairs   = deg_pairs,
        out_dir     = out_dir,
        ckpt_path   = Path("runs/bfs_smoke/gnn_model_valid_best.ckpt"),
        ppi_path    = Path("data/ppi_SHS27k.tsv"),
        esm_path    = Path("data/esm2_embeddings_SHS27k.pt"),
        pathway_path= Path("data/pathway/06_pathway_embeddings_combined.pt"),
    )

    # 4. compare predictions to SHS27k ground truth
    gt = load_shs27k_labels(Path("data/ppi_SHS27k.tsv"))
    summary = evaluate(inf_json, gt, out_dir / "evaluation.json")

    # 5. report
    print()
    print("=" * 70)
    print(" RESULTS — DEG pairs vs SHS27k ground truth")
    print("=" * 70)
    print(f"  total candidate pairs       : {summary['n_pairs_total']}")
    print(f"  in SHS27k graph             : {summary['n_pairs_in_graph']}")
    print(f"  with ground-truth label set : {summary['n_pairs_with_ground_truth']}")
    print(f"  in graph but no GT (zeros)  : {summary['n_pairs_no_ground_truth']}")
    print()
    print(f"  micro Precision : {summary['micro_precision']}")
    print(f"  micro Recall    : {summary['micro_recall']}")
    print(f"  micro F1        : {summary['micro_f1']}")
    print()
    print("  per-channel (over pairs that have ground truth):")
    for ch, s in summary["per_channel"].items():
        print(f"    {ch:11s} tp={s['tp']:>3} fp={s['fp']:>3} fn={s['fn']:>3}  "
              f"P={s['precision']:.3f}  R={s['recall']:.3f}")
    print()
    print(f"  evaluation JSON : {out_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
