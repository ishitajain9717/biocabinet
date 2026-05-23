"""Phase 1 of the RAG layer: build a searchable library of pathway documents.

Reads pathway text already on disk from the enrichment pipeline and produces
a uniform, embedding-aligned document store usable by later RAG phases.

INPUTS (already on disk from earlier enrichment work):
    data/pathway/01_gene_to_pathway_ids.json        KEGG  : Entrez gene → [pathway_id]
    data/pathway/02_pathway_id_to_name.json         KEGG  : pathway_id → name
    data/pathway/04_pathway_to_description.json     KEGG  : pathway_id → text
    data/pathway/reactome/R1_gene_to_pathways.json  Reactome: ENSG → [[id, name]]
    data/pathway/reactome/R2_summations/*.json      Reactome: pathway_id.json → {"text": "..."}

OUTPUTS (one library, written to data/rag/):
    docs.jsonl              one JSON document per line:
                              {id, source, pathway_id, pathway_name, text,
                               gene_ids, gene_id_type, n_genes, n_chars}
    embeddings.npy          float32 array of shape (N, 768), aligned to docs.jsonl
    id_map.json             {row_idx_str: doc_id} so we can map embeddings → docs
    gene_to_doc_ids.json    reverse index {gene_id: [doc_id, ...]} for fast filtering
                            (contains BOTH Entrez and ENSG keys)
    build_log.json          counts, BioBERT model name, paths, timestamp

USAGE:
    python -m scripts.rag.build_index
    python -m scripts.rag.build_index --limit 50            # quick smoke build
    python -m scripts.rag.build_index --batch-size 16       # control GPU/CPU memory
    python -m scripts.rag.build_index --skip-embed          # docs only, no encoding

DESIGN NOTES:
    * KEGG and Reactome use different gene identifier systems (Entrez vs ENSG).
      We keep both as-is and tag each doc with `gene_id_type`. The reverse index
      has both kinds of keys so a gene-filter query just works regardless of input.
    * BioBERT is the same model used by the enrichment pathway embedder, so
      retrieval queries embedded with the same tokenizer/model live in the
      same vector space as the docs (consistent cosine similarity).
    * Embeddings use mean-pooling over the last_hidden_state (mask-aware),
      identical to the strategy used by `merge_pathway_embeddings.py`.
    * Idempotent: re-running overwrites the output directory atomically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DEFAULT_BIOBERT = "dmis-lab/biobert-v1.1"
DEFAULT_MAX_LEN = 256


@dataclass
class BuildConfig:
    pathway_dir:    Path = Path("data/pathway")
    out_dir:        Path = Path("data/rag")
    biobert_model:  str  = DEFAULT_BIOBERT
    max_seq_length: int  = DEFAULT_MAX_LEN
    batch_size:     int  = 32
    limit:          int | None = None
    skip_embed:     bool = False
    device:         str  = "cpu"


# ---------------------------------------------------------------------------
# document type
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    """One searchable document.  Lives as one line of docs.jsonl."""
    id:            str          # globally unique, e.g. "kegg:hsa00010"
    source:        str          # "kegg" | "reactome"
    pathway_id:    str          # "hsa00010" | "R-HSA-1433617"
    pathway_name:  str
    text:          str          # the actual biology paragraph
    gene_ids:      list[str]    # member genes (Entrez for KEGG, ENSG for Reactome)
    gene_id_type:  str          # "entrez" | "ensg"
    n_genes:       int = 0
    n_chars:       int = 0

    def __post_init__(self) -> None:
        self.n_genes = len(self.gene_ids)
        self.n_chars = len(self.text)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_kegg_docs(pathway_dir: Path) -> list[Doc]:
    """Build one Doc per KEGG pathway."""
    p2n = json.loads((pathway_dir / "02_pathway_id_to_name.json").read_text())
    p2d = json.loads((pathway_dir / "04_pathway_to_description.json").read_text())
    g2p = json.loads((pathway_dir / "01_gene_to_pathway_ids.json").read_text())

    # invert g2p so we can look up genes per pathway in one pass
    pathway_to_genes: dict[str, list[str]] = {}
    for gene, pathways in g2p.items():
        for pid in pathways:
            pathway_to_genes.setdefault(pid, []).append(gene)

    docs: list[Doc] = []
    for pid, text in p2d.items():
        text = (text or "").strip()
        if not text:
            continue
        docs.append(Doc(
            id           = f"kegg:{pid}",
            source       = "kegg",
            pathway_id   = pid,
            pathway_name = p2n.get(pid, pid),
            text         = text,
            gene_ids     = sorted(set(pathway_to_genes.get(pid, []))),
            gene_id_type = "entrez",
        ))
    return docs


def load_reactome_docs(pathway_dir: Path) -> list[Doc]:
    """Build one Doc per Reactome pathway."""
    sum_dir = pathway_dir / "reactome" / "R2_summations"
    g2p     = json.loads((pathway_dir / "reactome" / "R1_gene_to_pathways.json").read_text())

    # build pathway_id → name AND pathway_id → list[gene] in one inversion pass
    pathway_to_name:  dict[str, str]       = {}
    pathway_to_genes: dict[str, list[str]] = {}
    for ensg, pairs in g2p.items():
        for pid, pname in pairs:
            pathway_to_name[pid] = pname
            pathway_to_genes.setdefault(pid, []).append(ensg)

    docs: list[Doc] = []
    for fname in sorted(os.listdir(sum_dir)):
        if not fname.endswith(".json"):
            continue
        pid = fname[:-len(".json")]
        try:
            payload = json.loads((sum_dir / fname).read_text())
        except Exception:
            continue
        text = (payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        if not text:
            continue
        docs.append(Doc(
            id           = f"reactome:{pid}",
            source       = "reactome",
            pathway_id   = pid,
            pathway_name = pathway_to_name.get(pid, pid),
            text         = text,
            gene_ids     = sorted(set(pathway_to_genes.get(pid, []))),
            gene_id_type = "ensg",
        ))
    return docs


# ---------------------------------------------------------------------------
# BioBERT encoder (mean pooling, mask-aware) — same recipe as enrichment
# ---------------------------------------------------------------------------

def _load_biobert(model_name: str, device: str):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            f"transformers not installed: {e}\n"
            "Install with:  pip install transformers torch"
        ) from e
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    return tok, mdl, torch


def encode_texts(
    texts:        list[str],
    model_name:   str,
    max_seq_len:  int,
    batch_size:   int,
    device:       str,
) -> np.ndarray:
    tok, mdl, torch = _load_biobert(model_name, device)
    out = np.zeros((len(texts), mdl.config.hidden_size), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True, truncation=True,
                max_length=max_seq_len, return_tensors="pt",
            ).to(device)
            hidden = mdl(**enc).last_hidden_state          # [B, L, H]
            mask   = enc["attention_mask"].unsqueeze(-1)   # [B, L, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out[i : i + batch_size] = pooled.cpu().numpy()

            if (i // batch_size) % 10 == 0:
                pct = 100.0 * (i + len(batch)) / len(texts)
                print(f"  encoding... {i + len(batch):>5}/{len(texts)} ({pct:5.1f}%)")
    return out


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def write_docs_jsonl(docs: Iterable[Doc], path: Path) -> None:
    with path.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def build_gene_to_doc_ids(docs: list[Doc]) -> dict[str, list[str]]:
    """Reverse index: gene_id -> [doc_id, ...].  Mixes Entrez + ENSG keys."""
    rev: dict[str, list[str]] = {}
    for d in docs:
        for g in d.gene_ids:
            rev.setdefault(g, []).append(d.id)
    return rev


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(cfg: BuildConfig) -> dict:
    t0 = time.time()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] loading KEGG pathway docs from {cfg.pathway_dir}")
    kegg = load_kegg_docs(cfg.pathway_dir)
    print(f"    -> {len(kegg)} KEGG docs")

    print(f"[2] loading Reactome pathway docs")
    reactome = load_reactome_docs(cfg.pathway_dir)
    print(f"    -> {len(reactome)} Reactome docs")

    docs = kegg + reactome
    if cfg.limit is not None:
        docs = docs[: cfg.limit]
        print(f"[!] --limit applied: keeping first {len(docs)} docs")

    docs_path = cfg.out_dir / "docs.jsonl"
    write_docs_jsonl(docs, docs_path)
    print(f"[3] wrote {docs_path}  ({len(docs)} docs)")

    rev = build_gene_to_doc_ids(docs)
    rev_path = cfg.out_dir / "gene_to_doc_ids.json"
    rev_path.write_text(json.dumps(rev))
    print(f"[4] wrote {rev_path}  ({len(rev)} genes indexed)")

    embeddings_path = cfg.out_dir / "embeddings.npy"
    id_map_path     = cfg.out_dir / "id_map.json"

    if cfg.skip_embed:
        print("[5] --skip-embed set; skipping BioBERT encoding")
    else:
        print(f"[5] encoding {len(docs)} docs with {cfg.biobert_model} on {cfg.device}")
        emb = encode_texts(
            texts        = [d.text for d in docs],
            model_name   = cfg.biobert_model,
            max_seq_len  = cfg.max_seq_length,
            batch_size   = cfg.batch_size,
            device       = cfg.device,
        )
        np.save(embeddings_path, emb)
        print(f"    -> wrote {embeddings_path}  shape={emb.shape}  dtype={emb.dtype}")
        id_map_path.write_text(json.dumps({str(i): d.id for i, d in enumerate(docs)}))
        print(f"    -> wrote {id_map_path}")

    elapsed = round(time.time() - t0, 1)
    log = {
        "elapsed_seconds":  elapsed,
        "n_docs_total":     len(docs),
        "n_docs_kegg":      sum(1 for d in docs if d.source == "kegg"),
        "n_docs_reactome":  sum(1 for d in docs if d.source == "reactome"),
        "n_genes_indexed":  len(rev),
        "biobert_model":    cfg.biobert_model,
        "max_seq_length":   cfg.max_seq_length,
        "batch_size":       cfg.batch_size,
        "device":           cfg.device,
        "skip_embed":       cfg.skip_embed,
        "pathway_dir":      str(cfg.pathway_dir),
        "out_dir":          str(cfg.out_dir),
    }
    (cfg.out_dir / "build_log.json").write_text(json.dumps(log, indent=2))
    print()
    print(f"DONE in {elapsed}s.  Build log → {cfg.out_dir / 'build_log.json'}")
    return log


def parse_args() -> BuildConfig:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pathway-dir",  type=Path, default=BuildConfig.pathway_dir)
    ap.add_argument("--out-dir",      type=Path, default=BuildConfig.out_dir)
    ap.add_argument("--biobert",      type=str,  default=DEFAULT_BIOBERT)
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--batch-size",   type=int,  default=32)
    ap.add_argument("--limit",        type=int,  default=None)
    ap.add_argument("--skip-embed",   action="store_true")
    ap.add_argument("--device",       type=str,  default="cpu",
                    help="cpu / cuda / mps")
    a = ap.parse_args()
    return BuildConfig(
        pathway_dir    = a.pathway_dir,
        out_dir        = a.out_dir,
        biobert_model  = a.biobert,
        max_seq_length = a.max_seq_length,
        batch_size     = a.batch_size,
        limit          = a.limit,
        skip_embed     = a.skip_embed,
        device         = a.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
