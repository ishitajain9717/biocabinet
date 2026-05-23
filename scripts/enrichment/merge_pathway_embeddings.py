"""Merge KEGG + Reactome + WikiPathways pathway descriptions into a single
per-protein BioBERT embedding, with cosine-similarity deduplication.

Pipeline:
    M1: Load source descriptions
        KEGG     : <kegg_descs_dir>/{entrez}.json        → list[str]
        Reactome : <reactome_descs_path>                 → {ENSG: list[str]}
        WP       : <wp_descs_path>                       → {Entrez: list[str]}

    M2: Normalize all gene keys to Entrez (Reactome's ENSG → Entrez via mygene).
        Build {entrez: [(description, source_priority), ...]}.

    M3: (Optional) Filter to a target protein set, mapping target ENSPs back to Entrez.

    M4: BioBERT-embed every (entrez, description) instance.

    M5: Per-gene cosine dedup with source priority (Reactome > KEGG > WP, threshold 0.90).
        Save per-gene provenance JSON sidecar.

    M6: Mean-pool kept descriptions per gene.

    M7: Map Entrez → ENSP isoforms via mygene; propagate gene-level vector to each isoform.

Output:
    <out>/06_pathway_embeddings_combined.pt    {ensp: torch.Tensor(768)}
    <out>/06_combined_provenance.json          per-gene dedup audit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


# ---------------------------------------------------------------------------
# constants / config
# ---------------------------------------------------------------------------

# Source priority: lower = kept first in greedy dedup. Reactome's prose summaries
# are richest, KEGG is the second-most curated, WP gives just pathway names.
SOURCE_PRIORITY = {"reactome": 0, "kegg": 1, "wikipathways": 2}

DEFAULT_BIOBERT = "dmis-lab/biobert-v1.1"


@dataclass
class MergeConfig:
    kegg_descs_dir:        Path  = Path("data/pathway/04_descriptions")
    reactome_descs_path:   Path  = Path("data/pathway/reactome/R3_gene_to_descriptions.json")
    wp_descs_path:         Path  = Path("data/pathway/wikipathways/W1_gene_to_descriptions.json")
    output_dir:            Path  = Path("data/pathway")
    target_proteins_file:  Optional[Path] = None    # one ENSP per line; None = all genes
    biobert_model:         str   = DEFAULT_BIOBERT
    dedup_threshold:       float = 0.90
    biobert_batch_size:    int   = 8
    device:                str   = "cpu"


# ---------------------------------------------------------------------------
# Stage M1 — loaders (one per source)
# ---------------------------------------------------------------------------

def load_kegg_descriptions(descs_dir: Path) -> dict[str, list[str]]:
    """Walk <dir>/{entrez}.json files → {entrez: list[str]}."""
    out: dict[str, list[str]] = {}
    if not descs_dir.exists():
        print(f"[M1.kegg] ⚠️  KEGG descriptions dir missing: {descs_dir}")
        return out
    for f in descs_dir.glob("*.json"):
        entrez = f.stem
        try:
            descs = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(descs, list) and descs:
            out[entrez] = [d for d in descs if isinstance(d, str) and d.strip()]
    print(f"[M1.kegg]      loaded {len(out)} genes from {descs_dir}")
    return out


def load_reactome_descriptions(path: Path) -> dict[str, list[str]]:
    """Load {ENSG: list[str]} from JSON."""
    if not path.exists():
        print(f"[M1.reactome] ⚠️  Reactome descriptions missing: {path}")
        return {}
    data = json.loads(path.read_text())
    print(f"[M1.reactome]  loaded {len(data)} ENSG genes from {path}")
    return data


def load_wp_descriptions(path: Path) -> dict[str, list[str]]:
    """Load {Entrez: list[str]} from JSON."""
    if not path.exists():
        print(f"[M1.wp] ⚠️  WikiPathways descriptions missing: {path}")
        return {}
    data = json.loads(path.read_text())
    print(f"[M1.wp]        loaded {len(data)} Entrez genes from {path}")
    return data


# ---------------------------------------------------------------------------
# Stage M2 — gene ID normalization (mygene.info)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _mygene_query_batch(ids: list[str], scopes: str, fields: str) -> list[dict]:
    import mygene
    mg = mygene.MyGeneInfo()
    return mg.querymany(ids, scopes=scopes, fields=fields, species="human", verbose=False)


def map_ensg_to_entrez(ensg_ids: list[str]) -> dict[str, str]:
    """Batch-query mygene.info: ENSG → Entrez."""
    if not ensg_ids:
        return {}
    print(f"[M2] mygene: mapping {len(ensg_ids)} ENSG → Entrez ...")
    results = _mygene_query_batch(ensg_ids, scopes="ensembl.gene", fields="entrezgene")
    mapping: dict[str, str] = {}
    for r in results:
        ensg = r.get("query")
        ent  = r.get("entrezgene")
        if ensg and ent:
            mapping[ensg] = str(ent)
    print(f"[M2] mapped {len(mapping)}/{len(ensg_ids)} ENSGs to Entrez")
    return mapping


def map_entrez_to_ensps(entrez_ids: list[str]) -> dict[str, list[str]]:
    """Batch-query mygene.info: Entrez → list of ENSP isoforms."""
    if not entrez_ids:
        return {}
    print(f"[M7] mygene: mapping {len(entrez_ids)} Entrez → ENSP isoforms ...")
    results = _mygene_query_batch(entrez_ids, scopes="entrezgene", fields="ensembl.protein")
    mapping: dict[str, list[str]] = {}
    for r in results:
        entrez = r.get("query")
        ens = r.get("ensembl")
        if not entrez or not ens:
            continue
        # ensembl can be a dict or a list of dicts
        if isinstance(ens, list):
            ensps: list[str] = []
            for e in ens:
                p = e.get("protein")
                if isinstance(p, list):
                    ensps.extend(p)
                elif isinstance(p, str):
                    ensps.append(p)
        else:
            p = ens.get("protein")
            ensps = p if isinstance(p, list) else ([p] if isinstance(p, str) else [])
        if ensps:
            mapping[entrez] = ensps
    print(f"[M7] mapped {len(mapping)}/{len(entrez_ids)} Entrez to ENSP isoforms")
    return mapping


def map_ensp_to_entrez(ensps: list[str]) -> dict[str, str]:
    """Batch-query mygene.info: ENSP → Entrez."""
    if not ensps:
        return {}
    print(f"[M3] mygene: mapping {len(ensps)} ENSP → Entrez ...")
    results = _mygene_query_batch(ensps, scopes="ensembl.protein", fields="entrezgene")
    mapping: dict[str, str] = {}
    for r in results:
        ensp = r.get("query")
        ent  = r.get("entrezgene")
        if ensp and ent:
            mapping[ensp] = str(ent)
    print(f"[M3] mapped {len(mapping)}/{len(ensps)} ENSPs to Entrez")
    return mapping


# ---------------------------------------------------------------------------
# Stage M2 — merge into one dict with provenance
# ---------------------------------------------------------------------------

@dataclass
class TaggedDesc:
    text:   str
    source: str          # "kegg" | "reactome" | "wikipathways"


def merge_sources(
    kegg:     dict[str, list[str]],
    reactome: dict[str, list[str]],
    wp:       dict[str, list[str]],
    ensg_to_entrez: dict[str, str],
) -> dict[str, list[TaggedDesc]]:
    """Build {entrez: [TaggedDesc, ...]} merging across sources."""
    merged: dict[str, list[TaggedDesc]] = {}

    for entrez, descs in kegg.items():
        for d in descs:
            merged.setdefault(entrez, []).append(TaggedDesc(d, "kegg"))

    for ensg, descs in reactome.items():
        entrez = ensg_to_entrez.get(ensg)
        if entrez is None:
            continue
        for d in descs:
            merged.setdefault(entrez, []).append(TaggedDesc(d, "reactome"))

    for entrez, descs in wp.items():
        for d in descs:
            merged.setdefault(entrez, []).append(TaggedDesc(d, "wikipathways"))

    return merged


# ---------------------------------------------------------------------------
# Stage M4 — BioBERT embedding
# ---------------------------------------------------------------------------

def _load_biobert(model_name: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModel
    print(f"[M4] loading BioBERT: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    return tok, mdl


@torch.no_grad()
def embed_descriptions(
    descs:        list[str],
    tokenizer,
    model,
    device:       torch.device,
    batch_size:   int = 8,
) -> torch.Tensor:
    """Mean-pool token embeddings → (N, 768) float32 tensor."""
    out_chunks: list[torch.Tensor] = []
    for i in range(0, len(descs), batch_size):
        chunk = descs[i : i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        outputs = model(**inputs)                           # last_hidden_state: (B, L, 768)
        mask = inputs.attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_emb = (outputs.last_hidden_state * mask).sum(dim=1)   # (B, 768)
        cnt = mask.sum(dim=1).clamp(min=1.0)                       # (B, 1)
        pooled = (sum_emb / cnt).cpu().float()
        out_chunks.append(pooled)
    return torch.cat(out_chunks, dim=0) if out_chunks else torch.zeros(0, model.config.hidden_size)


# ---------------------------------------------------------------------------
# Stage M5 — cosine dedup with source priority
# ---------------------------------------------------------------------------

def cosine_greedy_dedup(
    embeddings: torch.Tensor,    # (N, 768)
    sources:    list[str],
    threshold:  float = 0.90,
) -> tuple[list[int], list[dict]]:
    """Sort by source priority then greedy-keep, dropping anything > threshold cos sim with kept set.

    Returns (kept_indices_in_original_order, audit_records).
    """
    if embeddings.shape[0] == 0:
        return [], []
    n = embeddings.shape[0]

    # sort indices by (source priority, original index) — stable
    order = sorted(range(n), key=lambda i: (SOURCE_PRIORITY.get(sources[i], 99), i))

    # normalize for cosine
    normed = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)

    kept_indices: list[int] = []
    audit: list[dict] = []
    for idx in order:
        is_dup = False
        max_sim = -1.0
        dup_against = -1
        for ki in kept_indices:
            sim = float(torch.dot(normed[idx], normed[ki]))
            if sim > max_sim:
                max_sim = sim
                dup_against = ki
            if sim > threshold:
                is_dup = True
                break
        audit.append({
            "index":        idx,
            "source":       sources[idx],
            "kept":         not is_dup,
            "max_sim":      round(max_sim, 4) if kept_indices else None,
            "dup_against":  dup_against if is_dup else None,
        })
        if not is_dup:
            kept_indices.append(idx)

    kept_indices.sort()       # restore original order so the caller can read sources/embeddings naturally
    return kept_indices, audit


# ---------------------------------------------------------------------------
# top-level orchestration
# ---------------------------------------------------------------------------

def run(cfg: MergeConfig) -> dict:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    # ---- M1: load all 3 ----
    print("=" * 70)
    print("[M1] load all 3 sources")
    print("=" * 70)
    kegg     = load_kegg_descriptions(cfg.kegg_descs_dir)
    reactome = load_reactome_descriptions(cfg.reactome_descs_path)
    wp       = load_wp_descriptions(cfg.wp_descs_path)

    # ---- M2: normalize Reactome ENSG → Entrez ----
    print("\n" + "=" * 70)
    print("[M2] normalize gene IDs to Entrez")
    print("=" * 70)
    ensg_to_entrez = map_ensg_to_entrez(list(reactome.keys()))
    merged = merge_sources(kegg, reactome, wp, ensg_to_entrez)
    n_descs_total = sum(len(v) for v in merged.values())
    print(f"[M2] merged dict: {len(merged)} unique Entrez genes, {n_descs_total} total descriptions")

    # ---- M3: optional target protein filter ----
    target_entrez_set: Optional[set[str]] = None
    if cfg.target_proteins_file is not None:
        print("\n" + "=" * 70)
        print("[M3] filter to target proteins")
        print("=" * 70)
        ensps = [l.strip() for l in cfg.target_proteins_file.read_text().splitlines() if l.strip()]
        # support "9606.ENSP..." style by stripping species prefix
        ensps_bare = [e.split(".", 1)[-1] if e.startswith(("9606.", "ensg.")) or "." in e else e for e in ensps]
        ensp_to_ent = map_ensp_to_entrez(ensps_bare)
        target_entrez_set = set(ensp_to_ent.values())
        before = len(merged)
        merged = {e: descs for e, descs in merged.items() if e in target_entrez_set}
        print(f"[M3] filtered {before} → {len(merged)} genes via target ENSPs")

    # ---- M4: BioBERT embed ----
    print("\n" + "=" * 70)
    print("[M4] BioBERT embed descriptions")
    print("=" * 70)
    tokenizer, model = _load_biobert(cfg.biobert_model, device)
    bio_dim = model.config.hidden_size

    # flatten for batch embedding
    flat_entrez:  list[str] = []
    flat_text:    list[str] = []
    flat_source:  list[str] = []
    for entrez, tagged in merged.items():
        for td in tagged:
            flat_entrez.append(entrez)
            flat_text.append(td.text)
            flat_source.append(td.source)
    print(f"[M4] embedding {len(flat_text)} descriptions ...")
    t0 = time.time()
    flat_emb = embed_descriptions(flat_text, tokenizer, model, device, cfg.biobert_batch_size)
    print(f"[M4] done ({time.time()-t0:.0f}s)")

    # split back per gene
    per_gene: dict[str, dict] = {}
    cur = 0
    for entrez, tagged in merged.items():
        n = len(tagged)
        per_gene[entrez] = {
            "embeddings": flat_emb[cur : cur + n],
            "sources":    flat_source[cur : cur + n],
            "texts":      flat_text[cur : cur + n],
        }
        cur += n

    # ---- M5: cosine dedup per gene ----
    print("\n" + "=" * 70)
    print(f"[M5] cosine dedup per gene (threshold={cfg.dedup_threshold})")
    print("=" * 70)
    provenance: dict[str, dict] = {}
    n_total = 0
    n_kept  = 0
    for entrez, bag in per_gene.items():
        kept_idx, audit = cosine_greedy_dedup(
            bag["embeddings"], bag["sources"], cfg.dedup_threshold,
        )
        n_total += len(bag["sources"])
        n_kept  += len(kept_idx)
        bag["kept_idx"] = kept_idx
        provenance[entrez] = {
            "n_total": len(bag["sources"]),
            "n_kept":  len(kept_idx),
            "audit":   audit,
        }
    print(f"[M5] kept {n_kept}/{n_total} descriptions ({100*n_kept/max(n_total,1):.1f}%)")

    # ---- M6: mean-pool kept ----
    print("\n" + "=" * 70)
    print("[M6] mean-pool kept descriptions")
    print("=" * 70)
    gene_to_emb: dict[str, torch.Tensor] = {}
    for entrez, bag in per_gene.items():
        kept = bag["embeddings"][bag["kept_idx"]]    # (k, 768)
        if kept.shape[0] == 0:
            gene_to_emb[entrez] = torch.zeros(bio_dim)
        else:
            gene_to_emb[entrez] = kept.mean(dim=0)
    print(f"[M6] gene-level embeddings: {len(gene_to_emb)}")

    # ---- M7: map Entrez → ENSPs ----
    print("\n" + "=" * 70)
    print("[M7] map Entrez → ENSP isoforms")
    print("=" * 70)
    entrez_to_ensps = map_entrez_to_ensps(list(gene_to_emb.keys()))
    out_dict: dict[str, torch.Tensor] = {}
    for entrez, ensps in entrez_to_ensps.items():
        emb = gene_to_emb[entrez]
        for ensp in ensps:
            out_dict[ensp] = emb
            out_dict[f"9606.{ensp}"] = emb     # also include STRING-style key

    # ---- save ----
    out_pt   = cfg.output_dir / "06_pathway_embeddings_combined.pt"
    out_prov = cfg.output_dir / "06_combined_provenance.json"
    torch.save(out_dict, out_pt)
    out_prov.write_text(json.dumps(provenance, indent=2))
    print(f"\n[done] wrote {len(out_dict)} ENSP keys → {out_pt}")
    print(f"[done] provenance audit → {out_prov}")

    return {
        "n_genes_input":   len(merged) + (0 if target_entrez_set is None else 0),
        "n_descs_input":   n_total,
        "n_descs_kept":    n_kept,
        "n_genes_with_emb": len(gene_to_emb),
        "n_ensps_output":  len(out_dict),
        "out_pt":          str(out_pt),
        "out_prov":        str(out_prov),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge KEGG + Reactome + WikiPathways pathway embeddings")
    p.add_argument("--kegg-descs-dir",       type=Path, default=Path("data/pathway/04_descriptions"))
    p.add_argument("--reactome-descs-path",  type=Path, default=Path("data/pathway/reactome/R3_gene_to_descriptions.json"))
    p.add_argument("--wp-descs-path",        type=Path, default=Path("data/pathway/wikipathways/W1_gene_to_descriptions.json"))
    p.add_argument("--output-dir",           type=Path, default=Path("data/pathway"))
    p.add_argument("--target-proteins-file", type=Path, default=None,
                   help="optional file with one ENSP per line — only embed proteins in this set")
    p.add_argument("--biobert-model",        type=str,  default=DEFAULT_BIOBERT)
    p.add_argument("--dedup-threshold",      type=float, default=0.90)
    p.add_argument("--batch-size",           type=int,  default=8)
    p.add_argument("--device",               type=str,  default="cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = MergeConfig(
        kegg_descs_dir=args.kegg_descs_dir,
        reactome_descs_path=args.reactome_descs_path,
        wp_descs_path=args.wp_descs_path,
        output_dir=args.output_dir,
        target_proteins_file=args.target_proteins_file,
        biobert_model=args.biobert_model,
        dedup_threshold=args.dedup_threshold,
        biobert_batch_size=args.batch_size,
        device=args.device,
    )
    summary = run(cfg)
    print("\n=== Merge summary ===")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
