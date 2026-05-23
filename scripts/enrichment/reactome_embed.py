"""Reactome → gene-to-descriptions extraction.

Three stages, each with a file checkpoint so re-runs skip completed work:

    R1: download Ensembl2Reactome_All_Levels.txt, filter to Homo sapiens
        → ENSG → [(pathway_id, pathway_name), ...]
        Output: <out>/R1_gene_to_pathways.json

    R2: for each unique pathway_id, fetch summation via REST API (cached per-pathway)
        → pathway_id → summation_text
        Output: <out>/R2_summations/{pathway_id}.json + R2_DONE marker

    R3: per gene, compile description list = ["{name}. {summation}", ...]
        → ENSG → [description_1, description_2, ...]
        Output: <out>/R3_gene_to_descriptions.json   (this is what merge stage consumes)

Usage:
    python -m scripts.enrichment.reactome_embed                              # full human
    python -m scripts.enrichment.reactome_embed --gene-subset 50             # smoke test
    python -m scripts.enrichment.reactome_embed --output-dir other/dir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

REACTOME_MAPPING_URL = "https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt"
REACTOME_API_URL     = "https://reactome.org/ContentService/data/query/{pathway_id}"
HUMAN_SPECIES        = "Homo sapiens"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class ReactomeConfig:
    output_dir:  Path                = Path("data/pathway/reactome")
    sleep:       float               = 0.05     # seconds between API calls
    flush_every: int                 = 100      # progress print interval
    gene_subset: Optional[int]       = None     # None = all genes


# ---------------------------------------------------------------------------
# http helpers
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
    reraise=True,
)
def _http_get(url: str, timeout: int = 30) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------------
# Stage R1 — gene → pathway mappings
# ---------------------------------------------------------------------------

def fetch_gene_pathways(cfg: ReactomeConfig) -> dict[str, list[tuple[str, str]]]:
    """Stage R1: download + parse Ensembl2Reactome file → ENSG → [(pid, name), ...]"""
    output_path = cfg.output_dir / "R1_gene_to_pathways.json"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[R1] loading cached: {output_path}")
        return {k: [tuple(p) for p in v]                                       # tuple() back from list
                for k, v in json.loads(output_path.read_text()).items()}

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- download (cached) ----
    raw_path = cfg.output_dir / "Ensembl2Reactome_All_Levels.txt"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        print(f"[R1] downloading {REACTOME_MAPPING_URL}")
        resp = _http_get(REACTOME_MAPPING_URL, timeout=120)
        raw_path.write_bytes(resp.content)
        print(f"[R1] saved {len(resp.content)/1e6:.1f} MB → {raw_path}")
    else:
        print(f"[R1] using cached: {raw_path}")

    # ---- parse + filter to human ----
    gene_to_paths: dict[str, list[tuple[str, str]]] = {}
    n_total = 0
    n_human = 0
    with raw_path.open() as f:
        for line in f:
            n_total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            gene_id, pathway_id, _url, pathway_name, _evidence, species = parts[:6]
            if species != HUMAN_SPECIES:
                continue
            if not gene_id.startswith("ENSG"):
                continue
            n_human += 1
            gene_to_paths.setdefault(gene_id, []).append((pathway_id, pathway_name))

    # dedup per gene (the file sometimes has the same (gene, pathway) on multiple lines)
    for g, pairs in gene_to_paths.items():
        gene_to_paths[g] = list(dict.fromkeys(pairs))   # preserves order

    print(f"[R1] parsed {n_total} total entries, {n_human} human, "
          f"{len(gene_to_paths)} unique ENSG genes")

    output_path.write_text(json.dumps(gene_to_paths, indent=2))
    print(f"[R1] wrote → {output_path}")
    return gene_to_paths


# ---------------------------------------------------------------------------
# Stage R2 — pathway → summation  (cached per-pathway)
# ---------------------------------------------------------------------------

def fetch_pathway_summations(
    cfg:           ReactomeConfig,
    gene_to_paths: dict[str, list[tuple[str, str]]],
) -> dict[str, str]:
    """Stage R2: for each unique pathway_id, fetch summation. Cache per-pathway."""
    cache_dir   = cfg.output_dir / "R2_summations"
    done_marker = cfg.output_dir / "R2_DONE"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect unique pathway IDs across all genes ----
    pathway_ids: set[str] = set()
    for pairs in gene_to_paths.values():
        for pid, _name in pairs:
            pathway_ids.add(pid)
    pathway_ids_sorted = sorted(pathway_ids)
    print(f"[R2] {len(pathway_ids_sorted)} unique pathways need summations")

    summations: dict[str, str] = {}
    n_cached = 0
    n_fetched = 0
    n_failed = 0
    t0 = time.time()

    for i, pid in enumerate(pathway_ids_sorted):
        cache_path = cache_dir / f"{pid}.json"

        # try cache first
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                cached = json.loads(cache_path.read_text())
                summations[pid] = cached.get("text", "")
                n_cached += 1
                continue
            except json.JSONDecodeError:
                cache_path.unlink()   # corrupted, refetch

        # fetch
        try:
            url = REACTOME_API_URL.format(pathway_id=pid)
            r = _http_get(url, timeout=15)
            data = r.json()
            sumn = data.get("summation", [])
            text = sumn[0].get("text", "") if sumn else ""
            cache_path.write_text(json.dumps({"text": text}))
            summations[pid] = text
            n_fetched += 1
            time.sleep(cfg.sleep)
        except Exception as exc:
            print(f"[R2] FAILED for {pid}: {type(exc).__name__}: {exc}")
            summations[pid] = ""
            n_failed += 1

        if (i + 1) % cfg.flush_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            remaining = (len(pathway_ids_sorted) - (i + 1)) / max(rate, 1e-6)
            print(f"[R2] {i + 1}/{len(pathway_ids_sorted)}  "
                  f"(fetched={n_fetched}, cached={n_cached}, failed={n_failed})  "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    done_marker.write_text(
        f"fetched={n_fetched}, cached={n_cached}, failed={n_failed}\n"
    )
    print(f"[R2] done: fetched={n_fetched}, cached={n_cached}, failed={n_failed}")
    return summations


# ---------------------------------------------------------------------------
# Stage R3 — compile per-gene description lists
# ---------------------------------------------------------------------------

def compile_gene_descriptions(
    cfg:           ReactomeConfig,
    gene_to_paths: dict[str, list[tuple[str, str]]],
    summations:    dict[str, str],
) -> dict[str, list[str]]:
    """Stage R3: per gene, build description list = ['{name}. {summation}', ...]"""
    output_path = cfg.output_dir / "R3_gene_to_descriptions.json"

    gene_to_descs: dict[str, list[str]] = {}
    n_descs = 0
    n_with_summation = 0
    n_fallback_name_only = 0

    for gene_id, pairs in gene_to_paths.items():
        descs: list[str] = []
        for pid, name in pairs:
            sumn = summations.get(pid, "")
            if sumn:
                descs.append(f"{name}. {sumn}")
                n_with_summation += 1
            else:
                descs.append(name)             # fallback: just the pathway name
                n_fallback_name_only += 1
        if descs:
            gene_to_descs[gene_id] = descs
            n_descs += len(descs)

    output_path.write_text(json.dumps(gene_to_descs, indent=2))
    print(f"[R3] compiled {n_descs} descriptions across {len(gene_to_descs)} genes "
          f"({n_with_summation} with summation, {n_fallback_name_only} name-only)")
    print(f"[R3] wrote → {output_path}")
    return gene_to_descs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reactome → gene-to-descriptions extraction")
    p.add_argument("--output-dir",  type=Path,  default=Path("data/pathway/reactome"))
    p.add_argument("--sleep",       type=float, default=0.05,
                   help="seconds between API calls (default: 0.05)")
    p.add_argument("--gene-subset", type=int,   default=None,
                   help="limit to first N genes (for smoke testing)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    cfg = ReactomeConfig(
        output_dir=args.output_dir,
        sleep=args.sleep,
        gene_subset=args.gene_subset,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    gene_to_paths = fetch_gene_pathways(cfg)

    if cfg.gene_subset is not None:
        before = len(gene_to_paths)
        gene_to_paths = dict(list(gene_to_paths.items())[:cfg.gene_subset])
        print(f"[main] --gene-subset: limited to {len(gene_to_paths)}/{before} genes")

    summations = fetch_pathway_summations(cfg, gene_to_paths)
    gene_to_descs = compile_gene_descriptions(cfg, gene_to_paths, summations)

    elapsed = time.time() - t0
    n_total_descs = sum(len(v) for v in gene_to_descs.values())
    avg_per_gene = n_total_descs / max(len(gene_to_descs), 1)
    print()
    print("=== Reactome extraction summary ===")
    print(f"  Genes processed:    {len(gene_to_descs)}")
    print(f"  Unique pathways:    {len(summations)}")
    print(f"  Total descriptions: {n_total_descs}")
    print(f"  Avg desc/gene:      {avg_per_gene:.1f}")
    print(f"  Elapsed:            {elapsed:.0f}s")
    print(f"  Output:             {cfg.output_dir / 'R3_gene_to_descriptions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
