"""WikiPathways → gene-to-descriptions extraction.

WikiPathways' REST/SOAP API was deprecated in late 2024. We use their
GMT distribution file instead, which contains gene-pathway mappings + names.

Single stage:
    W1: discover latest GMT filename, download, parse
        → ENTREZ_id → [pathway_name, ...]
        Output: <out>/W1_gene_to_descriptions.json

Note: the GMT gives us pathway NAMES only, not full descriptions.
We use names directly. Cosine dedup in the merge stage will collapse
WP names against richer Reactome/KEGG descriptions when they refer
to the same pathway concept (e.g., "Wnt signaling pathway" from WP
will have a high cosine similarity to Reactome's prose description
of the same pathway and get collapsed).

Gene ID note: WikiPathways GMT files use NCBI Entrez gene IDs.
We keep them as-is here; the merge stage uses mygene.info to map
Entrez → ENSG → ENSP for the final output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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

WP_GMT_INDEX = "https://data.wikipathways.org/current/gmt/"
GMT_FILE_RE  = re.compile(r"wikipathways-\d+-gmt-Homo_sapiens\.gmt")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class WikiPathwaysConfig:
    output_dir: Path = Path("data/pathway/wikipathways")


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
    reraise=True,
)
def _http_get(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def _discover_latest_gmt_filename() -> str:
    """Scrape the WP index page to find the dated GMT filename for human."""
    resp = _http_get(WP_GMT_INDEX, timeout=30)
    matches = GMT_FILE_RE.findall(resp.text)
    if not matches:
        raise RuntimeError(
            f"could not find any wikipathways-*-gmt-Homo_sapiens.gmt entries "
            f"on {WP_GMT_INDEX}. Index page format may have changed."
        )
    # files are dated; take the lexicographically last (= newest)
    return sorted(set(matches))[-1]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _parse_gmt_line(line: str) -> tuple[str, str, list[str]] | None:
    """Parse one line of the WikiPathways GMT.

    Format:
        {name}%WikiPathways_{date}%{WPid}%Homo sapiens<tab>http://...<tab>gene_1<tab>gene_2<tab>...
    """
    line = line.rstrip("\n").rstrip("\r")
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t")
    if len(parts) < 3:
        return None

    header = parts[0]                                    # e.g. "Wnt Signaling%WikiPathways_20260410%WP428%Homo sapiens"
    header_parts = header.split("%")
    if len(header_parts) < 3:
        return None
    pathway_name = header_parts[0].strip()
    pathway_id   = header_parts[2].strip()                # WPxxx
    if not pathway_name or not pathway_id:
        return None

    # parts[1] is the URL, parts[2:] are gene IDs (Entrez integers)
    gene_ids = [g.strip() for g in parts[2:] if g.strip()]
    return pathway_id, pathway_name, gene_ids


# ---------------------------------------------------------------------------
# Stage W1
# ---------------------------------------------------------------------------

def fetch_and_parse_gmt(cfg: WikiPathwaysConfig) -> dict[str, list[str]]:
    """Stage W1: download + parse GMT → {entrez_id: [pathway_name, ...]}."""
    output_path = cfg.output_dir / "W1_gene_to_descriptions.json"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[W1] loading cached: {output_path}")
        return json.loads(output_path.read_text())

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- discover + download GMT ----
    gmt_filename = _discover_latest_gmt_filename()
    gmt_url = f"{WP_GMT_INDEX}{gmt_filename}"
    cache_path = cfg.output_dir / gmt_filename

    if not cache_path.exists() or cache_path.stat().st_size == 0:
        print(f"[W1] downloading {gmt_url}")
        resp = _http_get(gmt_url, timeout=120)
        cache_path.write_bytes(resp.content)
        print(f"[W1] saved {len(resp.content)/1e6:.1f} MB → {cache_path}")
    else:
        print(f"[W1] using cached: {cache_path}")

    # ---- parse ----
    gene_to_pathway_names: dict[str, list[str]] = {}
    n_lines     = 0
    n_pathways  = 0
    n_gene_hits = 0

    with cache_path.open() as f:
        for line in f:
            n_lines += 1
            parsed = _parse_gmt_line(line)
            if parsed is None:
                continue
            pathway_id, pathway_name, gene_ids = parsed
            n_pathways += 1
            for gid in gene_ids:
                gene_to_pathway_names.setdefault(gid, []).append(pathway_name)
                n_gene_hits += 1

    # dedup per gene (rarely a gene appears twice in same pathway)
    for g, names in gene_to_pathway_names.items():
        gene_to_pathway_names[g] = list(dict.fromkeys(names))

    print(f"[W1] parsed {n_lines} lines, {n_pathways} pathways, "
          f"{n_gene_hits} (gene, pathway) hits, {len(gene_to_pathway_names)} unique genes")

    output_path.write_text(json.dumps(gene_to_pathway_names, indent=2))
    print(f"[W1] wrote → {output_path}")
    return gene_to_pathway_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WikiPathways → gene-to-descriptions extraction")
    p.add_argument("--output-dir", type=Path, default=Path("data/pathway/wikipathways"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = WikiPathwaysConfig(output_dir=args.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    gene_to_descs = fetch_and_parse_gmt(cfg)
    elapsed = time.time() - t0

    n_total_descs = sum(len(v) for v in gene_to_descs.values())
    avg_per_gene = n_total_descs / max(len(gene_to_descs), 1)

    print()
    print("=== WikiPathways extraction summary ===")
    print(f"  Genes (Entrez IDs): {len(gene_to_descs)}")
    print(f"  Total descriptions: {n_total_descs}")
    print(f"  Avg desc/gene:      {avg_per_gene:.1f}")
    print(f"  Elapsed:            {elapsed:.1f}s")
    print(f"  Output:             {cfg.output_dir / 'W1_gene_to_descriptions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
