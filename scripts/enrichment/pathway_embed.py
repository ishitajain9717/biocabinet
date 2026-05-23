"""Build BioBERT-based pathway embeddings for STRING/PPI proteins.

A 6-stage pipeline implemented as a LangGraph subgraph. Each stage writes its
output to disk so the build is resumable across runs (re-running skips stages
whose output file already exists).

Final output: data/pathway/06_pathway_embeddings.pt → {ENSP_id: torch.Tensor[768]}

CLI:
    python3 -m scripts.enrichment.pathway_embed                         # 50-gene smoke test
    python3 -m scripts.enrichment.pathway_embed --gene-subset 500
    python3 -m scripts.enrichment.pathway_embed --gene-subset all       # full ~20k, ~40h
    python3 -m scripts.enrichment.pathway_embed --rebuild               # nuke intermediates

NOTE: This file is being written in 3 sections; sections B and C will be added next.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from scripts.common.node_result import NodeResult


# ---------- config ----------

@dataclass
class PathwayEmbedConfig:
    output_dir:      Path
    gene_subset:     int | None = 50           # None means "all genes"
    request_sleep_s: float       = 0.5         # KEGG asks for ≤3 req/sec
    retry_attempts:  int         = 5


# ---------- state ----------

class PathwayEmbedState(TypedDict):
    config:        PathwayEmbedConfig
    node_history:  list[NodeResult]
    output_paths:  dict[str, str]
    n_genes:       int
    n_pathways:    int
    n_proteins:    int
    error:         str | None


# ---------- shared helpers ----------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def _http_get(url: str, timeout: int = 30) -> requests.Response:
    """GET with exponential backoff (1s, 2s, 4s, 8s, 16s, capped 30s)."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def _maybe_skip(out_path: Path, label: str) -> NodeResult | None:
    """If output file already exists and is non-empty, return a 'skipped' NodeResult."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return NodeResult(
            name=label,
            ok=True,
            message=f"skipped — {out_path.name} already exists",
            outputs={"path": str(out_path)},
            metrics={"skipped": True},
        )
    return None


# ---------- stage 1: gene → pathway IDs ----------

def node_fetch_gene_pathways(cfg: PathwayEmbedConfig) -> tuple[NodeResult, Path | None]:
    """KEGG /link/pathway/hsa → {entrez_gene_id: [kegg_pathway_id, ...]}"""
    out_path = cfg.output_dir / "01_gene_to_pathway_ids.json"
    skip = _maybe_skip(out_path, "fetch_gene_pathways")
    if skip is not None:
        with out_path.open() as f:
            existing = json.load(f)
        skip.metrics["n_genes"] = len(existing)
        return skip, out_path

    url = "https://rest.kegg.jp/link/pathway/hsa"
    try:
        text = _http_get(url).text
    except Exception as exc:
        return NodeResult(
            name="fetch_gene_pathways", ok=False,
            message=f"KEGG fetch failed: {exc}",
            outputs={}, metrics={},
        ), None

    gene_to_pathway_ids: dict[str, list[str]] = {}
    for line in text.strip().split("\n"):
        gene, pathway = line.split("\t")
        gene = gene.replace("hsa:", "")
        pathway = pathway.replace("path:", "")
        gene_to_pathway_ids.setdefault(gene, []).append(pathway)

    if cfg.gene_subset is not None:
        gene_to_pathway_ids = dict(list(gene_to_pathway_ids.items())[:cfg.gene_subset])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(gene_to_pathway_ids, f)

    return NodeResult(
        name="fetch_gene_pathways", ok=True,
        message=f"got pathways for {len(gene_to_pathway_ids)} genes",
        outputs={"path": str(out_path)},
        metrics={"n_genes": len(gene_to_pathway_ids)},
    ), out_path


# ---------- stage 2: pathway ID → name ----------

def node_fetch_pathway_names(cfg: PathwayEmbedConfig) -> tuple[NodeResult, Path | None]:
    """KEGG /list/pathway/hsa → {kegg_pathway_id: human_name}"""
    out_path = cfg.output_dir / "02_pathway_id_to_name.json"
    skip = _maybe_skip(out_path, "fetch_pathway_names")
    if skip is not None:
        with out_path.open() as f:
            skip.metrics["n_pathways"] = len(json.load(f))
        return skip, out_path

    url = "https://rest.kegg.jp/list/pathway/hsa"
    try:
        text = _http_get(url).text
    except Exception as exc:
        return NodeResult(
            name="fetch_pathway_names", ok=False,
            message=f"KEGG fetch failed: {exc}",
            outputs={}, metrics={},
        ), None

    pathway_id_to_name: dict[str, str] = {}
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pid = parts[0].replace("path:", "")
        pathway_id_to_name[pid] = parts[1].split(" - ")[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(pathway_id_to_name, f)

    return NodeResult(
        name="fetch_pathway_names", ok=True,
        message=f"got {len(pathway_id_to_name)} pathway names",
        outputs={"path": str(out_path)},
        metrics={"n_pathways": len(pathway_id_to_name)},
    ), out_path


# ---------- stage 3: gene ID → gene symbol ----------

def node_fetch_gene_names(cfg: PathwayEmbedConfig) -> tuple[NodeResult, Path | None]:
    """KEGG /list/hsa → {entrez_gene_id: gene_symbol}"""
    out_path = cfg.output_dir / "03_gene_id_to_name.json"
    skip = _maybe_skip(out_path, "fetch_gene_names")
    if skip is not None:
        with out_path.open() as f:
            skip.metrics["n_genes_named"] = len(json.load(f))
        return skip, out_path

    url = "https://rest.kegg.jp/list/hsa"
    try:
        text = _http_get(url).text
    except Exception as exc:
        return NodeResult(
            name="fetch_gene_names", ok=False,
            message=f"KEGG fetch failed: {exc}",
            outputs={}, metrics={},
        ), None

    gene_id_to_name: dict[str, str] = {}
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        gid = parts[0].replace("hsa:", "")
        gene_id_to_name[gid] = parts[-1].split(";")[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(gene_id_to_name, f)

    return NodeResult(
        name="fetch_gene_names", ok=True,
        message=f"got names for {len(gene_id_to_name)} genes",
        outputs={"path": str(out_path)},
        metrics={"n_genes_named": len(gene_id_to_name)},
    ), out_path


# =====================================================================
# SECTION B: heavy stages 4-6
# =====================================================================

# ---------- stage 4: pathway_id → description (with cross-pathway cache) ----------

def node_fetch_pathway_descriptions(
    cfg: PathwayEmbedConfig,
    stage1_path: Path,
    stage2_path: Path,
) -> tuple[NodeResult, Path | None]:
    """KEGG /get/{pathway_id} → per-gene JSON files of pathway descriptions.

    Output layout:
        04_descriptions/<gene_id>.json  → list[str]   (one per pathway the gene is in)
        04_pathway_to_description.json  → {pid: desc}  (cross-gene cache for resume)

    The whole stage is "skipped" if the pathway-cache file already covers all
    pathways referenced in stage 1's gene→pathway map. Per-gene JSONs are
    re-derived from the cache on a re-run (cheap).
    """
    out_dir = cfg.output_dir / "04_descriptions"
    cache_path = cfg.output_dir / "04_pathway_to_description.json"
    sentinel = cfg.output_dir / "04_DONE"

    with stage1_path.open() as f:
        gene_to_pathways: dict[str, list[str]] = json.load(f)

    needed_pids = {pid for pids in gene_to_pathways.values() for pid in pids}

    cache: dict[str, str] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            cache = json.load(f)

    missing_pids = sorted(needed_pids - cache.keys())

    if sentinel.exists() and not missing_pids:
        n_genes = len(gene_to_pathways)
        return NodeResult(
            name="fetch_pathway_descriptions", ok=True,
            message=f"skipped — cache covers all {len(needed_pids)} pathways for {n_genes} genes",
            outputs={"path": str(out_dir), "cache": str(cache_path)},
            metrics={"skipped": True, "n_pathways_cached": len(cache), "n_genes": n_genes},
        ), out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    n_fetched = 0
    n_failed = 0
    for i, pid in enumerate(missing_pids, 1):
        try:
            text = _http_get(f"https://rest.kegg.jp/get/{pid}").text
        except Exception as exc:
            print(f"  [stage4] {i}/{len(missing_pids)} {pid} FAILED: {exc}")
            n_failed += 1
            cache[pid] = ""
        else:
            desc = ""
            for line in text.splitlines():
                if line.startswith("DESCRIPTION"):
                    desc = line.replace("DESCRIPTION", "", 1).strip()
                    break
            cache[pid] = desc
            n_fetched += 1
        if i % 25 == 0 or i == len(missing_pids):
            with cache_path.open("w") as f:
                json.dump(cache, f)
            print(f"  [stage4] flushed cache at {i}/{len(missing_pids)}")
        time.sleep(cfg.request_sleep_s)

    with cache_path.open("w") as f:
        json.dump(cache, f)

    for gene, pids in gene_to_pathways.items():
        descs = [cache.get(pid, "") for pid in pids]
        with (out_dir / f"{gene}.json").open("w") as f:
            json.dump(descs, f)

    sentinel.write_text("done")

    return NodeResult(
        name="fetch_pathway_descriptions", ok=True,
        message=f"fetched {n_fetched} descriptions ({n_failed} failed); per-gene files in {out_dir.name}/",
        outputs={"path": str(out_dir), "cache": str(cache_path)},
        metrics={
            "n_pathways_total":  len(needed_pids),
            "n_pathways_fetched": n_fetched,
            "n_pathways_failed":  n_failed,
            "n_genes":           len(gene_to_pathways),
        },
    ), out_dir


# ---------- stage 5: BioBERT embedding ----------

def node_embed_with_biobert(
    cfg: PathwayEmbedConfig,
    stage4_dir: Path,
) -> tuple[NodeResult, Path | None]:
    """For each gene: encode each pathway description with BioBERT, mean-pool tokens
    per description, then mean across descriptions → 768-dim per gene.

    Output: 05_gene_to_embedding.pt  → {gene_id: torch.Tensor[768]}
    """
    import torch

    out_path = cfg.output_dir / "05_gene_to_embedding.pt"
    skip = _maybe_skip(out_path, "embed_with_biobert")
    if skip is not None:
        existing = torch.load(out_path, weights_only=False)
        skip.metrics["n_genes_embedded"] = len(existing)
        return skip, out_path

    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        model.eval()
    except Exception as exc:
        return NodeResult(
            name="embed_with_biobert", ok=False,
            message=f"BioBERT load failed: {exc}",
            outputs={}, metrics={},
        ), None

    @torch.no_grad()
    def sentence_vector(text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(model.config.hidden_size)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        out = model(**enc)
        return out.last_hidden_state.mean(dim=1).squeeze(0)

    gene_files = sorted(stage4_dir.glob("*.json"))
    gene_to_embedding: dict[str, "torch.Tensor"] = {}
    n_skipped = 0

    for i, gf in enumerate(gene_files, 1):
        with gf.open() as f:
            descs = json.load(f)
        descs_nonempty = [d for d in descs if d]
        if not descs_nonempty:
            n_skipped += 1
            continue
        per_desc = torch.stack([sentence_vector(d) for d in descs_nonempty])
        gene_to_embedding[gf.stem] = per_desc.mean(dim=0)
        if i % 25 == 0 or i == len(gene_files):
            print(f"  [stage5] embedded {i}/{len(gene_files)} genes")

    torch.save(gene_to_embedding, out_path)

    return NodeResult(
        name="embed_with_biobert", ok=True,
        message=f"embedded {len(gene_to_embedding)} genes ({n_skipped} had no descriptions)",
        outputs={"path": str(out_path)},
        metrics={
            "n_genes_embedded":  len(gene_to_embedding),
            "n_genes_skipped":   n_skipped,
            "embedding_dim":     model.config.hidden_size,
        },
    ), out_path


# ---------- stage 6: gene → ENSP_id mapping via mygene.info ----------

def node_map_to_ensp(
    cfg: PathwayEmbedConfig,
    stage5_path: Path,
) -> tuple[NodeResult, Path | None]:
    """For each gene: query mygene.info for its Ensembl protein IDs, then
    propagate the gene-level embedding to every protein isoform.

    Output: 06_pathway_embeddings.pt → {ENSP_id: torch.Tensor[768]}
    """
    import torch

    out_path = cfg.output_dir / "06_pathway_embeddings.pt"
    skip = _maybe_skip(out_path, "map_to_ensp")
    if skip is not None:
        existing = torch.load(out_path, weights_only=False)
        skip.metrics["n_proteins"] = len(existing)
        return skip, out_path

    gene_to_embedding = torch.load(stage5_path, weights_only=False)

    ensp_to_embedding: dict[str, "torch.Tensor"] = {}
    n_failed = 0
    for i, (gene_id, emb) in enumerate(gene_to_embedding.items(), 1):
        try:
            data = _http_get(f"https://mygene.info/v3/gene/{gene_id}?fields=ensembl.protein").json()
        except Exception as exc:
            print(f"  [stage6] {i}/{len(gene_to_embedding)} gene {gene_id} mygene FAILED: {exc}")
            n_failed += 1
            continue

        ens = data.get("ensembl", {}) or {}
        entries = ens if isinstance(ens, list) else [ens]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            prot = entry.get("protein")
            if isinstance(prot, list):
                for p in prot:
                    if p:
                        ensp_to_embedding.setdefault(p, emb)
            elif prot:
                ensp_to_embedding.setdefault(prot, emb)

        if i % 25 == 0 or i == len(gene_to_embedding):
            torch.save(ensp_to_embedding, out_path)
            print(f"  [stage6] {i}/{len(gene_to_embedding)} genes → {len(ensp_to_embedding)} proteins (flushed)")
        time.sleep(cfg.request_sleep_s)

    torch.save(ensp_to_embedding, out_path)

    return NodeResult(
        name="map_to_ensp", ok=True,
        message=f"{len(gene_to_embedding) - n_failed}/{len(gene_to_embedding)} genes mapped → {len(ensp_to_embedding)} proteins",
        outputs={"path": str(out_path)},
        metrics={
            "n_genes":      len(gene_to_embedding),
            "n_genes_failed": n_failed,
            "n_proteins":   len(ensp_to_embedding),
        },
    ), out_path


# =====================================================================
# SECTION C: LangGraph subgraph + CLI
# =====================================================================

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated

# Extend state with messages bus for orchestrator integration
class PathwayEmbedStateLG(TypedDict):
    config:        PathwayEmbedConfig
    node_history:  list[NodeResult]
    output_paths:  dict[str, str]
    n_genes:       int
    n_pathways:    int
    n_proteins:    int
    error:         str | None
    messages:      Annotated[list, add_messages]


# ---------- LangGraph wrappers ----------

def graph_node_collect_config(state: PathwayEmbedStateLG) -> dict:
    """No-op; config is supplied at graph invocation. Returns initial state."""
    return {
        "node_history": [],
        "output_paths": {},
        "n_genes": 0, "n_pathways": 0, "n_proteins": 0,
        "error": None,
    }


def _appended(state: PathwayEmbedStateLG, result: NodeResult) -> list:
    return list(state["node_history"]) + [result]


def graph_node_fetch_gene_pathways(state: PathwayEmbedStateLG) -> dict:
    r, p = node_fetch_gene_pathways(state["config"])
    if not r.ok:
        return {"node_history": _appended(state, r), "error": r.message}
    return {
        "node_history": _appended(state, r),
        "output_paths": {**state["output_paths"], "stage1": str(p)},
        "n_genes":      int(r.metrics.get("n_genes", 0)),
    }


def graph_node_fetch_pathway_names(state: PathwayEmbedStateLG) -> dict:
    r, p = node_fetch_pathway_names(state["config"])
    if not r.ok:
        return {"node_history": _appended(state, r), "error": r.message}
    return {
        "node_history": _appended(state, r),
        "output_paths": {**state["output_paths"], "stage2": str(p)},
        "n_pathways":   int(r.metrics.get("n_pathways", 0)),
    }


def graph_node_fetch_gene_names(state: PathwayEmbedStateLG) -> dict:
    r, p = node_fetch_gene_names(state["config"])
    if not r.ok:
        return {"node_history": _appended(state, r), "error": r.message}
    return {
        "node_history": _appended(state, r),
        "output_paths": {**state["output_paths"], "stage3": str(p)},
    }


def graph_node_fetch_pathway_descriptions(state: PathwayEmbedStateLG) -> dict:
    r, p = node_fetch_pathway_descriptions(
        state["config"],
        Path(state["output_paths"]["stage1"]),
        Path(state["output_paths"]["stage2"]),
    )
    if not r.ok:
        return {"node_history": _appended(state, r), "error": r.message}
    return {
        "node_history": _appended(state, r),
        "output_paths": {**state["output_paths"], "stage4": str(p)},
    }


def graph_node_embed_with_biobert(state: PathwayEmbedStateLG) -> dict:
    r, p = node_embed_with_biobert(
        state["config"], Path(state["output_paths"]["stage4"]),
    )
    if not r.ok:
        return {"node_history": _appended(state, r), "error": r.message}
    return {
        "node_history": _appended(state, r),
        "output_paths": {**state["output_paths"], "stage5": str(p)},
    }


def graph_node_map_to_ensp(state: PathwayEmbedStateLG) -> dict:
    r, p = node_map_to_ensp(
        state["config"], Path(state["output_paths"]["stage5"]),
    )
    if not r.ok:
        return {"node_history": _appended(state, r), "error": r.message}
    return {
        "node_history": _appended(state, r),
        "output_paths": {**state["output_paths"], "stage6": str(p), "final": str(p)},
        "n_proteins":   int(r.metrics.get("n_proteins", 0)),
    }


def graph_node_summarize(state: PathwayEmbedStateLG) -> dict:
    final = state["output_paths"].get("final", "(missing)")
    body = (
        f"Pathway embedding build complete.\n"
        f"  Genes:     {state.get('n_genes', '?')}\n"
        f"  Pathways:  {state.get('n_pathways', '?')}\n"
        f"  Proteins:  {state.get('n_proteins', '?')}\n"
        f"  Output:    {final}"
    )
    return {"messages": [AIMessage(content=body)]}


def graph_node_error(state: PathwayEmbedStateLG) -> dict:
    err = state.get("error") or "unknown error"
    return {"messages": [AIMessage(content=f"Pathway embedding build halted: {err}")]}


# ---------- conditional router ----------

def _route_or_continue(next_step: str):
    def router(state: PathwayEmbedStateLG) -> str:
        return "error_node" if state.get("error") else next_step
    return router


# ---------- graph builder ----------

def build_graph() -> StateGraph:
    workflow = StateGraph(PathwayEmbedStateLG)

    workflow.add_node("collect_config",              graph_node_collect_config)
    workflow.add_node("fetch_gene_pathways",         graph_node_fetch_gene_pathways)
    workflow.add_node("fetch_pathway_names",         graph_node_fetch_pathway_names)
    workflow.add_node("fetch_gene_names",            graph_node_fetch_gene_names)
    workflow.add_node("fetch_pathway_descriptions",  graph_node_fetch_pathway_descriptions)
    workflow.add_node("embed_with_biobert",          graph_node_embed_with_biobert)
    workflow.add_node("map_to_ensp",                 graph_node_map_to_ensp)
    workflow.add_node("summarize",                   graph_node_summarize)
    workflow.add_node("error_node",                  graph_node_error)

    workflow.set_entry_point("collect_config")
    workflow.add_edge("collect_config", "fetch_gene_pathways")

    for prev, nxt in [
        ("fetch_gene_pathways",        "fetch_pathway_names"),
        ("fetch_pathway_names",        "fetch_gene_names"),
        ("fetch_gene_names",           "fetch_pathway_descriptions"),
        ("fetch_pathway_descriptions", "embed_with_biobert"),
        ("embed_with_biobert",         "map_to_ensp"),
        ("map_to_ensp",                "summarize"),
    ]:
        workflow.add_conditional_edges(
            prev,
            _route_or_continue(nxt),
            {nxt: nxt, "error_node": "error_node"},
        )

    workflow.add_edge("summarize",  END)
    workflow.add_edge("error_node", END)
    return workflow


# ---------- runner + CLI ----------

def run(cfg: PathwayEmbedConfig) -> dict:
    workflow = build_graph()
    graph = workflow.compile(checkpointer=MemorySaver())
    final = graph.invoke(
        {"config": cfg},
        config={"configurable": {"thread_id": f"pathway_{cfg.gene_subset or 'all'}"}},
    )
    if final.get("messages"):
        print("\n=== Pathway embedding summary ===")
        print(final["messages"][-1].content)
    return final


def _parse_args(argv: list[str]):
    import argparse
    p = argparse.ArgumentParser(description="Build BioBERT pathway embeddings for STRING/PPI proteins.")
    p.add_argument("--output-dir", default="data/pathway", help="Directory for stage outputs (default: data/pathway/)")
    p.add_argument("--gene-subset", default="50",
                   help="Number of genes to process (int) or 'all' for full ~20k human genes (default: 50)")
    p.add_argument("--rebuild", action="store_true", help="Delete all stage outputs and start fresh")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds between KEGG/mygene requests (default: 0.5)")
    return p.parse_args(argv)


def main():
    import shutil, sys
    args = _parse_args(sys.argv[1:])
    out_dir = Path(args.output_dir)

    if args.rebuild and out_dir.exists():
        print(f"--rebuild: removing {out_dir}")
        shutil.rmtree(out_dir)

    subset: int | None = None if args.gene_subset == "all" else int(args.gene_subset)
    cfg = PathwayEmbedConfig(
        output_dir=out_dir,
        gene_subset=subset,
        request_sleep_s=args.sleep,
    )
    print(f"Pathway embedding build → {out_dir}")
    print(f"  gene_subset={subset if subset is not None else 'all'}, sleep={args.sleep}s\n")
    run(cfg)


if __name__ == "__main__":
    main()

