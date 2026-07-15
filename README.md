[# Biocabinet

[![PyPI version](https://img.shields.io/pypi/v/biocabinet.svg)](https://pypi.org/project/biocabinet/)
[![Python](https://img.shields.io/pypi/pyversions/biocabinet.svg)](https://pypi.org/project/biocabinet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/ishitajain9717/biocabinet)

An agentic transcriptomics analysis platform built on [LangGraph](https://github.com/langchain-ai/langgraph).
It orchestrates FastQC → trimming → alignment → quantification → normalisation → differential expression → GNN-based PPI enrichment, with an optional RAG layer that grounds LLM summaries in KEGG and Reactome pathway knowledge.

The pipeline is **agentic at the decision points, not just the prose**: an LLM (with deterministic guardrails) decides whether a sample passes FastQC, what Trimmomatic steps to apply, what experiment type an input directory contains, and — in the spatial pipeline — whether suspicious clusters are a real cell type or an imaging batch effect. Every LLM decision is validated against hard metrics and falls back to rule-based logic if the model is unavailable or unreliable.

---

## Architecture

```
Orchestrator (LangGraph)
│
├── Bulk RNA-seq sub-pipeline
│   ├── FastQC  →  Trimmomatic  →  STAR  →  featureCounts
│   ├── Normalisation  (TPM / FPKM / RPKM)
│   ├── Differential Expression  (PyDESeq2)
│   └── Summarise  ← RAG-augmented (Phase 4a)
│
├── scRNA-seq sub-pipeline
│   ├── Load  (h5ad / 10x / pbmc3k)
│   ├── QC  →  Filter  →  Normalise  →  PCA
│   ├── Cluster  (Leiden / UMAP)
│   └── Marker genes  →  Summarise  ← RAG-augmented
│
├── Spatial (imaging) sub-pipeline  (in development — Vizgen/MERSCOPE)
│   ├── Load  (squidpy Vizgen cell tables)
│   ├── QC  (blank-FDR, per-cell counts/volume, FOV outliers)
│   ├── Cluster  (PCA → Leiden)
│   └── FOV-bias check  ← agentic (metrics + LLM + Harmony correct/re-cluster loop)
│
├── Enrichment sub-pipeline  (auto-chained after bulk)
│   ├── Load PPI graph  (STRING / SHS27k)
│   ├── GNN training  (GIN, BFS split)
│   ├── Evaluation  (test1/2/3 buckets)
│   └── Inference  →  Summarise  ← RAG-augmented
│
└── RAG Q&A session  (Phase 4b — interactive, after final report)
    └── Retriever  →  LLM synthesis  →  Citations
```

Graph diagrams (Mermaid + PNG) live in `docs/`.

---

## Installation

```bash
git clone https://github.com/ishitajain9717/biocabinet.git
cd biocabinet
python -m venv .venv && source .venv/bin/activate

# Option A: editable install (recommended — registers the `rnaseq-agent` CLI)
pip install -e ".[llm]"

# Option B: plain requirements files
pip install -r requirements.txt -r requirements-llm.txt
```

### Python dependencies (all modalities)

Installed via `pip install -e ".[llm]"` from `pyproject.toml`. These are **imported in code** — no separate binary on `PATH`.

| Package | Used by | Role |
|---|---|---|
| `langgraph`, `langchain-core` | Orchestrator, all graphs | Agent workflow + state |
| `pandas`, `numpy`, `scipy` | Bulk, scRNA, enrichment | Tables and numerics |
| `pydeseq2` | Bulk | Differential expression (DESeq2) |
| `scanpy` | scRNA, Spatial | Load h5ad/10x, QC, normalize, PCA, cluster, markers |
| `squidpy` | Spatial | Read MERSCOPE/Vizgen cell tables into AnnData |
| `harmonypy` | Spatial | FOV batch correction in the bias-check loop |
| `python-dotenv` | All | Auto-load LLM config from `.env` |
| `torch`, `torch-geometric` | Enrichment | GNN training and inference |
| `transformers` | Enrichment, RAG | BioBERT pathway text embeddings |
| `fair-esm` | Enrichment | ESM-2 protein sequence embeddings |
| `mygene` | Bulk (DEG → enrichment) | ENSG → ENSP mapping |
| `requests`, `tenacity` | Enrichment (pathway build) | KEGG / Reactome / WikiPathways APIs |
| `langchain-openai`, `langchain-ollama` | Summarise, RAG (optional) | LLM summaries and Q&A |

scRNA clustering (Leiden / UMAP) runs inside Scanpy. If Leiden fails at runtime, install graph backends:

```bash
pip install leidenalg python-igraph
```

---

## Requirements by modality

### Bulk RNA-seq

**Workflow:** FASTQ → FastQC → **agentic QC gate** → **agentic trim gate** (LLM picks Trimmomatic steps from the FastQC report) → STAR → featureCounts → TPM/FPKM/RPKM → PyDESeq2 → summarise (+ RAG).

The QC gate lets the agent **drop a failing sample and continue** with the rest of the batch rather than aborting the run; the trim gate generates sample-specific Trimmomatic arguments, validated against the FastQC metrics before execution.

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/) | QC | `brew install fastqc` |
| | [STAR](https://github.com/alexdobin/STAR) | Alignment | `brew install star` |
| | [featureCounts](https://subread.sourceforge.net/) (Subread) | Gene counts | `brew install subread` |
| | [Trimmomatic](http://www.usadellab.org/cms/?page=trimmomatic) | Adapter/quality trim | JAR + `java`; steps chosen by the agentic trim gate |
| | `java` | Runs Trimmomatic | Required only if trimming is enabled |
| **Python (`pip`)** | `pydeseq2`, `pandas`, `mygene` | DEG + pair export for enrichment | In `pyproject.toml` |
| **Reference data** | STAR genome index + GTF | Not in git | See [`data/README.md`](data/README.md) — GENCODE GRCh38 + `STAR --runMode genomeGenerate` |
| **Input** | Paired FASTQ + sample sheet | | Conditions for DESeq2 (≥2 per group) |
| **Optional** | LLM env vars | RAG-augmented summary | See [LLM configuration](#llm-configuration) |

---

### scRNA-seq

**Workflow:** Load (h5ad / 10x / pbmc3k) → QC metrics → filter → normalize → HVG → PCA → neighbors → Leiden → UMAP → marker genes → summarise (+ RAG).

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | *(none)* | Entire pipeline is Scanpy in Python |
| **Python (`pip`)** | `scanpy`, `numpy`, `scipy`, `pandas` | Core analysis | In `pyproject.toml` |
| | `leidenalg`, `python-igraph` | Leiden clustering | Often needed; install if clustering errors |
| **Input** | `.h5ad`, 10x folder, or `pbmc3k` | | Set in `ScrnaConfig` / interactive prompts |
| **Optional** | `data/pbmc3k_raw.h5ad` | Cached demo data | Download via Scanpy; see `data/README.md` |
| **Optional** | LLM + RAG index | Marker-gene pathway context in summary | Build RAG index once; see RAG section |

---

### Spatial transcriptomics (imaging-based)

**Status:** in active development. The v1 scope targets **imaging-based** platforms — **MERSCOPE / Vizgen** — starting from the post-segmentation cell tables (segmentation is the user's responsibility). The node-level building blocks are implemented and tested; final assembly into the orchestrated LangGraph (`scripts/spatial/graph.py`) is still in progress, so the orchestrator currently routes spatial inputs to a stub.

**Planned workflow:** Load (squidpy Vizgen reader) → QC (blank-FDR, per-cell counts/genes/volume, FOV outliers) → normalize → cluster (PCA → Leiden) → **FOV-bias check** → annotation (CeLLama-style, reimplemented in Python) → summarise (+ RAG).

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | *(none)* | Entire pipeline is Python |
| **Python (`pip`)** | `squidpy`, `scanpy`, `harmonypy` | Read tables, QC/cluster, batch correction | In `pyproject.toml` |
| | `leidenalg`, `python-igraph` | Leiden clustering | Install if clustering errors |
| **Input** | `*_cell_by_gene.csv` + `*_cell_metadata.csv` | MERSCOPE per-region exports | Auto-detected in the input dir |
| **Optional** | LLM env vars | Agentic FOV-bias adjudication | Deterministic rule fallback without LLM |

**The FOV-bias check** is the spatial pipeline's flagship agentic step. Imaging-based data is captured tile-by-tile (each tile = a *field of view*, FOV), and per-FOV imaging differences can split one real cell type into several clusters. The node:

1. Computes deterministic evidence — **iLISI** (do expression-neighbors mix across FOVs?), **Cramér's V** (is cluster identity explained by FOV?), and the **fraction of cells in FOV-pure clusters**.
2. Hands that evidence to an LLM, which returns `BATCH_EFFECT` / `BIOLOGY` / `UNCERTAIN` plus an action — with **guardrails** that override the model in both directions (never silently keep an obvious artifact, never over-correct clean biology) and a rule-based fallback.
3. If a batch effect is confirmed, runs **Harmony** correction on the FOV label, re-clusters, and **re-verifies** that the split actually collapsed before accepting the result.

---

### Enrichment (GNN-PPI)

**Workflow:** Load PPI graph + ESM-2 + pathway embeddings → train GIN (BFS split) → bucketed eval → inference on DEG pairs. Auto-chained after a successful bulk run from the orchestrator.

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | *(none)* | Training and inference are PyTorch |
| **Python (`pip`)** | `torch`, `torch-geometric`, `transformers`, `fair-esm` | Model + embeddings | In `pyproject.toml` |
| **Reference data** | `data/ppi_SHS27k.tsv`, `data/esm2_embeddings_SHS27k.pt`, `data/pathway/06_pathway_embeddings_combined.pt` | SHS27k benchmark | See `data/README.md` |
| **Upstream** | Bulk DEG `deg_pairs.tsv` | Optional | Orchestrator passes path when bulk DEG ran |
| **Optional** | GPU (`device=cuda` / `mps`) | Faster training | Default config uses `auto` |

Pathway embedding build (one-time, before merge):

```bash
python -m scripts.enrichment.pathway_embed
python -m scripts.enrichment.reactome_embed
python -m scripts.enrichment.wikipathways_embed
python -m scripts.enrichment.merge_pathway_embeddings
python -m scripts.enrichment.precompute_esm
```

---

### RAG (pathway Q&A)

**Workflow:** Build index once → retriever (BioBERT cosine search) → answerer (LLM + citations) → augment pipeline summaries (4a) → optional interactive chat after orchestrator (4b).

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | *(none)* | Search and embed in Python |
| **Python (`pip`)** | `transformers`, `torch` | BioBERT at index build and query time | In `pyproject.toml` |
| | `langchain-openai` or `langchain-ollama` | Prose answers | Optional; deterministic fallback without LLM |
| **Reference data** | KEGG + Reactome text under `data/pathway/` | Source for `build_index` | Built by enrichment pathway scripts |
| **Generated** | `data/rag/docs.jsonl`, `embeddings.npy`, `gene_to_doc_ids.json` | Local library | Not committed; run `build_index` |

```bash
python -m scripts.rag.build_index          # once, after pathway data exists
python -m scripts.rag.retriever "cell cycle CDK4"   # smoke-test retrieval
python -m scripts.rag.graph                  # interactive Q&A
```

Bulk runs can set **pathway interests** at config time (e.g. `cell cycle, apoptosis`); matching DEGs are always included in RAG gene selection before the top-60% rule.

---

### Orchestrator

| Type | Requirement | Notes |
|---|---|---|
| **Python (`pip`)** | All of the above for chosen modality | Lazy-imports child graphs |
| **Persistence** | `pipeline_runs.sqlite` | LangGraph `SqliteSaver`; created on first run |
| **LLM** | `OLLAMA_MODEL` or `OPENAI_API_KEY` | Final report + RAG chat; optional |

The orchestrator **auto-detects the modality** from the input path (FASTQ/sample-sheet → bulk, 10x/h5ad → scRNA, Vizgen CSVs or spatial-coordinate h5ad → spatial) via `scripts/common/data_detect.py`, shows the evidence, and asks for confirmation rather than presenting a blind menu. A confirmed `bulk_rnaseq` run chains bulk → enrichment (if bulk succeeds) → final report → optional RAG Q&A; `scrna` runs scRNA only, then report and optional RAG.

---

## Data setup

See [`data/README.md`](data/README.md) for instructions on downloading the required reference files, PPI graph, and pre-computed embeddings.

---

## Running

### Full orchestrated pipeline

```bash
python -m scripts.orchestrator
# resume a previous run
python -m scripts.orchestrator --thread-id orchestrator_20260520_094200
```

### Individual sub-pipelines

```bash
# Bulk RNA-seq
python -m scripts.bulk_rnaseq.graph

# scRNA-seq
python -m scripts.scrna.graph

# Enrichment (GNN-PPI)
python -m scripts.enrichment.graph

# Build RAG index (run once)
python -m scripts.rag.build_index

# Interactive RAG Q&A
python -m scripts.rag.graph
```

---

## LLM configuration

The pipeline uses an LLM for plain-English summaries and RAG-grounded answers.
Configure via environment variables — no API keys in code:

```bash
# Option A: local Ollama (free, offline)
export OLLAMA_MODEL=llama3

# Option B: OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional, default is gpt-4o-mini
```

These can also be placed in a `.env` file at the repo root — it is auto-loaded via `python-dotenv`. If neither provider is set, all summarise nodes **and every agentic gate** (FastQC, trimming, data-type detection, FOV-bias) fall back to deterministic rule-based logic.

---

## Project layout

```
scripts/
├── orchestrator.py          top-level LangGraph orchestrator
├── bulk_rnaseq/             bulk RNA-seq sub-pipeline
│   ├── config.py            PreprocessingConfig dataclass + CLI prompts
│   ├── nodes.py             core bioinformatics functions
│   ├── graph_nodes.py       LangGraph wrapper nodes
│   ├── graph_state.py       PipelineState TypedDict
│   └── graph.py             graph builder
├── scrna/                   scRNA-seq sub-pipeline  (same structure)
├── spatial/                 spatial (imaging / Vizgen) sub-pipeline — in development
│   ├── config.py            SpatialConfig (Vizgen CSVs + QC thresholds)
│   ├── nodes.py             load_vizgen, spatial_qc, clustering, fov_bias_check
│   ├── graph_nodes.py       LLM FOV-bias adjudication + guardrails
│   └── graph.py             graph builder (assembly in progress)
├── enrichment/              GNN-PPI enrichment sub-pipeline
│   ├── pathway_embed.py     KEGG pathway embeddings (BioBERT)
│   ├── reactome_embed.py    Reactome pathway embeddings
│   ├── wikipathways_embed.py WikiPathways embeddings
│   ├── merge_pathway_embeddings.py  multi-source merge + dedup
│   ├── precompute_esm.py    ESM-2 protein sequence embeddings
│   ├── gnn_data.py          PPI graph + feature assembly
│   ├── gnn_model.py         GIN model definition
│   ├── gnn_train.py         training loop (BFS split)
│   ├── gnn_test.py          bucketed evaluation (test1/2/3)
│   └── inference.py         predict interactions for novel pairs
├── rag/                     RAG layer
│   ├── build_index.py       Phase 1 — build pathway document library
│   ├── retriever.py         Phase 2 — cosine similarity search
│   ├── answerer.py          Phase 3 — LLM synthesis with citations
│   ├── augment.py           Phase 4a — auto-augment pipeline summaries
│   ├── graph_state.py       Phase 4b — RagChatState
│   ├── artifact_reader.py   scan a run dir for STAR/featureCounts/FastQC/DEG/GNN artifacts
│   ├── graph_nodes.py       Phase 4b — chat loop nodes (+ LLM data-type inference)
│   └── graph.py             Phase 4b — interactive Q&A graph
└── common/
    ├── node_result.py       NodeResult dataclass (shared across pipelines)
    └── data_detect.py       auto-detect modality (bulk / scRNA / spatial) from a path
```

---

## Checkpointing and resumption

All pipeline state is persisted to `pipeline_runs.sqlite` via LangGraph's `SqliteSaver`.
If a run is interrupted, resume it by passing the printed `--thread-id`.
Child pipelines (bulk, enrichment) share the same checkpointer as the orchestrator, so partial child runs are also resumable.

---

## License

MIT
](https://pypi.org/project/biocabinet/)
